# Flex-π：從輸入到 32 個 action 的逐元素推導

日期：2026-10-12｜作者：GPT｜類型：原始碼核對、數學推導、教學算例

**32 個 action 是 32 個未來時間位置，每個位置含 14 個數值，因此輸出是 `32×14`，共 448 個數值。模型先編碼觀測，再由高斯雜訊開始，重複計算速度場及 Euler 更新，最後反正規化成機器人命令。**

這份文件寫到每個運算可以展開成加、乘、除、平方根、指數及三角函數；給定權重與輸入即可逐元素代入。真實模型含約 60 億參數，手算完整模型在實務上不切實際，但不存在必須靠「Transformer 黑盒」才能理解的步驟。[手算範例](worked-example.md)另外提供全部已知的小型模型，完整生成 32 列動作。

## 文件與證據範圍

| 文件 | 內容 |
|---|---|
| 本文件 | 全流程、張量形狀、30 層運算、mask、4 步更新、動作反正規化 |
| [encoders.md](encoders.md) | RGB resize、Wan VAE 的每個子模組、DINOv3 12 層、文字編碼器 |
| [worked-example.md](worked-example.md) | 明列全部權重與輸入的小型 32-step 手算例 |
| [analysis/handcalc_reference.py](analysis/handcalc_reference.py) | 只用 Python 標準函式庫的逐元素算例與獨立矩陣式交叉驗算 |
| [results/handcalc_reference.json](results/handcalc_reference.json) | 算例每輪結果、第一列 block trace、32 列最終輸出 |
| [本輪 HUMAN.md](../../GPT/updates/2026-10-12-flexpi-hand-calculation/HUMAN.md) | 人類精華摘要 |

研究固定於模型 repo [commit `20c1b2b71ea35a415d5d47c39b04443cfadad7a1`](https://github.com/MaSuo1032/flex-pi/tree/20c1b2b71ea35a415d5d47c39b04443cfadad7a1)，研究 repo 起點 `9f85b33065459782b3afcc57eddf394bbdb487c3`。DINO 依模型 repo 的 `pyproject.toml` 使用 **timm 1.0.26**，不能拿其他版本的普通 ViT 公式替代。

以下主線使用 RoboTwin 三相機、14 維 action/state、`tshape_robotwin_384x320_uniform`、動作長度 32、去噪 4 輪、Euler、無量化、無 step-skip、文字 guidance scale=1。先詳述所有觀測都存在的 action-only，再詳述設定預設的 full-joint。它們共用 checkpoint，但輸出值通常不同。

本輪做的是原始碼核對及教學算例驗算，沒有 checkpoint、真實觀測或 GPU 推論。文件內的示例值不是 Flex-π 訓練權重的預測，也不是模型成功率或效能實測。

## 0. 要實際算出一筆真實結果，需要哪些已知數？

請把以下資料視為函數的輸入，而不是從架構自動推得的常數：

1. **同一 checkpoint 的所有權重**：視覺主幹、動作專家、DINO 投影器、pointmap 投影器/輸出頭、proprio encoder；凍結的 VAE、DINO 預訓練權重也需要。每層權重不同，不能把一層算完的結果複製 30 次。
2. **解析後的 config**：除了基本 model YAML，還包含 checkpoint 的 `config.yaml`、task override、部署 override。本文的 14 維、147 個 DINO token 不是任意 checkpoint 都相同。
3. 三張 RGB、三張深度、對應各深度原始解析度的相機內參、14 維當前關節狀態。
4. 文字 `context[128,4096]` 和 `context_mask[128]`；若由文字字串開始，還需 tokenizer 檔與 umT5 權重，見附錄。
5. `dataset_stats.json` 中 **state 與 action 各自**的 mean/std。不能用 state 統計值反正規化 action。
6. 初始動作雜訊 `A^(0)[32,14]`。joint 模式還需各未來視覺雜訊。僅記錄 seed 不足以跨 CPU/GPU、版本保證同一個陣列，應保存實際抽出的值。
7. 要比對電腦最後幾位，還需 dtype、kernel、累加/捨入規則、resize 實作版本。本文公式以實數算術描述；BF16 執行會在中間運算捨入。

架構和參數數量只能決定「如何計算」，不能決定特定指令下的 448 個輸出值。

## 1. 記號、方向與基礎運算

省略 batch 維 `B=1`。`i` 是 token/動作時間列，`d` 是特徵欄，`ℓ=0…29` 是網路層，`k=0…3` 是去噪輪。**網路深度 ℓ、去噪輪 k、動作時間 i 是三個不同索引。**

所有 `Linear(in,out)` 的 PyTorch 權重儲存為 `W[out,in]`：

```text
Linear(x;W,b)[i,o] = b[o] + Σ_(d=0…in-1) W[o,d] x[i,d]
```

不能把 `xW` 和 `xWᵀ` 的權重方向混用。以下所有線性層都按這個規則，未說無 bias 的層都有 bias。

逐元素非線性：

```text
SiLU(x) = x / (1 + exp(-x))
GELU_tanh(x) = 0.5 x [1 + tanh(√(2/π) (x + 0.044715 x³))]
GELU_exact(x) = 0.5 x [1 + erf(x/√2)]
```

Flex-π DiT 的 FFN/文字投影使用 `GELU_tanh`；DINOv3 的預設 MLP GELU 使用 `GELU_exact`，不要混淆。

LayerNorm 對每一列的 **D 個特徵**做：

```text
μ_i = (1/D) Σ_d x[i,d]
v_i = (1/D) Σ_d (x[i,d]-μ_i)²       # biased variance，除 D，不是 D-1
LN_ε(x)[i,d] = (x[i,d]-μ_i) / √(v_i+ε)
LN_affine(x)[i,d] = γ[d] LN_ε(x)[i,d] + β[d]
```

DiT `norm1`、`norm2` 無 γ/β；`norm3` 有 γ/β，ε=10⁻⁶。DINO 輸入到共享主幹前的 `dino_feature_norm` 有 γ/β，PyTorch 預設 ε=10⁻⁵。

DiT Q/K 的 RMSNorm：

```text
RMS_ε(x)[i,d] = g[d] x[i,d] / √((1/D) Σ_e x[i,e]² + ε)
```

**這裡 D=3072，先跨全部 3072 維算，再拆 24 heads；不是各 head 的 128 維獨立 RMSNorm。** `v` 不做 RMSNorm。

## 2. 先把實際輸入整理好

### 2.1 RGB：相機順序、大小與像素值

使用 per-camera 路徑時：head 原始 RGB resize 成 `256×320`；兩個 wrist 各 resize 成 `224×224`。均使用 antialiased bilinear，uint8 先除 255，再轉成 `2p-1`。完整 resize 求和公式見 [encoders.md §1](encoders.md#1-rgb-resize-與排列)。

VAE 的 RGB composite 使用 **H=384、W=320**：

| 相機 | composite 列範圍 | composite 欄範圍 | slot 尺寸 |
|---|---|---|---|
| head | 0…255 | 0…319 | 256×320 |
| left wrist | 256…383 | 0…159 | 128×160 |
| right wrist | 256…383 | 160…319 | 128×160 |

wrist 的 `224×224` 還需 antialiased resize 成 slot 大小。DINO 使用 per-camera 張量時讀原本的 `224×224` wrist，不從縮小的 composite 再放大。若 `per_cam=None`，DINO 才從 composite 裁切後 resize；兩條路徑的像素不保證相同。

`input_image` 最終形狀 `[1,3,384,320]`，範圍 `[-1,1]`。不要把文件中簡寫的 `384×320` 誤讀成寬 384、高 320。

### 2.2 深度與 3D 點雲

每臺相機深度是 uint16 毫米，先 `z=depth/1000` 得公尺。對原始深度網格的像素 `(v,u)`，內參為 `fx,fy,cx,cy`：

```text
X = (u-cx) z / fx
Y = (v-cy) z / fy
Z = z
```

有效條件是 `0.01 < z < 2.0`；不符合時，三座標一起置 0。接著先裁切再正規化：

```text
X̃ = 2 [clip(X,-0.5,0.5)+0.5] - 1
Ỹ = 2 [clip(Y,-0.5,0.5)+0.5] - 1
Z̃ = 2 clip(Z,0,1.5)/1.5 - 1
```

因此無效深度變成 `(0,0,-1)`，不是三通道全 0。這些座標在 **各相機座標系**，此模組沒有用外參轉到共用世界座標系。

先在原生深度網格反投影，再 nearest resize 點雲到上述 composite slots。不能先縮深度、卻繼續使用原網格 K。範例：`u=170,v=130,cx=160,cy=120,fx=fy=200,z=1` ⇒ `(X,Y,Z)=(0.05,0.05,1)` ⇒ 正規化 `(0.1,0.1,1/3)`。

### 2.3 關節狀態

RoboTwin `observation['joint_action']['vector']` 是 14 維。對每一欄 j：

```text
p[j] = clip((state[j]-μ_state[j])/(σ_state[j]+10⁻⁸), -5, 5)
```

`action_state_transforms=null`、原生 `ConcatLeftAlign`、14 維 output 時，不另做旋轉表示或相對動作變換。其他 checkpoint 可能不同。

### 2.4 文字與 proprio 合成 context

若已給 `C_text[128,4096]`，直接使用。若給 prompt，先計算 umT5；`sim_robotwin.yaml` 覆寫 `load_text_encoder=true`，部署程式在指令改變時編碼並緩存。因此 **不能把 model YAML 的 false 解讀成所有部署都不跑 T5**。

Proprio encoder 是 `Linear(14,4096)`：

```text
C_state[0,o] = b_state[o] + Σ_(j=0…13) W_state[o,j] p[j]
C = concat_rows(C_text, C_state)       # [129,4096]
m_ctx = concat(context_mask, [True])   # [129]
```

文字順序在前，state token 在最後。mask 必須保存真實內容；不是看到 128 長度就假設 128 個有效詞。`encode_prompt` 將 tokenizer padding 對應的 embedding 清零，然後把回傳 mask 全置 True；這些零向量仍會經過帶 bias 的投影。直接載入預計算 context 時則使用提供的 mask。

## 3. 三種觀測如何變成 387 個 token

### 3.1 RGB → VAE latent → 120 個 RGB token

RGB 加一個時間維：`[1,3,1,384,320]`。Wan2.2 VAE 的完整 residual/卷積/downsample 運算在 [encoders.md §2](encoders.md#2-wan22-vae-encoder逐模組計算)。它輸出確定性的 posterior mean 經標準化：

```text
Z_rgb[48,1,24,20]
```

它不在觀測編碼時再抽一個 VAE latent 雜訊。空間縮 16 倍；latent H=24、W=20。

`video_expert.patch_embedding` 是 `Conv3d(48,3072,kernel=(1,2,2),stride=(1,2,2))`。對 `r=0…11,c=0…9,o=0…3071`：

```text
T_rgb[10r+c,o]
  = b_patch[o]
  + Σ_(d=0…47) Σ_(a=0…1) Σ_(b=0…1)
      W_patch[o,d,0,a,b] Z_rgb[d,0,2r+a,2c+b]
```

結果 `[120,3072]`。flatten 順序是時間→列→欄，欄變化最快。這是學習到的 192→3072 投影，不是只把小 patch 攤平。

### 3.2 點雲 → 同一 VAE → 獨立 patch projector

正規化 XYZ composite 同樣是 `[1,3,1,384,320]`，使用 **同一個 VAE、同一套 VAE 權重**得到 `Z_pt[48,1,24,20]`。

再使用 `pt_patch_embedding` 依 §3.1 公式得到 `T_pt[120,3072]`。它初始化時 deepcopy RGB patch embedding，但之後是獨立的可訓練參數；不能在推論時假設兩份權重相等。

### 3.3 DINO → 147 個語意 token

各相機 resize 成 `224×224`，ImageNet normalize，再由 timm DINOv3 ViT-B/16 的 12 層得到 `[196,768]` patch features。完整 prefix token、RoPE、LayerScale、FFN 公式見 [encoders.md §3](encoders.md#3-dinov3-vit-b16完整的12層)。

每臺相機 14×14 網格做 **pixel-unshuffle**：

```text
F_cam[7r+c, 4d+2a+b] = D_cam[14(2r+a)+(2c+b), d]
r,c=0…6；a,b=0,1；d=0…767
```

得到 `[49,3072]`。沒有把四個 patch 平均；各自值搬到特徵欄。依 head→left→right concat 得 `[147,3072]`。

再計算：

```text
T_dino = Linear(LN_affine_1e-5(F); W_dino_embed,b_dino_embed)
```

結果 `[147,3072]`。fold、LN、投影的順序不能交換。

### 3.4 Action-only 的排列

```text
X_v^(0) = [T_rgb ; T_dino ; T_pt] ∈ R^(387×3072)
RGB   index   0…119
DINO  index 120…266
PT    index 267…386
```

觀測 token 都標記去噪時間 `t=0`。它們不是 387 個未來預測，而是目前觀測的條件。

## 4. 去噪時間與位置：先把可代入的常數算好

### 4.1 4 輪 action schedule

`N_train=1000` 是時間 embedding 的縮放常數，不是推論要跑 1000 步。推論步數 `K=4`：

```text
u_k = 1-k/4, k=0…4
φ_s(u) = su / [1+(s-1)u]
σ_k = φ_s(u_k)
t_k = 1000σ_k, k=0…3
Δ_k = σ_(k+1)-σ_k
```

action 預設 s=1：

| k | σ_k | 傳入 action 網路的 t_k | Δ_k |
|---|---:|---:|---:|
| 0 | 1 | 1000 | -0.25 |
| 1 | 0.75 | 750 | -0.25 |
| 2 | 0.5 | 500 | -0.25 |
| 3 | 0.25 | 250 | -0.25 |
| 終點 | 0 | 不另跑一次網路 | — |

**Δ 為負**。速度的訓練目標是 `noise-clean`；所以從雜訊往乾淨資料走是減去四分之一的預測速度。不要改成 `A+0.25v`。

### 4.2 時間 embedding 與 6 組調變

對任意時間 t：

```text
ω_j = 10000^(-j/128), j=0…127
e(t)[j] = cos(tω_j)
e(t)[128+j] = sin(tω_j)                 # [256]
τ_e(t) = Linear_2(SiLU(Linear_1(e(t)))) # [D_e]
M_e(t) = reshape(Linear_3(SiLU(τ_e(t))), [6,D_e])
```

視覺 e=v：`D_v=3072`；action e=a：`D_a=1024`。兩個專家各有自己的權重。`t=0` 的 e 前128個值是1、後128個是0，因此 **t=0不代表 time embedding 或 gate 為0**。

各層有自己的已訓練 `modulation[6,D_e]`：

```text
[shift_sa,scale_sa,gate_sa,shift_ff,scale_ff,gate_ff]
  = block_ℓ.modulation + M_e(t)
```

gate 是直接的實數向量，沒有 sigmoid，可能負值或大於1。`τ`、`M` 是 timestep MLP 的輸出，6 組中的各組都是 D_e 維。

### 4.3 Action RoPE

每個 head 為128維，64個相鄰二元對。對動作列 i=0…31：

```text
θ_(i,j) = i · 10000^(-2j/128), j=0…63
Rθ(x_2j,x_(2j+1))
 = (x_2j cosθ - x_(2j+1) sinθ,
    x_2j sinθ + x_(2j+1) cosθ)
```

每個 head 用同樣的位置頻率；i 是未來動作位置，不是去噪輪 k。

### 4.4 視覺 RoPE

每個128維 head分為 `(time=44,row=42,col=42)`：22、21、21個 complex pairs。各分段**分別**用 `θ=p·10000^(-2j/D_segment)`；不能對整個128維使用相同分母。

RGB/PT token `(f,r,c)` 的位置是 latent patch 網格：action-only f=0,r=0…11,c=0…9。DINO fold後的region會同步除2：

```text
head：      row=0…6, col=0…6
left wrist：row=7…13,col=0…6
right wrist：row=7…13,col=7…13
```

所以 DINO i=0…48 對 `(0,r,c)`；i=49…97 對 `(0,7+r,c)`；i=98…146 對 `(0,7+r,7+c)`。不同串流可以有部分相同RoPE座標，這是程式配置，不應自行重新編號成0…386的一維位置。

## 5. 一個 DiT block 的每個計算：30 層都重複此規則

本節是一個完全展開的「積木」，後面 prefill/action/joint 都使用它，只改專家權重、時間與可見key集合。

專家 e 的輸入 `X[N,D_e]`，視覺 `D_e=3072,F_e=14336`；動作 `D_e=1024,F_e=4096`。兩者注意力投影寬都為 `H d_h=24×128=3072`。

### 5.1 六組 shift/scale/gate

按 §4.2 計算 ℓ 層的六組向量。action所有32列使用同一輪時間；joint視覺各token的時間可不同。

### 5.2 Self/mixed-attention 輸入

```text
U[i,d] = LN_1e-6(X)[i,d] (1+scale_sa[i,d]) + shift_sa[i,d]
Q_raw = Linear(U;W_q,b_q)    # [N,3072]
K_raw = Linear(U;W_k,b_k)
V     = Linear(U;W_v,b_v)
Q = RoPE(RMS_1e-6(Q_raw))
K = RoPE(RMS_1e-6(K_raw))
```

LayerNorm沒有affine；Q/K RMS的g來自 `self_attn.norm_q.weight`、`norm_k.weight`。reshape爲 `Q[i,h,d]=Q[i,128h+d]`，h=0…23,d=0…127。

### 5.3 Attention 逐 head、逐 query、逐 key

令 `J_ℓ(i)` 是 mask允許query i讀取的key列集合，決定方法見 §6、§7、§10。

```text
s[i,h,j] = (1/√128) Σ_(d=0…127) Q[i,h,d] K[j,h,d]
m[i,h] = max_(j∈J_ℓ(i)) s[i,h,j]
α[i,h,j] = exp(s[i,h,j]-m[i,h]) /
           Σ_(j'∈J_ℓ(i)) exp(s[i,h,j']-m[i,h])
O[i,h,d] = Σ_(j∈J_ℓ(i)) α[i,h,j] V[j,h,d]
O_cat[i,128h+d] = O[i,h,d]
```

mask **True 表示允許**。不能先對所有key softmax後把禁止項乘0，因爲分母會錯。主線沒有全被禁止的query。所有head分別softmax，不對24heads一起softmax。

### 5.4 Self-attention 的 O 投影與 gated residual

```text
Y[i,d] = X[i,d] + gate_sa[i,d] Linear(O_cat;W_o,b_o)[i,d]
```

動作用 `W_o[1024,3072]`，視覺用 `[3072,3072]`。shortcut留的是本層原始X，不是已LN或調變的U。

### 5.5 Cross-attention：看文字＋當前state

先把 §2.4 的原始4096維context各自投影到專家寬度：

```text
C_e = Linear_2(GELU_tanh(Linear_1(C)))    # [129,D_e]
```

視覺和action的 `text_embedding` 權重不同。cross-attention的運算：

```text
B = LN_affine_1e-6(Y)                # block.norm3，帶γ/β
Qc = RMS_1e-6(Linear(B;W_cq,b_cq))   # [N,3072]
Kc = RMS_1e-6(Linear(C_e;W_ck,b_ck)) # [129,3072]
Vc = Linear(C_e;W_cv,b_cv)          # [129,3072]
Oc = Attention(Qc,Kc,Vc,m_ctx)      # 同§5.3，用√128
Yc = Y + Linear(Oc;W_co,b_co)
```

**Cross-attention不做RoPE，沒有額外gate。** 它的Q/K RMS也在完整3072維上做。文字mask按context列廣播給各query；只對有效列歸一化。context不更新，不把context當成第130個action。

### 5.6 FFN＋第二個 gated residual

```text
Uff[i,d] = LN_1e-6(Yc)[i,d] (1+scale_ff[i,d]) + shift_ff[i,d]
Hff[i,f] = GELU_tanh(b_1[f] + Σ_d W_1[f,d] Uff[i,d])
Rff[i,d] = b_2[d] + Σ_f W_2[d,f] Hff[i,f]
X_next[i,d] = Yc[i,d] + gate_ff[i,d] Rff[i,d]
```

視覺f=0…14335；action f=0…4095。沒有SwiGLU、沒有兩路MLP相乘、沒有把gate換成sigmoid。最終 `X_next` 才是下一層輸入。

## 6. Action-only：Prefill 30 層，固定目前世界

讓 `X_v^(0)` 是 §3.4 的387×3072矩陣，所有token用t=0，共用 `C_v[129,3072]`。

對 ℓ=0…29：

1. 用視覺 ℓ 層權重完成 §5.1–5.2，得到 `Q_v^ℓ,K_v^ℓ,V_v^ℓ`。
2. 保存 `K_cache[ℓ]=K_v^ℓ`、`V_cache[ℓ]=V_v^ℓ`。**存的是本層 attention 輸入投影出的K/V；不是本層post-block輸出再投影的值。** K已做RMS與RoPE，V沒有。
3. 依下表建立允許key集合，完成 §5.3–5.6，得到 `X_v^(ℓ+1)`。

| 層 ℓ | RGB query 可讀 | DINO query 可讀 | PT query 可讀 |
|---|---|---|---|
| 0…6 | RGB 120 | DINO 147 | PT 120 |
| 7…22 | 全部387觀測 | 全部387觀測 | 全部387觀測 |
| 23…29 | RGB 120 | DINO 147 | PT 120 |

0…6與23…29是HBridge outer layers，原因是 `int(30×0.25)=7`；剩16層融合。每層仍有文字cross-attention與FFN，outer不是把整層跳過。

注意第7層計算出K/V時，該層的跨串流 attention **尚未發生**；cross-stream訊息在post-block後進入第8層。最後的 `X_v^(30)` 不直接當動作；action讀取30層各自的K/V。

cache形狀：每份 `[30,1,387,3072]`。BF16 K+V為 `30×387×3072×2×2=142,663,680 bytes`。這是張量容量，不代表每次實際DRAM流量。

## 7. Action-only：每輪如何計算32×14的速度

### 7.1 初始動作不是由state複製而來

抽 `A^(0)[32,14] ~ N(0,1)`。它是未來action在正規化空間的初始樣本，不是目前state，不是已給定的32個動作。初始化一次後，4輪沿用並更新；不能每輪重抽。

### 7.2 每輪 k 的 action embedding

```text
X_a^(0,k)[i,o] = b_a_enc[o] + Σ_(j=0…13) W_a_enc[o,j] A^(k)[i,j]
```

得到 `[32,1024]`。按 §4.2以 `t_k` 算 `M_a(t_k)[6,1024]`；按 §4.3準備32個位置RoPE。用 `C_a[129,1024]`。原程式每輪重算context投影，但其數學值在本次呼叫內不變。

### 7.3 30層action decoder

每層都用 §5，只有self-attention key來源不同：

| 層 ℓ | Q | K/V排列 | query數×key數 |
|---|---|---|---|
| 0…6、23…29 | 本層32個action Q | 本層32個action K/V | 32×32 |
| 7…22 | 本層32個action Q | `[387 cached visual ; 32 current action]` | 32×419 |

所有32個action互相可見，**不使用LLM下三角因果mask**。例如第0個動作可以參照第31個動作的當輪表示。不同expert的hidden寬度不同，但投影後K/V都是3072維，才能concat。

每輪的30層hidden從 `Linear(A^(k))` 重新建立；上一輪留下的是更新後 `A^(k)`，不是直接把上一輪第30層hidden當新第0層。

### 7.4 最後線性head：1024→14

得到 `X_a^(30,k)[32,1024]` 後：

```text
v_a^(k)[i,j] = b_head[j] + Σ_(d=0…1023) W_head[j,d] X_a^(30,k)[i,d]
```

**實際 `ActionDiT.head` 是 `nn.Linear`，`post_dit`直接呼叫它。** 檔案雖另定義了帶LN與時間調變的`ActionHead`類別，本配置沒有使用它，不能把那段加入公式。head輸出是速度場v，不是已可執行的最終關節命令。

## 8. 4輪Euler更新，得到全部32個動作

對448個元素一起做：

```text
A^(k+1)[i,j] = A^(k)[i,j] + Δ_k v_a^(k)[i,j]
```

預設s=1展開：

```text
A^(1) = A^(0) - 0.25 v(A^(0),1000; observations,context)
A^(2) = A^(1) - 0.25 v(A^(1), 750; observations,context)
A^(3) = A^(2) - 0.25 v(A^(2), 500; observations,context)
A^(4) = A^(3) - 0.25 v(A^(3), 250; observations,context)
```

v每輪重算，不能把第一次v直接用四次；除非另啓用step-skip，此文件主線關閉。沒有抽詞概率、argmax、top-k；32列皆連續實數。

`infer_action` 回傳 `A^(4)[32,14]`，轉到CPU float32。這裏**還是正規化動作**。單列不能獨立從一張圖直算出來，因為attention依賴同一輪的其餘31列和觀測。

## 9. 反正規化，變成機器人控制命令

部署依序做 `merger.backward → normalizer.backward → 反向action transforms`。本RoboTwin配置下merger是同維concat/crop，transforms為null。因此：

```text
a_phys[i,j] = A^(4)[i,j] (σ_action[j]+10⁻⁸) + μ_action[j]
i=0…31，j=0…13
```

normalizer的forward有±5裁切，但 **backward沒有先裁切預測值**。不能自行加tanh、sigmoid、clip(-1,1)、夾爪threshold，或把action再加到state上。

此部署用 `task_env.take_action(action,action_type='qpos')`，所以是由dataset與機器人定義決定的關節/夾爪位置向量。14欄的語意順序必須以該資料集轉換與機器人定義核對，不能僅憑14就宣稱XYZ+歐拉角。

把前 `min(replan_steps,32)` 列依序放入queue，一次取一列送出。RoboTwin默認replan_steps=32；LIBERO配置可為10，所以「算32個」與「先執行32個」不是同一條件。

## 10. 預設full-joint：同時計算未來視覺時，哪些步驟改變？

`configs/model/flexpi.yaml` 的三個`joint_*`預設都true。只關掉`return_stream_latents`不會關掉視覺生成，只是不回傳其CPU副本。

### 10.1 初始化shape與排列

`num_video_frames=9`：從33個原始時間點按4倍頻率採樣成9個video frames，VAE壓成3個latent frames。推論只觀測第0幀，其餘兩latent frames是生成內容。

```text
RGB Z_v: [48,3,24,20]       # frame0=觀測；frame1,2=初始高斯雜訊
PT  Z_p: [48,3,24,20]
DINO F_d: [3072,3,147,1]
Action A: [32,14]
```

四個generator各自用同一seed重新建立，並不是讓一個generator連續抽完四串流。若要精確重現應保存四份實際初始張量。

每輪將latent重新投影成token，完整序列：

| 索引範圍，inclusive | 內容 | 數量 |
|---|---|---:|
| 0…119 | RGB觀測 | 120 |
| 120…359 | RGB未來 | 240 |
| 360…506 | DINO觀測 | 147 |
| 507…800 | DINO未來 | 294 |
| 801…920 | PT觀測 | 120 |
| 921…1160 | PT未來 | 240 |
| 1161…1192 | action | 32 |

共1193個token，其中387個觀測、774個未來視覺、32個action。視覺token `[1161,3072]`、action hidden `[32,1024]`，不是1193個token都經同一組權重。

### 10.2 Attention mask必須按query方向寫

outer層0…6、23…29：只看同一串流；視覺觀測只看同串流frame0；同串流未來可看該串流所有3幀；action看32個action。

中間層7…22，所有observations present、所有joint true：

| query種類 | 可讀keys |
|---|---|
| 任一視覺觀測 | 387個視覺觀測 |
| 任一未來視覺 | 全部1161個視覺token |
| action | 全部1161個視覺token＋32個action |

視覺 **任何時候都不讀action**；觀測不讀未來。未來兩幀互相可見，沒有未來時間下三角mask。跨串流雙向不等於視覺/動作雙向。

### 10.3 Action和視覺各自使用自己的時間排程

action s=1不變；RGB/DINO/PT s=6：

| k | 視覺σ_k | 視覺t_k | 視覺Δ_k |
|---|---:|---:|---:|
| 0 | 1 | 1000 | -1/19 ≈ -0.052632 |
| 1 | 18/19 | 18000/19 ≈ 947.368 | -12/133 ≈ -0.090226 |
| 2 | 6/7 | 6000/7 ≈ 857.143 | -4/21 ≈ -0.190476 |
| 3 | 2/3 | 2000/3 ≈ 666.667 | -2/3 |

每一串流的frame0 token時間始終爲0，未來token用本輪t。相同視覺time MLP共享於RGB、DINO、PT。若設置`sigma_shift` override，程式把相同override傳給各scheduler，以上默認表就不適用。

### 10.4 每輪30層mixed forward

用§5公式為視覺和action**各自**做Q/K/V投影。middle層將其concat做attention，隨後拆開輸出；視覺用視覺O/FFN，action用action O/FFN。所有串流共享同一`C_v`作視覺cross-attn，action用`C_a`。

Eager joint原始路徑每輪會重算觀測；KV-split優化則先算觀測cache，之後每輪只算774個future視覺與32個action。觀測不能看未來、t固定0，因此在實數算術下可緩存；不能把action-only輸出當joint輸出，因爲joint action也能讀future keys。

若按 `decode_joint_noisy` 手算，scatter不可用錯誤連續位置：

```text
anchor_pos = [0…119,360…506,801…920]
noisy_pos  = [120…359,507…800,921…1160]
```

把prefill cache和future K/V分別填回1161列視覺順序，再接32個action K/V。緩存array自身是packed387列，但完整key位置不是0…386。

### 10.5 輸出heads

Action仍是 §7.4 的1024→14線性速度頭。

RGB `Head` 對每個token：

```text
shift_h,scale_h = head.modulation[2,3072] + τ_v(t_i)廣播至兩組
B_i = LN_1e-6(X_v_final[i]) (1+scale_h[i]) + shift_h[i]
P_i = Linear(B_i;W_rgb_head,b_rgb_head)       # 3072→192
```

PT使用獨立`pt_head`做相同結構。兩者unpatchify順序是輸出feature `(a,b,channel)`：

```text
v_rgb[d,f,2r+a,2c+b] = P_rgb[120f+10r+c, (2a+b)48+d]
```

這與VAE輸入patchify的`(channel,col_offset,row_offset)`排列不同，不可混用。重組成 `[48,3,24,20]` 速度。

DINO `dino_proj_out` 是3072→3072 Linear，得到清淨特徵預測 `F̂_0`：

```text
v_dino = (F_current - F̂_0) / max(t_dino/1000,0.05)
```

減法方向不能反過來。該函數以float32計算，之後cast回輸出dtype。frame0的輸出會被reclamp覆蓋。

### 10.6 所有串流同步更新

用**本輪同一次forward**的各輸出：

```text
A_next   = A_current   + Δ_action v_action
Z_v_next = Z_v_current + Δ_visual v_rgb
F_d_next = F_d_current + Δ_visual v_dino
Z_p_next = Z_p_current + Δ_visual v_pt
```

各視覺frame0再寫回原觀測latent/features。不能先更新RGB，再用已更新RGB去算同一輪action；它們的速度來自更新前的同一組狀態。

4輪後依§9反正規化action。action推論不需要VAE decoder把未來latent變RGB；要做可視化才另解碼。

### 10.7 部分joint或缺少某種輸入

某`joint_X=false`，該串流只保留一個觀測latent frame、不生成未來；其t=0、不做Euler更新。某`present_X=false`則跳過該觀測encoder並遮住觀測token；允許cross-modal prediction時未來仍可能生成。

不要在所有模式沿用387/1161固定token數。Action-only缺輸入會直接移除該串流；joint實現有些位置會保留形狀但mask禁讀。必須按實際`infer_*`與mask builder執行。本文full-joint數表只針對全部present、全部joint。

## 11. 可以照抄的總演算法

```text
讀取checkpoint權重、resolved config、dataset_stats
準備RGB/per_cam、depth、K、normalized state、text context/mask
把state Linear到4096，接在text後面
RGB_VAE = 按encoders.md逐模塊計算RGB當前幀
PT_VAE = depth→XYZ→normalize→compose→同VAE
DINO_F = 按encoders.md跑DINO→pixel-unshuffle
visual_tokens = RGB patch projector / DINO LN+Linear / PT patch projector
保存實際初始A0[32,14]（joint另保存視覺未來初始值）

如果action-only：
    30層visual prefill（t=0），保存每層K/V
    對k=0,1,2,3：
        A_current Linear成32×1024
        算本輪時間embedding、6組調變
        30層action block，outer只看action、middle讀cache+action
        Linear head成32×14速度
        A_current += Δ_k * 速度
否則joint：
    對k=0,1,2,3：
        重新投影各當前latent→tokens
        各串流各token時間：觀測0、未來用本輪排程
        30層mixed block（或相同數學的anchor-cache分拆）
        計算四種heads；DINO x0轉換成velocity
        同步Euler更新四串流；重寫回視覺觀測幀

輸出A_current[32,14]
逆merger→反正規化→逆action transforms
得到32列物理動作，依replan_steps排隊執行
```

## 12. 如何核對自己的手算沒有算錯？

每個階段先比shape與排列，再比值。建議記錄第0輪第0、7、8、23、29層的：輸入hidden、時間六向量、LN、raw Q/K、RMS/RoPE後Q/K、允許keys、softmax、self residual、cross residual、FFN residual、最終速度，以及每輪更新後的A。

用float64公式對照float32 eager時應先比較允許的誤差；BF16、torch.compile、TensorRT結果可能有額外數值差異。只檢驗最終動作不足以定位是哪層、哪個排列出錯。

特別容易錯的項目：

- 用ActionHead類而不是實際Linear head；把DiT cross-attention加了gate/RoPE。
- Q/K RMS按每head128維計算；LN除D-1；把norm3的affine漏掉。
- H=384/W=320寫反；DINO fold做成average pool；RGB/PT投影誤用同一權重。
- 把動作爲因果自回歸；省略state context；忽略padding mask。
- 用0.25正步長；所有串流都用shift6或都用shift1；把t=0當全零time embedding。
- 保存錯層cache；joint cache scatter位置重疊；更新RGB後才算同輪action。
- 把normalized action直接送機器人；使用state mean/std反正規化；自行添加clip或相對state累加。

## 13. 原始碼對照表

以下均鏈接固定模型commit，行號對應本輪讀取版本。

| 內容 | 原始碼 |
|---|---|
| 模型配置 | [configs/model/flexpi.yaml](https://github.com/MaSuo1032/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/configs/model/flexpi.yaml) |
| 狀態/動作規格 | [configs/data/robotwin.yaml](https://github.com/MaSuo1032/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/configs/data/robotwin.yaml) |
| 評估override，包括T5 | [configs/sim_robotwin.yaml](https://github.com/MaSuo1032/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/configs/sim_robotwin.yaml) |
| action-only整條主線 | [flexpi.py L1610](https://github.com/MaSuo1032/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/src/flexpi/models/flexpi.py#L1610) |
| joint整條主線 | [flexpi.py L3267](https://github.com/MaSuo1032/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/src/flexpi/models/flexpi.py#L3267) |
| joint輸入/heads | [flexpi.py L1468](https://github.com/MaSuo1032/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/src/flexpi/models/flexpi.py#L1468) |
| block、prefill、cache decoder | [mot.py](https://github.com/MaSuo1032/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/src/flexpi/models/mot.py) |
| action embedding/head | [action_dit.py](https://github.com/MaSuo1032/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/src/flexpi/models/action_dit.py) |
| norm、RoPE、visual projector/head | [wan_video_dit.py](https://github.com/MaSuo1032/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/src/flexpi/models/wan_video_dit.py) |
| scheduler | [scheduler_continuous.py](https://github.com/MaSuo1032/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/src/flexpi/models/schedulers/scheduler_continuous.py) |
| state context、cached action預測 | [backbone.py](https://github.com/MaSuo1032/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/src/flexpi/models/backbone.py) |
| 反正規化/執行 | [deploy_policy.py L477](https://github.com/MaSuo1032/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/experiments/robotwin/flexpi_policy/deploy_policy.py#L477) |
| normalizer | [normalizer.py](https://github.com/MaSuo1032/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/src/flexpi/datasets/lerobot/utils/normalizer.py) |

本文件完成了公式級展開。實際checkpoint的中間值、機器人14欄語義與閉環質量仍需取得真實數據核對；這裏不把教學算例當作真實模型運行。
