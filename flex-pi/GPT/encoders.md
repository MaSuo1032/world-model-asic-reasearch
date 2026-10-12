# 編碼器附錄：從像素、深度與文字算到條件 token

日期：2026-10-12｜作者：GPT｜[回主文件](README.md)

本附錄把主文件中的 VAE、DINO、文字 encoder 展開；所有 Linear、LN、softmax 的逐元素規則沿用主文件 §1、§5。模型原始碼固定於 `20c1b2b71ea35a415d5d47c39b04443cfadad7a1`，DINO 依 timm `v1.0.26`。本附錄沒有隱含「使用一個任意 ViT」的替代。

## 1. RGB resize 與排列

### 1.1 Antialiased bilinear 可寫成加權求和

原網格尺寸 `H_s,W_s`，目標 `H_d,W_d`，`align_corners=false`。一維以 row 為例：

```text
q = H_s/H_d
center(o) = q(o+0.5)
support = max(q,1)
raw_w(o,i) = max(0,1-|i+0.5-center(o)|/support)
w(o,i) = raw_w(o,i) / Σ_(j=0…H_s-1) raw_w(o,j)
```

只加入原圖範圍內的 i，邊界處重新正規化。col 也同樣。二維是 separable triangle kernel：

```text
I_resize[c,r,u] = Σ_i Σ_j w_H(r,i) w_W(u,j) I_src[c,i,j]
```

downsample 的 support 大於1，所以不是僅讀最鄰近4個像素；upsample support=1。這是 PyTorch antialiased bilinear 的實數表達，具體 float32 運算順序仍應用指定版本驗證。

例如一維 `[0,2,4,6]` 從4縮成2：第0格未正規化權重對 i=0,1,2 為0.75、0.75、0.25，因此權重是3/7、3/7、1/7，結果 `10/7`；第1格為 `32/7`。不應以兩兩平均的1與5取代。

per-camera 部署先將 uint8 除255，再 resize，再 `2I-1`；float→BF16時間依部署程式。legacy直接 composite 路徑使用 PIL BILINEAR 並產生uint8結果，會有整數捨入。要重現特定呼叫，必須選定路徑。

pointmap 使用 nearest：對輸出 `(r,u)` 選 `i=floor(r H_s/H_d)`、`j=floor(u W_s/W_d)`，在合法索引範圍取值。它沒有RGB的抗鋸齒平均。

## 2. Wan2.2 VAE encoder：逐模組計算

來源：[wan_video_vae.py](https://github.com/MaSuo1032/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/src/flexpi/models/wan_video_vae.py)。實際類別是 `WanVideoVAE38 → VideoVAE38_ → Encoder3d_38`；不要套用同檔案較舊的16-channel VAE。

### 2.1 卷積的共同逐元素規則

PyTorch Conv 是 cross-correlation，不反轉 kernel。對已padding的 `X_pad`：

```text
Y[o,t,r,c] = b[o] + Σ_i Σ_a Σ_b Σ_d
  W[o,i,a,b,d] X_pad[i, s_t t+a, s_h r+b, s_w c+d]
```

kernel3×3×3、`padding=1` 的 `CausalConv3d`：時間**只在前面pad2**，後面pad0；spatial兩側pad1。等價來源時間 `t+a-2`、row `r+b-1`、col `c+d-1`，超界值0。

對唯一觀測frame t=0，a=0,1兩個時間切片讀0，只有a=2讀當前frame。這不同於一般對稱temporal padding。

kernel1×1×1不padding；一般Conv2d同公式省略t/a維。

### 2.2 VAE RMS 與 residual block

VAE 的 `RMS_norm` **不同於 DiT RMSNorm**。在每個時間/像素位置，對channel C：

```text
VNorm(x)_c = γ_c √C x_c / max(√Σ_j x_j²,10⁻¹²)
```

這是 `F.normalize(p=2)` 的norm下限，沒有在平方和平均後加10⁻⁶。預設bias=false。

`ResidualBlock(Cin,Cout)`：

```text
h = x                                      # Cin=Cout
h = CausalConv1x1x1(x)                     # Cin≠Cout
r = CausalConv3x3x3(SiLU(VNorm_Cin(x)))
r = CausalConv3x3x3(SiLU(VNorm_Cout(r)))
y = r+h
```

中間Dropout在eval是identity，不能把它當額外乘法縮放。所有conv bias、γ都從checkpoint取。

### 2.3 初始 patchify：channel順序要精確

RGB或XYZ輸入 `[3,1,384,320]`：

```text
X_patch[4c+2b+a,0,r,u] = X[c,0,2r+a,2u+b]
a=row_offset,b=col_offset；a,b∈{0,1}
```

這是原碼的 `(c r q)`順序，其中r是col offset、q是row offset，所以是 `4c+2col+row`。得到 `[12,1,192,160]`。與DINO pixel-unshuffle `4d+2row+col`不同。

### 2.4 Encoder 完整模組順序與尺寸

忽略batch，表中T皆1。channel基數dim=160，dim_mult=[1,2,4,4]，每個down group有2個ResidualBlock。

| 階段 | 實際順序 | 輸出 C×T×H×W |
|---|---|---|
| input patchify | 空間2×2搬到channel | 12×1×192×160 |
| `encoder.conv1` | causal Conv3d 12→160,k3 | 160×1×192×160 |
| down group0 | RB160→160；RB160→160；spatial down2；加AvgDown shortcut | 160×1×96×80 |
| down group1 | RB160→320；RB320→320；spatial down2；第一chunk略過temporal conv；加AvgDown | 320×1×48×40 |
| down group2 | RB320→640；RB640→640；spatial down2；第一chunk略過temporal conv；加AvgDown | 640×1×24×20 |
| down group3 | RB640→640；RB640→640；不downsample；加AvgDown(identity) | 640×1×24×20 |
| middle0 | RB640→640 | 640×1×24×20 |
| middle1 | 單head空間attention＋residual | 640×1×24×20 |
| middle2 | RB640→640 | 640×1×24×20 |
| encoder head | VNorm640；SiLU；causal Conv3d640→96,k3 | 96×1×24×20 |
| VAE `conv1` | causal Conv3d96→96,k1 | 96×1×24×20 |
| split | 前48為μ，後48為log_var | μ：48×1×24×20 |
| latent normalize | `(μ-mean)/std` | 48×1×24×20 |

表裡每個RB均按§2.2展開，因此沒有省略尚未定義的網路步驟。

### 2.5 Down group 的主路徑與 shortcut

主路徑的spatial down：對每個frame先在**右側、下側各pad1**，再Conv2d(k3,stride2,padding0)。對r,c：

```text
Y[o,r,c] = b[o] + Σ_i Σ_a=0…2 Σ_b=0…2 W[o,i,a,b] X_pad[i,2r+a,2c+b]
```

不是兩側各pad1。downsample3d在第一次frame chunk，feature cache原先None，此時只保存spatial down後feature，**不執行temporal Conv**。後續影片chunk才串接前一次最後frame並stride2 temporal conv。action-only以及joint的當前觀測只編碼一幀，都用第一次chunk規則。

AvgDown shortcut按原碼`AvgDown3D`：因子`f_t∈{1,2}`、`f_s∈{1,2}`，先在時間前端padding使T可被f_t整除，再把每個time/spatial block展到channel。明確寫成：

```text
f = f_t f_s²
R[c f + a_t f_s² + a_h f_s + a_w, t,r,u]
  = X_timepad[c, f_t t+a_t, f_s r+a_h, f_s u+a_w]
g = Cin f / Cout
shortcut[o,t,r,u] = (1/g) Σ_(j=0…g-1) R[og+j,t,r,u]
```

group0 `(f_t,f_s,Cin,Cout)=(1,2,160,160)`；group1 `(2,2,160,320)`；group2 `(2,2,320,640)`；group3 `(1,1,640,640)`。

第一幀在f_t=2時有前端零frame，平均不能忽略這個零。down group把 **RB+down後主路徑再加此shortcut**，不是僅普通RB residual。

### 2.6 Middle attention

對唯一frame的24×20=480個位置，展開為 `[480,640]`。

```text
U = VNorm640(X)
[Q,K,V] = Conv2d1x1(U)        # 640→1920，按channel分成三份640
S[i,j] = Σ_d=0…639 Q[i,d]K[j,d] / √640
P[i,j] = softmax_j(S[i,j])
O[i,d] = Σ_j P[i,j]V[j,d]
Y = X + Conv2d1x1(O)         # 640→640
```

單head，沒有Q/K RMS、沒有RoPE；只在同一frame做全空間attention。類別註解寫causal，但這個forward沒有傳temporal mask；手算應以實際forward為準。

### 2.7 μ、log_var 與標準化常數

`VideoVAE38_.encode`只返回μ，**不執行 reparameterization**，log_var不參與觀測結果。每個channel d：

```text
Z[d,t,r,c] = (μ[d,t,r,c]-m[d])/s[d]
```

WanVAE38的48個mean（0-index順序）：

```text
[-0.2289,-0.0052,-0.1323,-0.2339,-0.2799,0.0174,0.1838,0.1557,
 -0.1382,0.0542,0.2813,0.0891,0.1570,-0.0098,0.0375,-0.1825,
 -0.2246,-0.1207,-0.0698,0.5109,0.2665,-0.2108,-0.2158,0.2502,
 -0.2055,-0.0322,0.1109,0.1567,-0.0729,0.0899,-0.2799,-0.1230,
 -0.0313,-0.1649,0.0117,0.0723,-0.2839,-0.2083,-0.0520,0.3748,
  0.0152,0.1957,0.1433,-0.2944,0.3573,-0.0548,-0.1681,-0.0667]
```

std：

```text
[0.4765,1.0364,0.4514,1.1677,0.5313,0.4990,0.4818,0.5013,
 0.8158,1.0344,0.5894,1.0901,0.6885,0.6165,0.8454,0.4978,
 0.5759,0.3523,0.7135,0.6804,0.5833,1.4146,0.8986,0.5659,
 0.7069,0.5338,0.4889,0.4917,0.4069,0.4999,0.6866,0.4093,
 0.5709,0.6065,0.6415,0.4944,0.5726,1.2042,0.5458,1.6887,
 0.3971,1.0600,0.3943,0.5537,0.5444,0.4089,0.7468,0.7744]
```

電腦先cast常數到μ dtype後再算，所以拿float64常數手算與BF16不會逐bit相同。RGB與XYZ均用同一套標準化常數。

## 3. DINOv3 ViT-B/16：完整的12層

核對來源：timm 1.0.26 的 [eva.py](https://github.com/huggingface/pytorch-image-models/blob/v1.0.26/timm/models/eva.py#L2832)、[pos_embed_sincos.py](https://github.com/huggingface/pytorch-image-models/blob/v1.0.26/timm/layers/pos_embed_sincos.py#L979)，以及Flex-π的 [dino_encoder.py](https://github.com/MaSuo1032/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/src/flexpi/models/dino_encoder.py)。

### 3.1 像素標準化與patch conv

先依§1將每個view變成224×224。輸入原範圍[-1,1]，轉[0,1]，各channel：

```text
I_norm[c,r,u] = [(I[c,r,u]+1)/2 - mean_c]/std_c
mean = (0.485,0.456,0.406)
std  = (0.229,0.224,0.225)
```

Conv2d(kernel16,stride16,padding0,3→768)，第14r+c個patch：

```text
P[14r+c,o] = b_patch[o]
  + Σ_i=0…2 Σ_a=0…15 Σ_b=0…15 W_patch[o,i,a,b] I_norm[i,16r+a,16c+b]
```

得到196×768。前面接1個checkpoint中的CLS token與4個register/storage tokens：

```text
X^(0) = [CLS ; REG0 ; REG1 ; REG2 ; REG3 ; P0 ; … ; P195]
```

共201×768。**雖最後丟掉前5列，12層attention仍須計算它們，它們會影響patch結果。** 不加絕對position embedding，不以196列代替201列跑attention。

### 3.2 DINO自身的2D RoPE

12heads×64dim。只對patch Q/K旋轉，前5個prefix Q/K原樣保留。

patch座標r,c=0…13先算：

```text
ρ_r = 2(r+0.5)/14 - 1
ρ_c = 2(c+0.5)/14 - 1
period_j = 100^(j/16), j=0…15
θ_row,j = 2πρ_r/period_j
θ_col,j = 2πρ_c/period_j
θ = [16個row角度 ; 16個col角度]     # 長32
```

half-rotation對配對d與d+32，而非相鄰二元對：

```text
q'_d = q_d cosθ_d - q_(d+32) sinθ_d
q'_(d+32) = q_d sinθ_d + q_(d+32) cosθ_d, d=0…31
```

K同樣。**這個DINO encoder內部RoPE，不同於DINO token進Flex-π主幹後的3D RoPE。** 兩者都要做，不能省一個。

### 3.3 第ℓ=0…11層

權重名稱使用timm model的 `blocks.ℓ`：

```text
U = LN_affine_1e-5(X)                         # norm1
[Q,K,V] = Linear(U;W_qkv,0)                   # 768→2304，無bias
reshape成12 heads×64；patch Q/K依§3.2旋轉
S[i,h,j] = Σ_d Q[i,h,d]K[j,h,d]/√64
P = softmax_j(S)                             # i,j均201列，沒有causal mask
O = concat_heads(Σ_j P[i,h,j]V[j,h,d])
Y = X + γ1_ℓ ⊙ Linear(O;W_proj,b_proj)
U2 = LN_affine_1e-5(Y)                       # norm2
F = Linear_2(GELU_exact(Linear_1(U2)))        # 768→3072→768
X_next = Y + γ2_ℓ ⊙ F
```

γ1/γ2是768維 **已訓練LayerScale權重**，雖初始化1e-5，不可在checkpoint推論時直接固定為1e-5。沒有Q/K RMSNorm，沒有attention inner norm，沒有SwiGLU；eval dropout/DropPath為identity。

最後 `X_final=LN_affine_1e-5(X^(12))`，保留列5…200。沒有呼叫分類head、global-pooling或CLS-only輸出；`forward_features`返回所有token。

### 3.4 三相機fold與對齊

每camera都跑上述同一份DINO權重，結果196×768。按主文件§3.3 fold成49×3072，三camera concatenate再做Flex-π的 `dino_feature_norm`與`dino_embedder`。

傳入per_cam時直接對其view resize；不傳則先按layout slots裁切composite。資料排列順序、resize鏈與影像小數都會影響結果，即使最後shape一樣。

## 4. 若由文字字串開始：umT5 encoder 也可逐項算

來源：[wan_video_text_encoder.py](https://github.com/MaSuo1032/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/src/flexpi/models/wan_video_text_encoder.py)。模型類別預設vocab=256384、D=4096、attention總寬4096、64heads×64、FFN=10240、24層、relative buckets32、`shared_pos=false`。實際載入checkpoint shape應核對這些值。

### 4.1 字串變ids不是隨意按字編號

部署使用資料集的 `DEFAULT_PROMPT.format(task=instruction)`。為可重現，應保存 **實際format後字串、tokenizer檔、ids[128]、原mask[128]**。用Python字元序號或空格切詞無法代替已訓練tokenizer。

loader的clean是whitespace：`ftfy.fix_text`（未安裝時identity）→兩次HTML unescape→strip→連續空白縮成一個空白。AutoTokenizer依自己的詞表/model加special tokens、截斷至128、padding至128。給定保存的ids後，後續都能按下列公式代入。本文沒有tokenizer詞表與一筆任務輸入，所以不捏造特定指令的ids。

### 4.2 Token embedding

```text
X^(0)[i,d] = E[ids[i],d]
```

4096維embedding，eval dropout為identity。沒有DiT時間embedding。

### 4.3 Relative position bucket

對query i、key j，令r=j-i、n=|r|。雙向32buckets分成各16：

```text
sign_offset = 16 if r>0 else 0
bucket_distance = n                                  if n<8
bucket_distance = min(15,8+floor(log(n/8)/log(128/8)·8)) otherwise
bucket(i,j) = sign_offset+bucket_distance
B_ℓ[h,i,j] = relative_embedding_ℓ[bucket(i,j),h]
```

r=0走n<8分支，不算log0。`shared_pos=false`時每層各有embedding，不用第一層bias代替所有層。

### 4.4 每層attention＋gated FFN

T5Norm是不減mean的RMS形式（有weight、無bias，ε=10⁻⁶）：

```text
RMS_T5(x)_d = g_d x_d / √(mean_e x_e²+10⁻⁶)
```

每層：

```text
U = RMS_T5(X)
Q=U Wqᵀ；K=U Wkᵀ；V=U Wvᵀ           # 全部無bias
S[i,h,j] = Σ_d=0…63 Q[i,h,d]K[j,h,d] + B_ℓ[h,i,j]
```

**T5此處沒有除√64。** 原mask=0的key加上該dtype最小有限值，float32 softmax後cast回dtype。對有效集合按softmax求和，再Wo線性（無bias）加回X：

```text
Y = X + Linear(Attention;Wo,0)
U2 = RMS_T5(Y)
F1 = Linear(U2;W_fc1,0)                    # 4096→10240
G  = GELU_tanh(Linear(U2;W_gate,0))        # 4096→10240
X_next = Y + Linear(F1⊙G;W_fc2,0)         # 10240→4096
```

這裡有兩路相乘，是T5 gated FFN；不要混到DiT的單路GELU FFN。若float16出現inf，原碼有dtype專屬clamp；BF16不走該float16分支。

24層後final RMS_T5。`encode_prompt`把原padding列輸出置0，mask改成全True。然後才append proprio到129列，按主文件繼續。

## 5. 數值運算到什麼程度算「可手算」？

本附錄與主文件把每個神經網路module定義成標量和、乘法與明列的非線性。需要查表的數量只有已訓練權重、輸入、tokenizer ids和統計值；每個中間張量的計算順序與shape已指定。

若只想理解觀測如何影響action，可從保存的VAE latent與DINO feature開始核算主文件，省去大型encoder的工作量；若要從原始像素核算，則必須按本附錄跑完，不能以任意數值替代encoder輸出。

本文的算術驗證使用另附的已知小型範例，沒有把原始模型權重或encoder結果實測過。真實模型還應核對權重版本、資料前處理與BF16捨入。
