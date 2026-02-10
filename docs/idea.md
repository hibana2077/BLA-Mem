## 新方向：**Geodesic-Selective SSM（GS-SSM）**

> 用「**李群上的可逆幾何記憶**」解決 parity 這種 *mod-2* 記憶，再用「**Mamba 式選擇性衰減**」做 Adding 這種高精度連續累加；兩者用同一套可微分、PyTorch 友善的狀態更新框架串起來。

---

## 1) 為什麼這個方向值得做（缺口）

* **Selective SSM（Mamba）**本質是「帶衰減的線性遞迴」，非常擅長壓縮/遺忘與長序列效率，但對 **parity（XOR / mod-2）** 這類「需要**無衰減、可逆、群運算**的記憶」常會出現長度變長就崩的現象（因為任何 (\alpha_t<1) 的忘卻都會把 *exact parity* 洗掉）。Mamba 的核心就是輸入相依的 (\Delta,B,C) 與衰減門控。([arXiv][1])
* **Unitary / Orthogonal RNN** 走的是另一條路：用「特徵值模長為 1」避免梯度消失/爆炸，在 Adding 這類長距依賴 toy task 上非常強。([Proceedings of Machine Learning Research][2])
* 幾何上，**在李群上用 exponential map 參數化**（把李代數映射到正交/酉群）可以自然地保證「可逆 + 範數守恆」，而且有成熟的一階最佳化做法。([arXiv][3])

**核心洞察（Insight）**：
把「長期、符號/模運算型記憶」交給 **compact Lie group（如 (U(1))、(SO(2))）上的可逆流**；把「需要選擇性保留/遺忘的連續資訊」交給 **Mamba 式 selective decay 的歐氏狀態**。這在同一個 block 裡是互補的（lossless vs lossy），而且都能做到 **O(N)** sequential 推論、也具有 scan/並行化潛力（群乘法也是 associative）。([arXiv][1])

---

## 2) 模型定義（Methodology）

令狀態分成兩部分：

* **幾何記憶（Group memory）**：(g_t \in U(1)^k)（用複數單位圓表示，計算超便宜）
* **選擇性累加（Selective Euclidean memory）**：(s_t \in \mathbb{R}^d)

### (A) 李群上的「可逆」更新（解 parity）

用 exponential map / rotation 的等價形式：
[
g_t ;=; g_{t-1} \odot \exp\big(i,\theta_t\big),\quad \theta_t = \pi \cdot \underbrace{u_t}_{\text{select}}
]
其中 (u_t) 是由輸入產生的 gating（對 parity 就直接用 bit 本身或其線性投影）。
這等價於在 (SO(2)) 上做旋轉累積，天然就是「加總再 mod (2\pi)」，因此 **mod-2 parity** 變成「旋轉 (\pi) 的次數奇偶」。這種「特徵值在單位圓上」的想法與 uRNN 的動機一致：避免長期記憶被衰減破壞。([Proceedings of Machine Learning Research][2])

### (B) Mamba 式「可遺忘」更新（解 Adding）

[
s_t ;=; \alpha_t \odot s_{t-1} + \beta_t \odot \phi(x_t)
]
其中
[
\alpha_t=\exp(-\Delta_t \lambda_t)\in(0,1),\quad \beta_t=\Delta_t
]
(\Delta_t,\lambda_t) 皆可由輸入產生（selective），完全對齊 Mamba 的「input-dependent discretization + decay」精神。([arXiv][1])

### (C) 讀出（Readout）

最簡單做法：只在最後一步讀出

* parity：(\hat{y}_{parity} = \text{sign}(\Re(g_T)))（或用 MLP 讀出）
* adding：(\hat{y}_{add} = w^\top s_T)

也可做 multi-task head，或把 (g_t) 的相位/實虛部 concat 進 readout。

---

## 3) 你要的「Novelty」可以怎麼寫

1. **從「線性 SSM」推廣到「李群值（group-valued）SSM」**：把 state transition 從 (\mathbb{R}^d) 的線性衰減，提升成 (G) 上的可逆流（exponential map），同時保留 Mamba 的選擇性（input-dependent）。([arXiv][1])
2. **toy task 的理論對齊**：parity 本質是 ( \mathbb{Z}_2) 群累積；用 (U(1)) 的二元表示（旋轉 (\pi)）是最乾淨的可微分替代。
3. **訓練穩定性論點**：可逆/範數守恆分支對抗 vanishing gradient；adding 分支保留 selective decay 以擬合連續訊號與雜訊壓縮。([香港浸會大學計算機科學系][4])

---

## 4) toy experiments（parity + adding）設計

### Parity（分類，長度泛化重點）

* 資料：長度 (T\in{50,100,200,500,1000,2000}) 的 bit 序列
* 標籤：(y = \sum_t x_t \bmod 2)
* 指標：Accuracy vs (T)（看是否長度變長崩掉）
* 對照：LSTM（經典長期依賴）、純 Mamba/SSM、純 unitary RNN（若你願意實作）([cs.toronto.edu][5])

### Adding problem（回歸，長距選擇性記憶）

用經典設定：每步輸入 ((v_t, m_t))，(v_t\sim U[-1,1])，(m_t) 是 marker（兩個位置為 1），目標為兩個 marker 對應的 (v_t) 相加。這是 LSTM/uRNN 常用 benchmark。([people.idsia.ch][6])

* 指標：MSE vs (T)
* 期望：GS-SSM 的 (s_t) 分支用 marker 產生 (\beta_t)（或直接 (\phi(x_t)=m_t v_t)）會非常乾淨地解掉。

### 必做 ablation（讓 paper 像研究）

* 拿掉 group 分支：parity 長度泛化應該顯著下降
* 拿掉 selective decay：adding 在長 (T) 會變差或需要更大模型
* 把 group 更新從 rotation 改成 unconstrained linear：觀察訓練穩定性/梯度范數變化（對應可逆性價值）([香港浸會大學計算機科學系][4])

---

## 5) PyTorch 可直接做的最小實作（核心片段）

下面用 **(U(1))**（複數單位圓）實作 group 記憶；adding 用一個標準 selective 累加 state。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GSSM(nn.Module):
    def __init__(self, d_in, d_state=32, k_group=1):
        super().__init__()
        # group (U(1)^k): produce angle theta_t
        self.theta = nn.Linear(d_in, k_group)

        # selective euclidean branch (Mamba-like simplified)
        self.to_lambda = nn.Linear(d_in, d_state)
        self.to_delta  = nn.Linear(d_in, d_state)
        self.to_inp    = nn.Linear(d_in, d_state)

        # readouts (example: multitask)
        self.parity_head = nn.Linear(2*k_group, 2)   # use (Re, Im)
        self.add_head    = nn.Linear(d_state, 1)

    def forward(self, x):  # x: (B, T, d_in)
        B, T, _ = x.shape
        device = x.device

        # group state as complex unit: store (Re, Im) for autograd simplicity
        g_re = torch.ones(B, 1, device=device)
        g_im = torch.zeros(B, 1, device=device)

        s = torch.zeros(B, self.to_lambda.out_features, device=device)

        for t in range(T):
            xt = x[:, t, :]

            # --- (A) group update: g <- g * exp(i * theta)
            th = torch.pi * self.theta(xt)  # (B, k)
            c, s_th = torch.cos(th), torch.sin(th)

            # complex multiply: (a+ib)(c+id) = (ac-bd) + i(ad+bc)
            new_re = g_re * c - g_im * s_th
            new_im = g_re * s_th + g_im * c
            g_re, g_im = new_re, new_im

            # --- (B) selective euclidean update
            lam = F.softplus(self.to_lambda(xt))      # >=0
            delt = F.softplus(self.to_delta(xt))      # >=0
            alpha = torch.exp(-delt * lam)            # (0,1]
            inp = self.to_inp(xt)
            s = alpha * s + delt * inp

        # readouts
        g_feat = torch.cat([g_re, g_im], dim=-1)  # (B, 2k)
        parity_logits = self.parity_head(g_feat)
        add_pred = self.add_head(s)

        return parity_logits, add_pred
```

你做 toy exp 時：

* **parity**：把 input 做成單一 bit（或 one-hot），只用 `parity_logits` 訓練 CE loss
* **adding**：把 input 做成 ((v_t, m_t)) 兩維（或更多），只用 `add_pred` 訓練 MSE
* 若想同一模型 multi-task：混合兩個 loss（或交替訓練）

---

## 6) 你可以怎麼寫「一段話的主張」（paper-style claim）

> 我們提出 GS-SSM：將 Mamba 式 input-dependent discretization/decay 的 selective SSM，與李群（如 (U(1))）上的可逆幾何狀態更新結合。幾何分支提供無衰減、可逆的長期記憶以對齊 parity 的群結構；歐氏分支保留可遺忘的高精度累加以解 Adding problem。該架構保持 O(N) 推論並可用 PyTorch 直接驗證，在長度泛化上預期同時超越純衰減 SSM 與純 unitary RNN。([arXiv][1])

---

如果你接下來想把 novelty 再「研究化」一點：可以把 (U(1)) 推到 (SO(d))/(U(d))，用 **exponential map / Cayley transform** 做更一般的可逆 state transition（對應李群參數化文獻），再觀察 parity 的長度泛化是否隨維度變化、以及 selective gate 是否會自動把「符號記憶」與「連續記憶」分工出來。([Proceedings of Machine Learning Research][7])

[1]: https://arxiv.org/abs/2312.00752?utm_source=chatgpt.com "Linear-Time Sequence Modeling with Selective State Spaces"
[2]: https://proceedings.mlr.press/v48/arjovsky16.pdf?utm_source=chatgpt.com "Unitary Evolution Recurrent Neural Networks"
[3]: https://arxiv.org/abs/1901.08428?utm_source=chatgpt.com "A Simple Parametrization of the Orthogonal and Unitary ..."
[4]: https://www.comp.hkbu.edu.hk/~markus/teaching/comp7650/tnn-94-gradient.pdf?utm_source=chatgpt.com "Learning long-term dependencies with gradient descent is ..."
[5]: https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec15.pdf?utm_source=chatgpt.com "CSC321 Lecture 15: Recurrent Neural Networks"
[6]: https://people.idsia.ch/~juergen/nipslstm/node4.html?utm_source=chatgpt.com "EXPERIMENT 1: ADDING PROBLEM"
[7]: https://proceedings.mlr.press/v97/lezcano-casado19a/lezcano-casado19a.pdf?utm_source=chatgpt.com "Cheap Orthogonal Constraints in Neural Networks"
