# Research Idea Proposal：BCH Log-Signature Algebraic Memory（BLA-Mem）

（以 log-signature 的自由 Lie 代數結構做**可結合、可平行 scan** 的序列記憶）

---

## 0. 一句話摘要

把序列視為路徑，對每個片段取 **log-signature（位於自由 Lie 代數）**當作「片段記憶」，再用 **Chen identity 對應的路徑拼接**，在 log 域用 **BCH（Baker–Campbell–Hausdorff）**做成**可結合（associative）**的記憶合成運算，從而用 **parallel prefix scan / segment tree** 在 GPU 上平行計算長序列的全域或逐點記憶。

---

## 1. 背景與動機

長序列建模常見兩個瓶頸：

1. **RNN 類**：時間步依賴造成序列長度越長越難平行化；梯度/記憶也容易退化。
2. **Attention 類**：計算/記憶體常見 (O(T^2))，長度上去很吃力。

**Signature / log-signature**（rough paths 理論核心物件）提供一個不同視角：

* Signature 將路徑映射成一組迭代積分特徵，具有良好代數結構；**Chen’s identity**給出「路徑拼接 ↔ 特徵可組合」的精確關係，能把整段的 signature 由子段組合而得。([arXiv][1])
* log-signature 是 signature 的「對數」，落在自由 Lie 代數（Lie series），通常在特徵維度上更緊湊，且更貼近可組合的群/代數幾何結構。([researchers.ms.unimelb.edu.au][2])
* 更關鍵的是：**兩段路徑拼接的 log-signature，可用 BCH 把兩段 log-signature 合成**。這直接把「記憶合成」變成一個（截斷後）可實作的代數運算。([warwick.ac.uk][3])

因此你提出的 BLA-Mem 方向，本質是在做一個**不靠 recurrence、也不靠 attention**的「代數式可結合摘要（associative summary）」，天然支援 parallel scan（類 prefix-sum）。([cs.cmu.edu][4])

---

## 2. 研究問題（Research Questions）

**RQ1（可行性）**：用「片段 log-signature + 截斷 BCH 合成」做記憶，是否能在經典 long-range toy tasks（adding / parity）中隨長度擴展而維持穩定表現？
**RQ2（平行化收益）**：在相同/相近參數量下，BLA-Mem 的 throughput / latency 是否能因 associative scan 顯著優於序列式 recurrence？([cs.cmu.edu][4])
**RQ3（不變性/歸納偏置）**：在「同一訊號不同速度採樣」的時間重參數測試中，BLA-Mem 是否能展現 signature 家族的優勢（並釐清何時需要 time augmentation）？([arXiv][1])
**RQ4（你的 novelty 點）**：**自適應截斷階數**能否把計算量變成「內容驅動」，同時維持準確率/穩定性？

---

## 3. 方法：BLA-Mem 模組設計

### 3.1 路徑化（Pathification）與 time augmentation

給定序列 (x_{1:T}\in\mathbb{R}^d)，把它視為分段線性路徑 (X(t))。
若任務對「發生速度」敏感，signature 對時間重參數本身具有不變性，常見做法是把時間作為一個額外通道（time augmentation）以保留速率資訊。([turing.ac.uk][5])

### 3.2 片段表徵：截斷 log-signature

把路徑切成片段（window / chunk）({X^{(i)}}_{i=1:N})，每段長度 (\Delta)：
[
\ell_i = \log \mathrm{Sig}^{\le m}(X^{(i)}) \in \mathfrak{g}^{\le m}(\mathbb{R}^d)
]
其中 (\mathfrak{g}^{\le m}) 是截斷到階數 (m) 的自由 Lie 代數。

**工程落地**：用 Signatory（PyTorch，支援 GPU/backprop）直接算 signature / logsignature。([OpenReview][6])

### 3.3 記憶合成：截斷 BCH 作為二元可結合運算

由 Chen identity 可知「路徑拼接 ↔ signature 的乘法結構」，而在 log 域，拼接對應到 BCH：兩段 log-signature 的合成可用 BCH 算出。([arXiv][1])

定義二元運算（你的“memory merge”）：
[
\ell_{a}\ \oplus\ \ell_{b} ;=; \mathrm{BCH}*{\le m}(\ell_a,\ell_b)
]
其中 (\mathrm{BCH}*{\le m}) 是截斷到階 (m) 的 BCH Lie 多項式（由巢狀 commutator 組成）。

### 3.4 Parallel scan：全域/逐點記憶

因為 (\oplus) 對應路徑拼接（在截斷設定下設計成一致的合成規則），你可以用：

* **segment tree reduce** 得到整段全域記憶 (\ell_{1:N}^{\mathrm{global}})
* **prefix scan** 得到每個位置的前綴記憶（類 hidden state，但可平行）

Prefix scan 是典型可平行 primitive（Blelloch scan 等），深度 (O(\log N)) 且總工作量 (O(N))（以每次合成成本為單位）。([cs.cmu.edu][4])

### 3.5 Readout（下游頭）

* 分類：(y=\mathrm{MLP}(\ell^{\mathrm{global}}))
* 序列標註：(y_t=\mathrm{MLP}(\ell^{\mathrm{prefix}}_t)) 或加上 local features（例如最後一段的 (\ell)）

---

## 4. 你的 novelty：自適應截斷階數（content-driven truncation）

固定 (m) 的痛點是計算量與特徵維度可能爆炸（尤其 (d) 大時）。log-signature 雖比 full tensor signature 更緊湊，但仍會隨階數成長。([Adeline Fermanian][7])

**提案：讓 (m) 變成動態的**
對每段（或對合成節點）估計「高階尾項能量」並決定是否需要更高階：

* 先算到 (m_{\max})，得到各階 block (\ell^{(k)})，用
  [
  r_k=\frac{|\ell^{(k)}|}{\sum_{j\le k}|\ell^{(j)}|}
  ]
  若 (r_k<\tau) 連續多個 (k) 成立則提前截斷到 (m=k)。
* 或者用「合成後增量」來判斷：比較 (\mathrm{BCH}*{\le k}) 與 (\mathrm{BCH}*{\le (k+1)}) 的差，差小就停止加階。

你可以把這做成兩種版本：

1. **硬截斷（hard adaptive m）**：節省算力最明顯。
2. **軟門控（soft gating）**：對每階係數學一個 gate（例如 sigmoid），讓模型自己抑制不需要的高階項，同時保留端到端可微。

---

## 5. 理論與分析（你可以寫成 analysis paper 的骨架）

### 5.1 可結合性與正確性（相對於路徑拼接）

* Signature 的 Chen identity 給出「子段可組合」的嚴格代數結構。([arXiv][1])
* log-signature 拼接用 BCH 表達。([warwick.ac.uk][3])
* 截斷帶來近似誤差：你的 analysis 可以聚焦在

  * 誤差如何隨 (m)、(\Delta)、路徑變差/控制量改變
  * 自適應截斷如何在固定誤差門檻下節省計算

### 5.2 計算複雜度與平行化

* 設 (N=T/\Delta)。scan 深度 (O(\log N))，適合 GPU。([cs.cmu.edu][4])
* 主要成本在「logsignature 計算」與「BCH 合成」；你可以把論文的工程亮點放在：

  * BCH 的高效實作（預先列出 commutator basis / 查表）
  * 分層合成時的快取策略（tree nodes reuse）
  * adaptive truncation 的實際 wall-clock 改善

---

## 6. 實驗計畫（Toy → 性質驗證 → 延伸）

### 6.1 Toy tasks（快速驗證）

1. **Adding problem / long-range sum**：長度拉到 8k、16k 看是否穩定。
2. **k-step parity**：測長距離依賴與梯度/記憶退化。

對照組（baselines）建議至少含：

* GRU/LSTM（序列式）
* Transformer（可限制 attention window，公平比較）
* 一個長序列友善模型（你選：S4/Hyena/Mamba 類；若你要寫 analysis paper，也可只做少量代表性對照）

### 6.2 不變性/歸納偏置測試：時間重參數

* 同一條訊號，做不同速度採樣（stretch / compress），測分類/回歸是否一致。
* 兩種設定：
  (a) **不加 time channel**：預期對速度較不敏感（更接近重參數不變）。([arXiv][1])
  (b) **加入 time augmentation**：若任務確實依賴速率，預期能恢復可辨識性。([turing.ac.uk][5])

### 6.3 Ablation（把論文 novelty 打實）

* 固定 (m) vs adaptive (m)（同準確率下的 FLOPs / latency）
* 不同 chunk size (\Delta)
* BCH 截斷階數對穩定性的影響
* prefix memory（逐點）vs global memory（整段）

### 6.4 系統面評估（你主打 parallel scan 就要量）

* 端到端吞吐（tokens/s）、延遲（ms）、顯存占用
* 序列長度 scaling curve
* scan 實作：tree reduce vs Blelloch scan（或直接用框架 primitive）

---

## 7. 預期貢獻（寫在 proposal 的 “Contributions”）

1. **新的序列記憶模組**：以 log-signature 作為路徑幾何不變量特徵，並用 BCH 提供可組合的記憶合成。([researchers.ms.unimelb.edu.au][2])
2. **可平行化的長序列建模路線**：把「hidden state 更新」改寫成 associative scan primitive，充分利用 GPU 平行 prefix-sum 類算法。([cs.cmu.edu][4])
3. **自適應截斷階數**：把計算量變成內容驅動，並用實證 + 誤差分析支持。
4. **可重現的 PyTorch 實作**：基於 Signatory 的 logsignature + 自行實作 BCH merge（並提供 benchmark）。([OpenReview][6])

---

## 8. 最小可行落地（MVP）建議

* 先做：固定 (m=2,3,4) + chunked logsig + BCH 合成 + global readout
* 只跑：adding、parity、重參數不變性
* 同時記錄：準確率 vs 長度、訓練穩定性、吞吐/顯存
* 然後再加：adaptive truncation（硬截斷版本通常最好先出結果）

---

如果你願意，我可以把這份 proposal 直接整理成「論文式」的 2–4 頁格式（含 related work 段落結構、公式編號、實驗表格模板、以及你 novelty 的 claim/lemma 該怎麼寫得最像 analysis paper）。

[1]: https://arxiv.org/pdf/1603.03788?utm_source=chatgpt.com "A Primer on the Signature Method in Machine Learning"
[2]: https://researchers.ms.unimelb.edu.au/~xgge%40unimelb/Files/Papers/The%20Signature%20of%20a%20Rough%20Path-Uniqueness.pdf?utm_source=chatgpt.com "The Signature of a Rough Path: Uniqueness"
[3]: https://warwick.ac.uk/jreizenstein/logsignatures.pdf?utm_source=chatgpt.com "Calculation of Iterated-Integral Signatures and Log ..."
[4]: https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf?utm_source=chatgpt.com "Prefix Sums and Their Applications"
[5]: https://www.turing.ac.uk/sites/default/files/2023-11/ons_nowcasting_2023.pdf?utm_source=chatgpt.com "Nowcasting with signature methods"
[6]: https://openreview.net/forum?id=lqU2cs3Zca&utm_source=chatgpt.com "Signatory: differentiable computations of the signature and ..."
[7]: https://afermanian.github.io/assets/docs/master_thesis_fermanian.pdf?utm_source=chatgpt.com "Signature and statistical learning"
