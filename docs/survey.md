# 先行研究サーベイ：微分可能生物物理 × 電気拡散 × 代謝制約

- **作成日:** 2026-11-14
- **最終更新:** 2026-11-14
- **目的:** KNP-EMI / PNP + HH + ATP 制約と、JAX 自動微分・随伴・代理勾配によるタスク学習の交差点を整理し、研究ギャップと最小トイモデルを定義する。

各文献の **Links** は、出版社ページ・DOI・arXiv・コードリポジトリなど、PDF を辿れる公式入口です。

---

## 0. 本研究の位置づけ（1 行）

> **順問題の忠実度（KNP-EMI / イオン + ポンプ + ATP）** と **逆問題の道具立て（JAX / 随伴 / 代理勾配 / タスク損失）** を一つのパイプラインに統合し、符号化・短期記憶・粗視化の創発を検証する。

---

## 1. カテゴリ A — 微分可能な生物物理ニューロンシミュレーション

### A1. Deistler et al., 2025 — Jaxley

- **書誌:** Michael Deistler et al., *Jaxley: differentiable simulation enables large-scale training of detailed biophysical models of neural dynamics*, Nature Methods, 2025.
- **核心:** JAX 上の HH / 多コンパートメント・ケーブル方程式。陰的 Euler、GPU 並列、`jax.grad` によるチャネル・シナプス・形態パラメータの勾配最適化。タスク制約学習（ワーキングメモリ、視覚）まで実証。
- **本研究との関連と限界:** 「JAX で生物物理を学習する」基盤に最も近い。イオン拡散・ポンプは拡張可能な設計だが、フル PNP / KNP-EMI + ATP ホメオスタシスを制約として表現創発を見る枠ではない。LIF 向け代理勾配は未実装。
- **Links:**
  - Nature Methods: https://www.nature.com/articles/s41592-025-02895-w
  - GitHub: https://github.com/jaxleyverse/jaxley
  - Docs: https://jaxley.readthedocs.io/en/latest/
  - PyPI: https://pypi.org/project/Jaxley/

### A2. Wang et al., 2023 — BrainPy / BrainX エコシステム

- **書誌:** Chaoming Wang et al., *BrainPy, a flexible, integrative, efficient, and extensible framework for general-purpose brain dynamics programming*, eLife, 2023.
- **核心:** JAX / JIT ベースの脳ダイナミクス。点ニューロン SNN から詳細細胞（BrainCell）、オンライン勾配（BrainTrace）まで。
- **関連と限界:** インフラ候補。電気拡散・代謝制約の標準モジュールは弱く、PNP 結合はユーザー実装が必要。
- **Links:**
  - eLife: https://doi.org/10.7554/eLife.86365
  - BrainPy: https://github.com/brainpy/BrainPy
  - BrainCell: https://github.com/chaobrain/braincell
  - BrainX: https://brainx.chaobrain.com/

### A3. Neftci, Mostafa & Zenke, 2019 — Surrogate Gradient Learning

- **書誌:** Emre O. Neftci, Hesham Mostafa, Friedemann Zenke, *Surrogate Gradient Learning in Spiking Neural Networks*, IEEE Signal Processing Magazine / arXiv:1901.09948, 2019.
- **核心:** スパイク非微分性を代理勾配で緩和し、BPTT 等で SNN を直接学習。
- **関連と限界:** RQ1 の発火符号化学習に必須。主対象は LIF。HH はゲートが滑らかな経路もあり、イオン濃度・ATP 制約下での代理勾配設計は未整備。
- **Links:**
  - arXiv: https://arxiv.org/abs/1901.09948
  - PDF: https://arxiv.org/pdf/1901.09948

### A4. Huh & Sejnowski, 2018 — Gradient Descent for Spiking Neural Networks

- **書誌:** Dongsung Huh, Terrence J. Sejnowski, *Gradient Descent for Spiking Neural Networks*, NeurIPS / arXiv:1706.04698, 2018.
- **核心:** スパイク生成の微分可能定式化と勾配降下。
- **関連と限界:** 閾値近傍の勾配安定化の参考。代謝・PNP とは独立。
- **Links:**
  - arXiv: https://arxiv.org/abs/1706.04698
  - PDF: https://arxiv.org/pdf/1706.04698

### A5. 関連ツール（コード）

| 名前 | 役割 | Link |
|------|------|------|
| Diffrax | Stiff ODE + RecursiveCheckpointAdjoint | https://docs.kidger.site/diffrax/ |
| Equinox | JAX 上のモジュール / ニューラル ODE 周辺 | https://docs.kidger.site/equinox/ |
| Optax | 最適化 | https://optax.readthedocs.io/ |

---

## 2. カテゴリ B — 電気拡散・KNP-EMI・PNP と代謝（ATP）

### B1. Mori, Jerome & Peskin — 3D 細胞電気活動 / 電気拡散

- **書誌:** Yoichiro Mori, Joseph W. Jerome, Charles S. Peskin, *A Three-Dimensional Model of Cellular Electrical Activity*（および続報のエネルギー構造論文）。
- **核心:** 細胞内外ドリフト拡散 + Poisson、膜を界面とする PDE。電中性極限の議論。浸透圧流付きモデルでは自由エネルギー散逸恒等式。
- **関連と限界:** PNP 系生物物理の理論的祖。学習・JAX・タスク最適化は対象外。
- **Links:**
  - プロジェクトページ: https://sites.math.northwestern.edu/~jwj/preprints/Kyoto/
  - PDF: https://sites.math.northwestern.edu/~jwj/preprints/Kyoto/Kyoto.pdf
  - 浸透圧・エネルギー構造 (arXiv): https://arxiv.org/abs/1101.5193
  - PDF: https://arxiv.org/pdf/1101.5193

### B2. Halnes et al., 2013 — 神経・アストロサイト電気拡散

- **書誌:** Geir Halnes et al., *Electrodiffusive Model for Astrocytic and Neuronal Ion Concentration Dynamics*, PLoS Computational Biology, 2013.
- **核心:** Nernst–Planck ベースの神経・アストロサイト内外イオン動態（K⁺ 緩衝など）。
- **関連と限界:** RQ2「イオン濃度 = 短期記憶」の物理的土台。順シミュレーション中心。
- **Links:**
  - PLoS: https://doi.org/10.1371/journal.pcbi.1003386
  - 記事: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003386

### B3. Solbrå et al., 2018 — 3D Kirchhoff–Nernst–Planck (KNP)

- **書誌:** Andreas Solbrå et al., *A Kirchhoff-Nernst-Planck framework for modeling large scale extracellular electrodiffusion surrounding morphologically detailed neurons*, PLoS Computational Biology, 2018.
- **核心:** 電中性仮定の KNP で Debye 長スケールを回避し、mm・秒スケールの細胞外イオン・電位。形態詳細ニューロンと結合。実装: KNPsim (FEniCS)。
- **関連と限界:** KNP の実用実装の代表。微分可能でなく、タスク学習・ATP 制約学習なし。
- **Links:**
  - PLoS: https://doi.org/10.1371/journal.pcbi.1006510
  - 記事: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006510
  - KNPsim: https://github.com/CINPLA/KNPsim

### B4. Halnes et al., 2016 — 細胞外電位への拡散効果

- **書誌:** Geir Halnes et al., *Effect of Ionic Diffusion on Extracellular Potentials in Neural Tissue*, PLoS Computational Biology, 2016.
- **核心:** NEURON + KNP ハイブリッド。拡散が LFP 低周波に効くことを示す。
- **関連と限界:** ハイブリッド順問題の参考。学習なし。
- **Links:**
  - PLoS: https://doi.org/10.1371/journal.pcbi.1005193
  - ModelDB 関連: https://github.com/ModelDBRepository/225311

### B5. Ellingsrud et al., 2020 — KNP-EMI（有限要素）

- **書誌:** Ada J. Ellingsrud, Andreas Solbrå, Gaute T. Einevoll, Geir Halnes, Marie E. Rognes, *Finite Element Simulation of Ionic Electrodiffusion in Cellular Geometries*, Frontiers in Neuroinformatics, 2020.
- **核心:** **KNP-EMI** = KNP + EMI（細胞内外・膜を幾何的に明示）。軸索束の ephaptic など。
- **関連と限界:** 本研究が想定する順問題の最寄り定式化。自動微分・タスク最適化は未実装。
- **Links:**
  - Frontiers: https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2020.00011/full
  - DOI: https://doi.org/10.3389/fninf.2020.00011

### B6. Ellingsrud et al. — Cell-based KNP-EMI + Na/K-ATPase

- **書誌:** Ada J. Ellingsrud et al., *A Cell-Based Model for Ionic Electrodiffusion in Excitable Tissue*（書籍章; Springer）。
- **核心:** KNP-EMI 導出。膜に HH + **Na/K/ATPase** \(I_\mathrm{ATP}(c_{\mathrm{Na}}^i, c_{\mathrm{K}}^e)\) + KCC2/NKCC1。
- **関連と限界:** RQ1/3 のポンプ・ATP 消費の物理側テンプレート。ATP を学習制約にする枠はない。
- **Links:**
  - Springer 章: https://doi.org/10.1007/978-3-030-61157-6_2
  - 関連 splitting DG solver (arXiv HTML): https://arxiv.org/html/2404.08320v2

### B7. Attwell & Laughlin, 2001 — 灰白質シグナル伝達のエネルギー予算

- **書誌:** David Attwell, Simon B. Laughlin, *An Energy Budget for Signaling in the Grey Matter of the Brain*, Journal of Cerebral Blood Flow & Metabolism, 2001.
- **核心:** AP・シナプス・静止電位の ATP コスト見積もり；スパース符号化の代謝的動機。
- **関連と限界:** RQ1/3 の経験的根拠。力学系シミュレーションではない。
- **Links:**
  - Sage / 論文ページ: https://journals.sagepub.com/doi/10.1097/00004647-200110000-00001
  - DOI: https://doi.org/10.1097/00004647-200110000-00001

### B8. 代謝効率・Na 流入（Jolivet / Sengupta 系）

- **書誌例:** Renaud Jolivet 関連 / *Sodium entry efficiency during action potentials*（NeurIPS 2010 系）; Sengupta et al. の AP エネルギー効率。
- **核心:** スパイク中 Na 流入 → ポンプ ATP；SER（Sodium Entry Ratio）。
- **関連と限界:** 損失関数の物理項の設計に使える。ネットワーク学習とは未結合。
- **Links:**
  - NeurIPS PDF（SER / HH 効率）: https://proceedings.neurips.cc/paper_files/paper/2010/file/8065d07da4a77621450aa84fee5656d9-Paper.pdf
  - レビュー例: https://doi.org/10.1016/j.conb.2016.09.004

### B9. 代謝制約 SNN・コーディング（現象論）

| 文献 | 要点 | Links |
|------|------|-------|
| Chien et al., 2021 | LIF + ATP；低代謝で同期・バースト抑制様 | https://doi.org/10.3389/fncom.2021.738362 · https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2021.738362/full |
| PNAS 2023 *Metabolically regulated spiking* | ATP/ROS ホメオスタシスが発火・アヴァランチを制御 | https://doi.org/10.1073/pnas.2306525120 · https://www.pnas.org/doi/10.1073/pnas.2306525120 |
| PLoS Comp Biol 2021 eLIF / mAdExp | エネルギー制約付き現象論モデル | https://doi.org/10.1371/journal.pcbi.1008503 · https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008503 |
| Local metabolic constraints on SNNs (2025) | エネルギー依存 LIF + 3 因子可塑性 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12201681/ |
| Multi-timescale NGV (Jolivet et al., 2015) | 神経–グリア–血管代謝 + Na/K-ATPase | https://doi.org/10.1371/journal.pcbi.1004036 |

### B10. Aubert & Costalat 系 — 電気活動と代謝の結合（古典）

- **書誌:** *Modelling of the Coupling between Brain Electrical Activity and Metabolism*（Springer / Acta Biotheoretica 系）。
- **核心:** Na 膜輸送、Na/K-ATPase、解糖・PCr・ミトコンドリア、BBB・血流動態の ODE。
- **Links:**
  - Springer: https://link.springer.com/article/10.1023/A:1014286728421

---

## 3. カテゴリ C — PNP・反応拡散への随伴法・陰関数定理

### C1. Diffrax — RecursiveCheckpointAdjoint / ImplicitAdjoint

- **書誌 / ドキュメント:** Patrick Kidger ら, Diffrax（JAX 用微分可能微分方程式ソルバ）。
- **核心:** デフォルトは **discretize-then-optimize**（`RecursiveCheckpointAdjoint` + 二項チェックポイント）。Stiff は Kvaerno 系 + PID。定常なら IFT ベース `ImplicitAdjoint`。連続随伴 `BacksolveAdjoint` は勾配が近似的で非推奨。
- **関連と限界:** トイモデル勾配計算の推奨基盤。空間 PNP の陰的ステップはユーザー実装。
- **Links:**
  - Adjoints API: https://docs.kidger.site/diffrax/api/adjoints/
  - Solver 選択: https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/
  - GitHub: https://github.com/patrick-kidger/diffrax

### C2. DiffEC — JAX 微分可能電気化学（NPP 含む）

- **書誌 / コード:** BattModels / TianyMa, DiffEC — differentiable electrochemistry（ボルタンメトリ、Nernst–Planck–Poisson）。
- **核心:** 陰的 FD/FEM を JAX で端到端微分し、パラメータ推定。
- **関連と限界:** **PNP の JAX 微分可能実装の最寄り先行例**。HH 膜・ATP・スパイク学習・神経タスクはない。
- **Links:**
  - GitHub (BattModels): https://github.com/BattModels/DiffEC
  - ミラー: https://github.com/TianyMa/DiffEC

### C3. jax-fenics / dolfin-adjoint

- **核心:** FEniCS 求解を JAX `vjp` に接続；dolfin-adjoint は Poisson 最適制御の古典例。時間依存は別パッケージが必要になる場合あり。
- **関連と限界:** KNP-EMI 級 FEM の勾配経路。大規模 GPU 学習とは摩擦あり。
- **Links:**
  - jax-fenics: https://github.com/IvanYashchuk/jax-fenics
  - dolfin-adjoint 最適化: https://www.dolfin-adjoint.org/en/latest/documentation/optimisation.html
  - Poisson 最適制御デモ: https://www.dolfin-adjoint.org/en/stable/documentation/poisson-mother/poisson-mother.html

### C4. 実装指針メモ（discretize-then-optimize）

- 連続随伴 vs 離散随伴の比較文献・講義資料例:  
  https://mediatum.ub.tum.de/doc/1616339/lim2p0bl24rszcp8m0894qwq3.pdf  
- 本研究の第一選択: **離散随伴 + チェックポイント**（Diffrax の立場と一致）。

---

## 4. ミッシングリンク（研究ギャップ）

先行研究はおおむね 3 つの島に分かれる。

| 島 | できること | 足りないこと |
|----|-----------|-------------|
| A. Jaxley / BrainX | 微分可能 HH・タスク学習 | フル電気拡散 + ATP を状態として制約し表現創発を測る |
| B. KNP-EMI / Mori / KNPsim | イオン動態・ポンプ・病態スケールの順問題 | 自動微分・代理勾配・タスク損失による学習 |
| C. 代謝 SNN / DiffEC / dolfin-adjoint | ATP 現象論学習、または PNP パラメータ推定 | 神経符号化タスク × 生物物理イオン × 代謝の統合 |

**未解決ギャップ:**

1. **物理:** 「KNP-EMI（または簡約 PNP）+ HH + 動的 ATP」を一つの微分可能な状態空間に閉じ、ポンプコストを損失またはハード制約としてタスク学習に入れる枠がない。
2. **計算:** 多重時間スケール（膜電位 µs–ms、イオン秒、代謝分）の Stiff 結合系に対し、安定な時間発展 + メモリ効率的随伴 + スパイク近傍の代理勾配を同時に設計したレシピが未確立。
3. **科学:**
   - **RQ1:** 代謝／イオン疲労からバースト・時間符号化・スパース性が自己組織化するかは、現象論 SNN では示唆されるが生物物理 PDE 結合では未検証。
   - **RQ2:** 局所イオン濃度の緩和がリザーバ的短期記憶としてタスク成績に寄与するかの因果検証がない。
   - **RQ3:** ATP 上限を下げたときの粗視化相転移はコーディング理論・代謝 SNN で類推があるが、学習済み生物物理ネットでの介入実験はほぼ空白。

---

## 5. 最小プロトタイプ（1D / 少数コンパートメント）設計

数時間で動かすなら、フル KNP-EMI ではなく **点／少数コンパートメント + 集中定数イオン + ATP** から始める。

### 5.1 最小状態変数

コンパートメント \(j = 1\ldots N\)（まず \(N=1\)〜\(3\)）:

| 変数 | 意味 |
|------|------|
| \(V_j\) | 膜電位 |
| \(m,h,n\) | HH ゲート（または簡約興奮モデル） |
| \(\mathrm{Na}_i, \mathrm{K}_i\) | 細胞内濃度（細胞外は共有プール） |
| \(a\) | 正規化 ATP（\(0 < a \le 1\)） |

### 5.2 支配方程式の骨格（無次元）

\[
\begin{aligned}
C\,\dot V &= -I_{\mathrm{Na}}-I_{\mathrm{K}}-I_{\mathrm{L}}-I_{\mathrm{pump}}(a,\mathrm{Na}_i,\mathrm{K}_o)+I_{\mathrm{syn}}+I_{\mathrm{ext}} \\
\dot m &= \alpha_m(V)(1-m)-\beta_m(V)m \quad(\text{同様に }h,n) \\
\dot{\mathrm{Na}}_i &= -\gamma_{\mathrm{Na}}(I_{\mathrm{Na}}+3 I_{\mathrm{pump}}) + D_{\mathrm{Na}}(\cdots) \\
\dot{\mathrm{K}}_i &= -\gamma_{\mathrm{K}}(I_{\mathrm{K}}-2 I_{\mathrm{pump}}) + D_{\mathrm{K}}(\cdots) \\
\dot a &= \eta_{\mathrm{prod}}(a_{\max}-a) - \eta_{\mathrm{cost}}\,I_{\mathrm{pump}}
\end{aligned}
\]

ネルンスト: \(E_{\mathrm{Na}}=\ln(\mathrm{Na}_o/\mathrm{Na}_i),\ E_{\mathrm{K}}=\ln(\mathrm{K}_o/\mathrm{K}_i)\)。  
ポンプ簡約例: \(I_{\mathrm{pump}} \propto a\cdot (1+m_{\mathrm{K}}/\mathrm{K}_o)^{-2}(1+m_{\mathrm{Na}}/\mathrm{Na}_i)^{-3}\)。

さらに削るなら空間拡散 \(D\) を落とし \(N=1\)、細胞外固定バッファ → 「イオン疲労 + ATP」のみ。RQ2 用に後から細胞外共有プールを追加。

### 5.3 数値スキーム推奨

1. 固定刻み半陰的 / 陰的 Euler、または Diffrax `Kvaerno3/4` + `PIDController`、`jax_enable_x64=True`。
2. 勾配: `RecursiveCheckpointAdjoint`（`BacksolveAdjoint` は避ける）。
3. 多重スケール: operator splitting（電気サブステップ複数 → 濃度 1 回）。
4. スパイク読み出し: `custom_vjp` で代理勾配（sigmoid / arctan）。
5. ATP: 状態埋め込みを優先し、必要ならソフトペナルティを併用。
6. Optax；損失 = タスク + \(\lambda_{\mathrm{ATP}}\) 代謝コスト（+ 任意スパース項）。

### 5.4 スモーク目標

電流刺激 → スパイク → Na 蓄積 → ポンプ → ATP 低下 → 発火抑制、までを 1 本のスクリプトで可視化する（`scripts/00_forward_smoke.py`）。

---

## 6. 必読ショートリスト（ダウンロード優先順）

PDF を手元に置くなら、まず次の順が効率的です。

1. Deistler et al. 2025 (Jaxley) — Nature / 機関アクセス  
2. Ellingsrud et al. 2020 (KNP-EMI) — Frontiers オープンアクセス  
3. Solbrå et al. 2018 (KNP) — PLoS オープンアクセス  
4. Attwell & Laughlin 2001 — 出版社 / 機関  
5. Neftci et al. 2019 — arXiv PDF 直リンクあり  
6. DiffEC README + 弱電解質 NPP 例 — GitHub  
7. Diffrax Adjoints ドキュメント  
8. PNAS 2023 metabolically regulated spiking — オープンアクセス寄り  

オープンアクセスが多いのは **PLoS / Frontiers / arXiv / PNAS（状況による）** です。Nature Methods と一部ジャーナルは機関 VPN または著者原稿版を探してください。

---

## 7. 改訂履歴

| 日付 | 内容 |
|------|------|
| 2026-11-14 | 初版：3 カテゴリ文献整理、ギャップ、トイ設計、リンク一覧 |
