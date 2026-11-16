# Results Summary（検証済み到達点）

- **作成日:** 2026-11-15
- **最終更新:** 2026-11-16（C′′′ Ask クローズ）

> **予兆レベルのトイ結果。創発・脳再現・Adam代替の証明ではない。**  
> **物語版:** [`story.md`](story.md)　願望カタログは [`vision.md`](vision.md)。実験日誌の一次ソースは [`STATUS.md`](../STATUS.md)。

---

## スコープとモデル

- **コア:** 点 Hodgkin–Huxley + 細胞内 \(\mathrm{Na}_i\) + Na/K ポンプ + 正規化 ATP \(a\)
- **任意拡張:** 局所 \(K_e\)（RQ2）、現象論 ATP→gNa、簡約 \(I_{\mathrm{KATP}}\)、eligibility×ATP 局所学習則
- **経緯（要約）:** JAX 固定刻み順問題 → `jax.grad` スモーク → Optax 玩具学習 → RQ1–3 最小介入 → Phase 4 で K_ATP 再比較と局所則対照
- **方程式:** [`model.md`](model.md)　**プロトコル:** [`experiments.md`](experiments.md)

---

## 結果一覧

| Phase | 問い | 主スクリプト・図 | 主結果（1行） | 限界（1行） |
|-------|------|------------------|---------------|-------------|
| 1 | 順問題スモーク | `00_forward_smoke.py` · [`figures/forward_smoke.png`](../figures/forward_smoke.png) | スパイク → Na↑ → ATP↓ → 後半抑制の定性ループ | イオン収支テスト・空間構造なし |
| 2a | AD スモーク | `01_grad_smoke.py` | 固定刻み scan 経由で `jax.grad` が通る | スパイク感度で FD 照合は ε 依存 |
| 2b | 玩具学習 | `02_train_stim_gain.py` · [`figures/train_stim_gain.png`](../figures/train_stim_gain.png) | Optax で `stim_gain` が損失低下方向に動く | 課題はダミー |
| 2c | 代謝ペナルティ | `03_train_metabolic_penalty.py` · [`figures/train_metabolic_penalty.png`](../figures/train_metabolic_penalty.png) | \(L_{\mathrm{task}}+\lambda L_{\mathrm{met}}\) で A/B 差の予行 | 表現創発の主張なし |
| 3a RQ1 | 選択的入力＋λ | `04_rq1_selective_metabolic.py` · [`figures/rq1_selective_metabolic.png`](../figures/rq1_selective_metabolic.png) | λ>0 でノイズ重み削減など選択・節約の予兆 | スパース創発の証明ではない |
| 3b RQ2 | 遅い \(K_e\) 痕跡 | `05_rq2_ke_memory.py` · [`figures/rq2_ke_trace.png`](../figures/rq2_ke_trace.png), [`rq2_delay_probe.png`](../figures/rq2_delay_probe.png) | cue 後 probe 分離が ion-on で出、ion-off で消える | WM 実装・体系的 delay 曲線なし |
| 3c RQ3 | 学習後 \(a_{\max}\)↓ | `06_rq3_atp_stress.py` · [`figures/rq3_atp_stress.png`](../figures/rq3_atp_stress.png) | 現象論 ATP→gNa 付きで weak-first 粗視化予兆 | **ゲート off（ポンプのみ）は本スケールで null** |
| 4-Bio | 簡約 K_ATP | `07_rq4_katp_rq3.py` · [`figures/rq4_katp_unit.png`](../figures/rq4_katp_unit.png), [`rq4_katp_rq3_compare.png`](../figures/rq4_katp_rq3_compare.png) | A null / B・C で weak-first；K_ATP でも RQ3 予兆 | 分子忠実な K_ATP ではない |
| 4-Learn | 局所則 vs Adam | `08_rq4_local_metabolic_learning.py` · [`figures/rq4_local_vs_adam.png`](../figures/rq4_local_vs_adam.png) | 純局所で選択予兆（w2↓・mean_N→0） | 節約は Adam 優勢；**w1 が clip 飽和** |
| 4-Learn A1 | Local+homeo | `09_rq4_local_homeo.py` · [`figures/rq4_local_homeo.png`](../figures/rq4_local_homeo.png) | 区間S発火ホメオで w1 非クリップ・選択維持（**純局所ではない**） | 代謝は旧Localより僅差改善；Adam 節約には未達 |
| B1 | 2細胞シナプス非対称① | `10_b1_two_cell_asymm.py` · [`figures/b1_two_cell_asymm.png`](../figures/b1_two_cell_asymm.png) | 構造非対称の弱い相互抑制で rate ratio≈1.73（off≈1） | 対称結合だけでは出ず；肩代わり・リズムは未実施 |
| B1 補強 | \|w\| スイープ | `11_b1_w_sweep.py` · [`figures/b1_w_sweep.png`](../figures/b1_w_sweep.png) | 8:1 縮小では①消；(-15,-3) で ratio≈2.4・V健全 | 基準 (-80,-10) は Vmin≈−252 |
| B1-② | 肩代わり \(a_{\max}\) | `12_b1_compensation.py` · [`figures/b1_compensation.png`](../figures/b1_compensation.png) | **null**（ゲート off）; 高発火=cell1 | ポンプのみではストレス側が減速せず |
| B1-②′ | K_ATP 肩代わり | `13_b1_compensation_katp.py` · [`figures/b1_compensation_katp.png`](../figures/b1_compensation_katp.png) | **部分成功:** r1↓・Δr2≤0 | ブレーキ有効、肩代わり未達；4-Bio 単細胞とは条件差 |
| B2 | 2細胞 \(K_e\) 拡散 | `14_b2_ke_diffusion.py` · [`figures/b2_ke_diffusion.png`](../figures/b2_ke_diffusion.png) | **成功:** D>0 で Ke2↑・probe s2↑ | \(w=0\)；2点拡散のみ（フル1D PDEではない） |
| C′ | syn+\(K_e\) 共存 | `15_cprime_syn_plus_ke.py` · [`figures/cprime_syn_plus_ke.png`](../figures/cprime_syn_plus_ke.png) | **拡散支配の共存:** both≈diff | cue→probe では電車サイン弱；バスは \(w\neq0\) でも残る |
| C′′ | 同時駆動で再採点 | `16_cdoubleprime_co_drive.py` · [`figures/cdoubleprime_co_drive.png`](../figures/cdoubleprime_co_drive.png) | **部分成功:** syn≠neither・both≈syn | 長駆動で Ke クリップしバス弱；C′と対で課題依存 |
| C′′′ | 短駆動同時採点 | `17_ctripleprime_short_codrive.py` · [`figures/ctripleprime_short_codrive.png`](../figures/ctripleprime_short_codrive.png) | **電車のみ:** syn ratio→1.13・Ke未clip | 対称駆動では diff≈neither（Dに勾配なし） |

**B1 Ask（2026-11-16）:** ①成功。結合オンで非対称＋高発火側 Δa 大；オフは対称。成功は構造非対称抑制依存（対称結合からの自発破れは未達）。詳細は STATUS「Ask 検証 — Phase B1 クローズ」。

**B1 |w| 補強（2026-11-16）:** 基準 \(w=(-80,-10)\) は ratio≥1.5 だが過分極が深い。8:1 レイを弱めると①は消える。比を少し緩めた \(w=(-15,-3)\) が②用推奨（ratio≈2.4, Vmin≈−94）。

**B1 肩代わり（② null → ②′ 部分成功）:**

- **②（ゲート off）— 良い null:** \(w=(-15,-3)\)・ポンプ+ATPのみ。\(a_{\max1}\downarrow\) で r1・r2 不変・Δr2≈0。速いブレーキ無しでは代謝ストレス→役割付け替えの前提が立たない。
- **②′（K_ATP on）— 良い部分成功:** 同プロトコルに 4-Bio K_ATP（\(\bar g=0.3\), \(a_{1/2}=0.2\)）両細胞オン・gNa オフ。G0 高発火=細胞1（0.0162 / 0.0129）。G1 \(a_{\max1}=0.2\) で r1→0.0029（明確↓）だが Δr2≈−0.003（上がらず）。G2 off でも相手上昇なし。
- **分離できたこと:** 「ブレーキの有無」と「補償の有無」。固定シナプス相互抑制だけでは補償予兆不足。リハビリ／原理的不可能性の否定はしない。次部品（可塑性・タスク共有等）は急がない。

**B2 \(K_e\) 拡散（2026-11-16）— 成功:** シナプス \(w=0\)・2点拡散。H0 \(D=0\): delay 中 Ke2≈4.56。H1 \(D=0.005\): Ke2@probe=6.42、mean s2 probe 0.032→0.089；Ke1_max もやや低下（隣へ逃げるサイン）。**地図:** B1=速いシナプス電流（電車）、B2=遅い共有 \(K_e\) 場（バス）— 両方の最小デモが揃った。脳波・組織拡散忠実再現・1D連続体そのものは主張しない（\(D,\tau\) 依存の1点・2点近似）。

**C′ / C′′ / C′′′（三代対比）:**

| | 主役 | 核 |
|--|------|----|
| C′ cue→probe | **バス** | 活動非対称→Ke勾配 |
| C′′ 長同時 | **電車**（バスはclip） | 同時押し引き；Ke飽和 |
| C′′′ 短同時 | **電車のみ** | clip解消；**対称駆動ではバス null** |

幹線はシナプス電流が基本。バス支配はよほどな状況（非対称活動・遅い窓＝C′寄り）。「短くすれば両方見える」は対称駆動のままでは成り立たない。最適ハイブリッド証明は主張しない。

**A1 Ask（2026-11-15）:** 合格＝欠点修正。w1 飽和は緩和、選択は維持；節約は未解決。純局所主張は不可。詳細は STATUS「Ask 検証 — Phase 4-Learn A1 クローズ」。


---

## 横断メッセージ（慎重）

1. 代謝・イオン制約を、同じ微分可能な点ニューロン・トイに載せられる。
2. 選択（RQ1）、遅い痕跡（RQ2）、供給ストレス下の粗視化（RQ3）で**予兆**が出た。
3. 速い代謝ブレーキ（現象論 gNa または簡約 K_ATP）が RQ3 に効く；ポンプ→Na→\(E_{\mathrm{Na}}\) のみは本スケールで **null**。
4. 純局所 ATP ゲート学習でもノイズ抑制の予兆；省エネ最適と w1 制御は純局所では**未達**（A1 の発火ホメオで w1 暴走は緩和可）。
5. 2細胞でシナプス（電車）と \(K_e\) 拡散（バス）を載せられ、**課題依存**で採点に載る回線が変わる（C′ バス支配 ↔ C′′ 電車が見える）。
6. 研究の**種**としては有望だが、論文級の主張には追加検証が必要。

---

## 明示的に主張しないこと

- 脳の符号化・記憶・忘却・省エネ機構を**再現**した
- 本質特徴抽出の**相転移**や創発を**証明**した
- 本物の K_ATP チャネルや分子機序を再現した
- 局所則が Adam/BPTT を**代替できる**
- North star（夢・ひらめき・多細胞脳波など；[`vision.md`](vision.md)）が検証済みである

---

## 次の分岐（決めない。リンクのみ）

実装の次手は未決定。候補のみ:

| ID | 内容 | 参照 |
|----|------|------|
| **A** | 4-Learn 補強（w1 飽和・ホメオスタシス・g(a)） | [`vision.md`](vision.md) / STATUS |
| **B1** | 少数細胞（リズム・分担の芽） | vision Later |
| **B2** | 1D 局所 \(K_e\) 拡散（フル PNP/KNP-EMI ではない） | vision Phase 5 |
| **B3** | 統合デモ（既存部品の見せ方整理） | STATUS |

詳細・願望は [`vision.md`](vision.md)。日誌は [`STATUS.md`](../STATUS.md)。

---

## 図一覧（既存・再生成なし）

| 図 | Phase |
|----|-------|
| [`figures/forward_smoke.png`](../figures/forward_smoke.png) | 1 |
| [`figures/train_stim_gain.png`](../figures/train_stim_gain.png) | 2b |
| [`figures/train_metabolic_penalty.png`](../figures/train_metabolic_penalty.png) | 2c |
| [`figures/rq1_selective_metabolic.png`](../figures/rq1_selective_metabolic.png) | 3a |
| [`figures/rq2_ke_trace.png`](../figures/rq2_ke_trace.png) | 3b B0 |
| [`figures/rq2_delay_probe.png`](../figures/rq2_delay_probe.png) | 3b B1 |
| [`figures/rq3_atp_stress.png`](../figures/rq3_atp_stress.png) | 3c |
| [`figures/rq4_katp_unit.png`](../figures/rq4_katp_unit.png) | 4-Bio D0 |
| [`figures/rq4_katp_rq3_compare.png`](../figures/rq4_katp_rq3_compare.png) | 4-Bio D1 |
| [`figures/rq4_local_vs_adam.png`](../figures/rq4_local_vs_adam.png) | 4-Learn |
| [`figures/rq4_local_homeo.png`](../figures/rq4_local_homeo.png) | 4-Learn A1 |
| [`figures/b1_two_cell_asymm.png`](../figures/b1_two_cell_asymm.png) | B1 |
| [`figures/b1_w_sweep.png`](../figures/b1_w_sweep.png) | B1 \|w\| 補強 |
| [`figures/b1_compensation.png`](../figures/b1_compensation.png) | B1-② |
| [`figures/b1_compensation_katp.png`](../figures/b1_compensation_katp.png) | B1-②′ |
| [`figures/b2_ke_diffusion.png`](../figures/b2_ke_diffusion.png) | B2 |
| [`figures/cprime_syn_plus_ke.png`](../figures/cprime_syn_plus_ke.png) | C′ |
| [`figures/cdoubleprime_co_drive.png`](../figures/cdoubleprime_co_drive.png) | C′′ |
| [`figures/ctripleprime_short_codrive.png`](../figures/ctripleprime_short_codrive.png) | C′′′ |
