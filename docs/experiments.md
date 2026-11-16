# 実験プロトコル（RQ1–3 + Phase 4）

- **作成日:** 2026-11-14
- **最終更新:** 2026-11-15
- **状態:** RQ1–3 最小一周 + Phase 4-Bio / 4-Learn まで Ask 検証済み。次実装は未決定。

**検証済み到達点の要約:** [`results.md`](results.md)（願望は [`vision.md`](vision.md)）。

**Phase 橋渡し:** Phase 1–2c = 物理コアと玩具 AD/Optax。Phase 3 = RQ1–3 最小。Phase 4 = K_ATP 再比較と局所則対照。創発証明にはまだしない。

## RQ1 — スパース性と符号化

- **仮説:** ATP / イオン疲労制約下で、バースト・時間符号化・自動スパース化が自己組織化する。
- **操作:** 代謝コスト係数 \(\lambda\) やポンプ感度をスイープ。
- **指標:** 発火率、ISI 統計、バースト比、タスク精度 vs エネルギー。

### Phase 3a プロトコル（実装済み）

- **スクリプト:** `scripts/04_rq1_selective_metabolic.py`
- **入力:** \(I_{\mathrm{ext}}=w_1 I_{\mathrm{sig}}+w_2 I_{\mathrm{noise}}\)。前半 S で sig パルス、後半 N で noise パルス。
- **損失:** \(L=(m_S-s_{\mathrm{hi}})^2+(m_N-s_{\mathrm{lo}})^2+\lambda\,\mathrm{mean}(\max(I_{\mathrm{pump}},0))\)
- **比較:** A \(\lambda=0\) vs B \(\lambda>0\)（同一初期 w・同一刺激）
- **主張上限:** 代謝ペナルティが選択性・節約方向に効く**予兆**まで。創発の証明はしない。
- **TODO（本格 RQ1）:** 指標・λ スケール・多試行統計、必要ならネット化。

### Phase 3a 結果サマリ（Ask 検証済み）

- **図:** `figures/rq1_selective_metabolic.png`
- **主比較:** A λ=0 → w≈(0.53, 0.09), mean_S≈0.081, L_met≈0.071; B λ=15 → w≈(0.39, 0), mean_S≈0.070, L_met≈0.069
- **判定:** 予兆レベル合格（B は w2→0・代謝わずかに節約・mean_S 維持）。mean_N 差は弱い。λ=5 は無応答崩壊。
- **詳細:** [`STATUS.md`](../STATUS.md) の「Ask 検証 — Phase 3a クローズ」日誌。

## RQ2 — イオン濃度勾配による短期記憶

- **仮説:** 局所イオン緩和がリザーバー的ワーキングメモリに寄与する。
- **操作:** イオン拡散 / 細胞外プール on/off、濃度時定数をアブレーション。
- **指標:** 遅延マッチング精度、delay 長に対する成績曲線。

### Phase 3b プロトコル（実装済み・最小）

- **スクリプト:** `scripts/05_rq2_ke_memory.py`
- **モデル:** 局所 \(K_e\)（`simulate_driven_ke`）。式は [`model.md`](model.md) §12。
- **B0:** パルス列 → \(K_e(t),E_K(t),V(t)\)（`figures/rq2_ke_trace.png`）
- **B1 Protocol X:** cue（強/無）→ delay → 弱い probe。ion-on vs ion-off で probe 期 mean soft-spike を比較（`figures/rq2_delay_probe.png`）
- **主張上限:** 遅い \(K_e\) が短期痕跡として効く予兆。WM 実装の主張はしない。
- **TODO（本格 RQ2）:** 学習付き遅延マッチ、細胞外共有プール、統計。

### Phase 3b 結果サマリ（Ask 検証済み）

- **B0:** \(K_e\) 4 → ≈9.3 mM → 緩やかに低下（`figures/rq2_ke_trace.png`）
- **B1:** probe mean_s sep (strong−weak) = **+0.0338** (ion-on) vs ≈0 (ion-off)
- **判定:** RQ2 予兆として合格。ion-off で差が消えるため痕跡の因果は \(K_e\) 側。WM 実装の主張はしない。
- **詳細:** [`STATUS.md`](../STATUS.md) の「Ask 検証 — Phase 3b クローズ」日誌。

## RQ3 — 代謝ストレス下の粗視化（最小プロトコル）

- **仮説（予兆レベル）:** 学習後に ATP 供給上限を下げると、弱い／ノイズ入力の応答が先に落ち、強いシグナルはある程度残る（粗視化の予兆）。
- **スクリプト:** `scripts/06_rq3_atp_stress.py` → `figures/rq3_atp_stress.png`
- **禁止:** スイープ中の再学習・大規模ネット。主張上限は粗視化予兆まで（相転移／本質特徴の証明ではない）。

### C0 — 学習済み重み

1. Phase 3a 同型タスク（区間 S/N、soft-spike 目標）で **λ=0**、ゲート off のまま短学習。
2. 最終 \(w^*=(w_1,w_2)\) を固定（以降更新しない）。埋め込み既定またはスクリプト内再学習。

### C1 — \(a_{\max}\) スイープ（評価のみ）

| 項目 | 既定 |
|------|------|
| グリッド | \(a_{\max}\in\{1.0,0.7,0.4,0.2\}\) |
| Strong | \(I=w_1^* I_{\mathrm{sig}}\)（S 区間） |
| Weak/noise | \(I=g_{\mathrm{weak}} I_{\mathrm{noise}}\)（N 区間; 閾値近傍、既定 \(g=0.45\)） |
| 興奮性 | 任意ゲート `atp_gNa_floor`（既定 0.35）。\(I_{\mathrm{Na}}\leftarrow I_{\mathrm{Na}}(f+(1-f)a)\) |

**必須指標:** mean soft-spike（条件別）、mean(\(a\))、\(\Delta a\)、L_met≈mean(max(\(I_{\mathrm{pump}}\),0))。  
**任意:** V トレース重ね描き。

**成功条件（予兆）:** overall 応答低下；weak が相対的に強く落ちる；strong が完全沈黙しない区間がある。  
**既知の限界:** ゲート off（生物物理ポンプ経路のみ）では Phase-1 スケールで mean_s がほぼ平坦 → STATUS に null を記録。

**Ask 解釈（2026-11-15）:** ゲート付きでは \(a_{\max}\) 低下で overall 応答が落ち、weak が先に沈黙し strong は残存する粗視化トイ予兆として合格。有効経路は現象論 ATP→興奮性ゲートであり、純ポンプ経路の相転移や本質特徴抽出・忘却再現は主張しない。ゲート無し null は「速い代謝ブレーキが無いと短時間では選別が出にくい」読みと整合。詳細は [`STATUS.md`](../STATUS.md) の「Ask 検証 — Phase 3c クローズ」。

結果の記録先: リポジトリ直下の [`STATUS.md`](../STATUS.md)。

願望・今後の時間軸ロードマップ（実験結果ではない）: [`docs/vision.md`](vision.md)。

---

## Phase 4-Bio — 簡約 K_ATP と RQ3 再比較

- **仮説（予兆）:** 速い代謝ブレーキ（簡約 \(I_{\mathrm{KATP}}\)）があれば、学習後 \(a_{\max}\)↓ で弱い応答から落ちる粗視化が、現象論 gNa ゲートに頼らず出うる。
- **スクリプト:** `scripts/07_rq4_katp_rq3.py`
- **図:** `figures/rq4_katp_unit.png`（D0）, `figures/rq4_katp_rq3_compare.png`（D1）
- **式:** [`model.md`](model.md) §8c。\(\sigma(a)=1/(1+(a/a_{1/2})^n)\)。
- **K_e:** オフ（A/B/C 共通）。

### D0 — ユニットスモーク

固定パルス + `fix_a`。低 \(a\) で \(I_{\mathrm{KATP}}\) 増・発火抑制を確認。

### D1 — A/B/C × \(a_{\max}\)（主）

| 条件 | 内容 |
|------|------|
| A | 生物物理のみ（ポンプ+ATP; 追加ブレーキなし） |
| B | 現象論 `atp_gNa_floor`（Phase 3c 再現） |
| C | 簡約 K_ATP（gNa ゲート off） |

- w* 固定（λ=0 再学習または埋め込み値）
- \(a_{\max}\in\{1.0,0.7,0.4,0.2\}\); strong / weak は 3c 同型
- 成功目安: C で weak-first; A は null 寄りを隠さず; B≈3c

**主張上限:** 簡約 K_ATP 経路のトイ予兆まで。分子忠実・脳の省エネ証明ではない。

**Ask 検証（2026-11-15）:** 合格。B/C で weak-first、A は null。詳細は [`STATUS.md`](../STATUS.md) の「Ask 検証 — Phase 4-Bio クローズ」。

---

## Phase 4-Learn — 局所 ATP 3因子則 vs Adam

- **仮説（予兆）:** eligibility × ATP ゲートの局所更新でも、3a 同型 S/N 課題で選択／節約の予兆が出うる（出なくても null として記録）。
- **スクリプト:** `scripts/08_rq4_local_metabolic_learning.py` → `figures/rq4_local_vs_adam.png`
- **式:** [`model.md`](model.md) §8d。\(M\equiv 1\)（報酬なし）。Adam は対照として残す。
- **採用:** \(\tau_e=20\,\mathrm{ms}\), \(I_{\mathrm{ref}}=15\), \(g=\tanh((a-a_{\mathrm{crit}})/\beta)\), 純局所（task assist off）
- **対照:** Adam lr=0.05, λ=0, 同 steps・同 w0・同刺激
- **主張上限:** 局所 ATP ゲート学習の予兆／null まで。BPTT 不要の証明・脳学習則再現ではない。

**Ask 検証（2026-11-15）:** 合格（おもしろい予兆）。選択◯・節約×・w1 飽和。詳細は [`STATUS.md`](../STATUS.md) の「Ask 検証 — Phase 4-Learn クローズ」。

---

## Phase 4-Learn A1 — Local + 発火率ホメオスタシス

- **目的:** 純局所の w1 clip 張り付きを、区間 S の弱い発火率ホメオスタシスで緩和（**純局所ではない**）。
- **スクリプト:** `scripts/09_rq4_local_homeo.py` → `figures/rq4_local_homeo.png`
- **式:** \(\Delta w=\Delta w^{\mathrm{met}}+\Delta w^{\mathrm{homeo}}\), \(\Delta w_1^{\mathrm{homeo}}=-\eta_h(\mathrm{mean}_S-s^\star)\)
- **対照:** Adam / Local（homeo off）/ Local+homeo
- **既定例:** \(s^\star=0.08\), \(\eta_h=8\), \(\eta_{\mathrm{met}}=0.25\), \(a_{\mathrm{crit}}=0.45\)
- **主張上限:** ゲイン暴走抑制＋選択維持の予兆まで。脳ホメオスタシス再現・Adam 不要の証明ではない。

**Ask 検証（2026-11-15）:** 合格（欠点修正）。詳細は [`STATUS.md`](../STATUS.md) の「Ask 検証 — Phase 4-Learn A1 クローズ」。

---

## Phase B1 — 2細胞弱いシナプス結合（成功条件①のみ）

- **目的:** 同一 \(I_{\mathrm{drive}}\) 下で、結合オン時に発火非対称／さぼりの予兆が出るか（学習なし）。
- **スクリプト:** `scripts/10_b1_two_cell_asymm.py` → `figures/b1_two_cell_asymm.png`
- **式:** [`model.md`](model.md) §13。\(K_e\) / K_ATP / gNa ゲートは既定オフ。
- **E0:** \(w_{12}=w_{21}=0\) → \(r_1\approx r_2\)
- **E1:** 相互抑制＋構造非対称（例 \(w_{12}=-80\), \(w_{21}=-10\)）＋微小 \(V(0)\) シード
- **成功①:** on の rate ratio ≳ 1.5 かつ off より差が大きい（②肩代わり・③リズムは対象外）
- **主張上限:** 2細胞シナプスによる非対称・さぼり予兆まで。

**Ask 検証（2026-11-16）:** ①成功。構造非対称依存を明記。詳細は [`STATUS.md`](../STATUS.md) の「Ask 検証 — Phase B1 クローズ」。

## Phase B1-② — 肩代わり（片方 \(a_{\max}\) 低下）

- **目的:** 代謝供給を片方だけ下げたとき、結合相手の発火上昇（肩代わり）予兆があるか。
- **スクリプト:** `scripts/12_b1_compensation.py` → `figures/b1_compensation.png`
- **結合:** \(w=(-15,-3)\)（|w| スイープ推奨）。駆動・τ・V0 は B1 と同一。
- **F0:** 結合オン・\(a_{\max}=(1,1)\) → 高発火=細胞1 を固定記録。
- **F1:** \(a_{\max1}\downarrow\)（主）および \(a_{\max2}\downarrow\)（対照操作）；グリッド {1.0, 0.4, 0.2}。
- **F2:** \(w=0\) で同じ \(a_{\max}\) 操作。
- **ゲート:** \(K_e\) / K_ATP / ATP→gNa 既定オフ。
- **主張上限:** 「代謝ストレス下で結合相手が発火を上げる予兆」まで。リハビリ・脳損傷補償は言わない。

## Phase B1-②′ — K_ATP 付き肩代わり再試行

- **目的:** 簡約 K_ATP を両細胞にオンし、②と同型で肩代わりを再検証。
- **スクリプト:** `scripts/13_b1_compensation_katp.py` → `figures/b1_compensation_katp.png`
- **結合・駆動:** \(w=(-15,-3)\); B1 と同一 \(I\) / \(\tau_{\mathrm{syn}}\) / \(V_0\)。
- **K_ATP:** 4-Bio C（\(\bar g=0.3\), \(a_{1/2}=0.2\), \(n=2\)）両細胞同一。ATP→gNa オフ。
- **G0/G1/G2:** baseline → \(a_{\max1}\downarrow\) on → 同操作 off。G3 \(a_{\max2}\downarrow\) は参考。
- **主張上限:** 「K_ATP 付きで代謝ストレス下の肩代わり予兆」まで。リハビリ再現は言わない。

## Phase B2 — 2細胞局所 \(K_e\) 拡散（シナプスなし最小）

- **目的:** 片側刺激の \(K_e\) 痕跡が \(D>0\) でのみ隣へ染みるかを示す。
- **スクリプト:** `scripts/14_b2_ke_diffusion.py` → `figures/b2_ke_diffusion.png`
- **式:** [`model.md`](model.md) §14。\(w=0\)；K_ATP / gNa オフ。
- **H0:** \(D=0\) — 細胞1 cue のみ；delay 中の \(K_{e2}\) ≈ bath 近傍。
- **H1:** \(D>0\)（例 0.005）— \(K_{e2}\) 遅延上昇；任意で細胞2へ弱い probe。
- **成功目安:** H1 の \(K_{e2}\)（と可能なら probe soft-spike）が H0 より大きい。
- **主張上限:** 隣への遅い \(K_e\) 痕跡の予兆まで。電場結合・脳波・組織拡散忠実再現は言わない。

## Phase C′ — シナプス＋\(K_e\) 拡散の共存

- **目的:** 同一2細胞で電車（syn）とバス（\(K_e\) 拡散）を同時に通し、条件比較する。
- **スクリプト:** `scripts/15_cprime_syn_plus_ke.py` → `figures/cprime_syn_plus_ke.png`
- **条件:** neither / syn / diff / both（\(w=(-15,-3)\), \(D=0.005\)）。
- **刺激:** B2 型（細胞1 cue → 細胞2 probe）。K_ATP / gNa オフ。
- **主張上限:** 「シナプス電流と \(K_e\) 拡散を同一トイで共存させた」まで。ハイブリッド脳通信・最適運用は言わない。

## Phase C′′ — 同時駆動で syn/diff/both 再採点

- **目的:** C′（cue→probe）と同じ4条件を B1型共有 \(I_{\mathrm{drive}}\) で再採点し、電車サインを見る。
- **スクリプト:** `scripts/16_cdoubleprime_co_drive.py` → `figures/cdoubleprime_co_drive.png`
- **刺激:** 両細胞同一 \(I=9\)（例 20–480 ms）。\(w=(-15,-3)\), \(D=0.005\)。
- **対比:** C′=バス支配 ↔ C′′=電車が採点に載りうる（課題依存）。

## Phase C′′′ — 短駆動で電車＋バス同時採点

- **目的:** 短い共有駆動で Ke クリップ前に、syn 差と diff 差が同じ採点に載るかを見る。
- **スクリプト:** `scripts/17_ctripleprime_short_codrive.py` → `figures/ctripleprime_short_codrive.png`
- **刺激:** 両細胞同一 \(I=9\)；駆動幅スイープ（主結果例 150 ms）。
- **対比:** C′=バス / C′′=電車+clip / C′′′=電車・無clip・対称駆動ではバス null。
