# 研究ストーリー（到達点の言語化）

- **作成日:** 2026-11-16
- **最終更新:** 2026-11-16（C′′′ Ask クローズ反映）

> これは検証済みトイ結果の物語要約である。創発・脳再現・Adam代替の証明ではない。一次データは STATUS / results。

事実の表: [`results.md`](results.md)　日誌: [`STATUS.md`](../STATUS.md)　願望: [`vision.md`](vision.md)

---

## 0. 一文の到達点

代謝・イオン制約つきの微分可能ニューロン・トイで、選択・痕跡・供給ストレスと、シナプス／Ke の二回線、および課題依存の効き方まで、対照実験つきで地図ができた。いい研究の種として再現可能なプロトタイプが一周した段階。

---

## 1. なぜこのプロジェクトか（短く）

機械学習側の ANN／LIF 抽象と、生物物理側の KNP 順問題／代謝 SNN は長く別棚だった。前者は学習しやすいが電気化学が薄い。後者は輸送に厚いが、微分可能学習と小さな介入実験がつながりにくい。

本研究はそのギャップを「全部解く」のではなく、**最小結合**で載せて、選択・痕跡・供給ストレス・二回線の**予兆**が対照つきで見えるかを見る。North star（脳波・リハビリ・最適ハイブリッド通信）は [`vision.md`](vision.md) に棚上げする。

---

## 2. 物語の起承転結

### 起 — 1細胞を物理＋学習可能にする

まず点 Hodgkin–Huxley に細胞内 Na、ポンプ、正規化 ATP を載せ、スパイク → Na 蓄積 → ポンプ負荷 → ATP 低下 → 後半の発火抑制、という定性ループを通した（Phase 1）。固定刻みの `scan` 経由で `jax.grad` が通り、玩具の刺激ゲイン学習と代謝ペナルティの予行まで進んだ（Phase 2）。ここまでで「物理つきで微分可能」な最小土台ができた。

**代表図:** [`figures/forward_smoke.png`](../figures/forward_smoke.png) · [`figures/train_stim_gain.png`](../figures/train_stim_gain.png) · [`figures/train_metabolic_penalty.png`](../figures/train_metabolic_penalty.png)

### 承 — 単細胞で RQ1–3 の予兆

同じ土台で最小介入を一周した。RQ1 では代謝ペナルティつきで選択・節約の予兆。RQ2 では局所 \(K_e\) の遅い痕跡が、cue→delay→probe の分離を ion-on で出し ion-off で消す。RQ3 では学習後に供給天井 \(a_{\max}\) を下げると、現象論の速いブレーキ付きで weak-first の粗視化予兆が出る一方、**ポンプのみでは本スケールで null**——減速の前提には速い代謝ブレーキが要る、と分かった。

Phase 4-Bio では簡約 K_ATP でも同様の予兆。4-Learn では純局所則で選択（ノイズ抑制）は見えるが節約は Adam 優勢、かつ w1 がクリップ飽和。A1 で発火ホメオを足すと飽和は緩和するが、純局所主張はできず、Adam 級の節約には未達のまま閉じた。

**代表図:** [`figures/rq1_selective_metabolic.png`](../figures/rq1_selective_metabolic.png) · [`figures/rq2_ke_trace.png`](../figures/rq2_ke_trace.png) · [`figures/rq2_delay_probe.png`](../figures/rq2_delay_probe.png) · [`figures/rq3_atp_stress.png`](../figures/rq3_atp_stress.png) · [`figures/rq4_katp_rq3_compare.png`](../figures/rq4_katp_rq3_compare.png) · [`figures/rq4_local_vs_adam.png`](../figures/rq4_local_vs_adam.png) · [`figures/rq4_local_homeo.png`](../figures/rq4_local_homeo.png)

### 転 — 2細胞で二回線

地図を広げるため、2細胞へ。**電車**は速いシナプス電流（B1）。構造非対称の弱い相互抑制で発火の役割分担（ratio）が出る。対称結合からの自発的破れは未達で、成功は構造非対称に依存する。衛生スイープで弱め \(w=(-15,-3)\) を選び、肩代わりを試した。ゲート off（②）は良い null——ストレス側が減速しない。K_ATP 付き（②′）では減速は出るが相手の上昇（補償）は出ず、固定シナプスだけでは足りない地図になった。

**バス**は遅い局所 \(K_e\) の2点拡散（B2）。シナプスなし・\(D>0\) で隣へ染み出し、probe 応答も増える。フル1D PDE や組織拡散の忠実再現ではない。

**代表図:** [`figures/b1_two_cell_asymm.png`](../figures/b1_two_cell_asymm.png) · [`figures/b1_w_sweep.png`](../figures/b1_w_sweep.png) · [`figures/b1_compensation.png`](../figures/b1_compensation.png) · [`figures/b1_compensation_katp.png`](../figures/b1_compensation_katp.png) · [`figures/b2_ke_diffusion.png`](../figures/b2_ke_diffusion.png)

### 結 — 課題が回線の点数を決める

同じトイに電車とバスを同時搭載し、採点だけ変えた。三代の対比:

| | 主役 | 核 |
|--|------|----|
| C′ cue→probe | **バス** | 活動非対称→Ke勾配 |
| C′′ 長同時 | **電車**（バスはclip） | 同時押し引き；Ke飽和 |
| C′′′ 短同時 | **電車のみ** | clip解消；**対称駆動ではバス null** |

短同時（主窓 150 ms）では Ke 未クリップで電車は見える（ratio≈1.13、both≈syn）が、対称駆動では両細胞の \(K_e\) が揃い \(D\) に勾配が立たない。「短くすれば両方見える」は対称駆動のままでは成り立たない。

**直観との整合:** 幹線の伝達はシナプス電流が基本。拡散が支配的なのはよほどな状況（非対称活動・遅い窓など）。C′が例外寄り、C′′′が普段寄り、と読める。

教訓: **両方載せられる。どれが点数に載るかは課題と時間スケール次第。** 最適ハイブリッド運用の証明ではない。

**代表図:** [`figures/cprime_syn_plus_ke.png`](../figures/cprime_syn_plus_ke.png) · [`figures/cdoubleprime_co_drive.png`](../figures/cdoubleprime_co_drive.png) · [`figures/ctripleprime_short_codrive.png`](../figures/ctripleprime_short_codrive.png)

---

## 3. いま言えること / 言えないこと

### 言えること

- 代謝・イオン制約を、同じ微分可能な点ニューロン・トイに載せられる
- RQ1 選択・節約、RQ2 \(K_e\) 痕跡、RQ3 供給ストレスの**予兆**（対照つき）
- 速い代謝ブレーキ（現象論 gNa または簡約 K_ATP）が RQ3／肩代わりの「減速」に要る；ポンプのみは null
- 2細胞でシナプス（電車）と \(K_e\) 拡散（バス）の最小デモが揃い、同一系に共存させられる
- 課題（cue→probe vs 同時駆動）で、採点に載る回線が変わる
- 肩代わりは、ブレーキありでも固定シナプス相互抑制だけでは補償予兆不足
- いい研究の**種**としての再現可能プロトタイプが一周した

### 言えないこと

- 脳の符号化・記憶・忘却・省エネ・リハビリを**再現**した
- 創発や相転移を**証明**した
- Adam／逆伝播の代替を示した
- 組織拡散・電場結合・脳波・1D 連続体そのもの
- シナプス不要、バスが常に主役、最適ハイブリッド通信
- 対称結合からの自発的役割分担や、肩代わりの完成

---

## 4. 未解決と次の分岐

決め打ちしない。併記のみ。

- **肩代わり:** 可塑性・タスク共有など追加部品（②′で減速までは見えた）
- **わずかに非対称な同時駆動**で二回線同時採点（任意）
- **肩代わり＋可塑性**
- **3細胞・リズム**など地図の次のマス
- 願望カタログは別棚: [`vision.md`](vision.md)

---

## 5. 代表図ギャラリー

### 起

| 図 | 内容 |
|----|------|
| [`figures/forward_smoke.png`](../figures/forward_smoke.png) | Phase 1 順問題スモーク |
| [`figures/train_stim_gain.png`](../figures/train_stim_gain.png) | Phase 2b 玩具学習 |
| [`figures/train_metabolic_penalty.png`](../figures/train_metabolic_penalty.png) | Phase 2c 代謝ペナルティ |

### 承

| 図 | 内容 |
|----|------|
| [`figures/rq1_selective_metabolic.png`](../figures/rq1_selective_metabolic.png) | RQ1 |
| [`figures/rq2_delay_probe.png`](../figures/rq2_delay_probe.png) | RQ2 probe |
| [`figures/rq3_atp_stress.png`](../figures/rq3_atp_stress.png) | RQ3 |
| [`figures/rq4_local_homeo.png`](../figures/rq4_local_homeo.png) | 4-Learn A1 |

### 転

| 図 | 内容 |
|----|------|
| [`figures/b1_two_cell_asymm.png`](../figures/b1_two_cell_asymm.png) | B1 電車 |
| [`figures/b1_compensation_katp.png`](../figures/b1_compensation_katp.png) | 肩代わり ②′ |
| [`figures/b2_ke_diffusion.png`](../figures/b2_ke_diffusion.png) | B2 バス |

### 結

| 図 | 内容 |
|----|------|
| [`figures/cprime_syn_plus_ke.png`](../figures/cprime_syn_plus_ke.png) | C′ バス支配 |
| [`figures/cdoubleprime_co_drive.png`](../figures/cdoubleprime_co_drive.png) | C′′ 電車が載る（Ke clip） |
| [`figures/ctripleprime_short_codrive.png`](../figures/ctripleprime_short_codrive.png) | C′′′ 短駆動・電車のみ |

数値・表の根拠はすべて [`results.md`](results.md) と [`STATUS.md`](../STATUS.md) を参照。
