# モデル仕様（方程式・記号表）

- **作成日:** 2026-11-14
- **最終更新:** 2026-11-15
- **状態:** Phase 1 正式版 + Phase 3b 局所 \(K_e\) 拡張

本ドキュメントは実装の正本である。実装は `src/jnpnp/models/hh_point.py`、
`ions_atp.py`、`point_sim.py`（`simulate_driven` / `simulate_driven_ke`）。

---

## 1. スコープと符号規約

| 項目 | 内容 |
|------|------|
| 対象 | 点ニューロン 1 個の順シミュレーション |
| 時間単位 | **ms** |
| 膜電位 | **mV** |
| 濃度 | **mM** |
| ATP | 無次元 \(a \in (0,1]\) |
| 電流 | **外向き正**（\(\mu\mathrm{A}/\mathrm{cm}^2\)） |
| 非対象（Phase 1） | 学習、ネットワーク、空間 PNP/KNP、動的 \(K_o\)、多コンパートメント |

細胞外は定数バッファ:

- \(\mathrm{Na}_o = 145\,\mathrm{mM}\)
- \(\mathrm{K}_o = 4\,\mathrm{mM}\)
- \(\mathrm{K}_i = 140\,\mathrm{mM}\)（固定 → \(E_K\) は定数）

---

## 2. 状態変数

\[
y = (V,\, m,\, h,\, n,\, \mathrm{Na}_i,\, a)
\]

| 記号 | 意味 | 単位 / 範囲 |
|------|------|-------------|
| \(V\) | 膜電位 | mV |
| \(m,h,n\) | HH ゲート | 無次元 \(\in [0,1]\) |
| \(\mathrm{Na}_i\) | 細胞内ナトリウム | mM（実装では \([5,80]\) に clip） |
| \(a\) | 正規化 ATP | \((0,1]\)（実装では \([10^{-3},1]\) に clip） |

---

## 3. ネルンスト電位 (E1)

室温付近の \(RT/F \approx 26.7\,\mathrm{mV}\) を用い:

\[
\begin{aligned}
E_{\mathrm{Na}} &= 26.7\,\ln(\mathrm{Na}_o / \mathrm{Na}_i) && [\mathrm{mV}] \\
E_{\mathrm{K}}  &= 26.7\,\ln(\mathrm{K}_o / \mathrm{K}_i) && [\mathrm{mV}]
\end{aligned}
\]

\(K_i, K_o\) 固定のため \(E_K\) は定数。\(\mathrm{Na}_i\) が上昇すると \(E_{\mathrm{Na}}\) が低下し、駆動力 \((V-E_{\mathrm{Na}})\) が弱まる。

---

## 4. 古典 Hodgkin–Huxley (E2)

### 4.1 ゲート速度

標準 HH（\(V\) は mV、静止付近で \(\alpha,\beta\) が定義される形）:

\[
\begin{aligned}
\alpha_m &= \frac{0.1(V+40)}{1-\exp(-(V+40)/10)}, &
\beta_m &= 4\exp(-(V+65)/18) \\
\alpha_h &= 0.07\exp(-(V+65)/20), &
\beta_h &= \frac{1}{1+\exp(-(V+35)/10)} \\
\alpha_n &= \frac{0.01(V+55)}{1-\exp(-(V+55)/10)}, &
\beta_n &= 0.125\exp(-(V+65)/80)
\end{aligned}
\]

\[
\dot g = \alpha_g(V)(1-g)-\beta_g(V)\,g,\qquad g\in\{m,h,n\}
\]

数値実装では指数 Euler（安定化）または通常 Euler を用いる。

### 4.2 イオン電流

\[
\begin{aligned}
I_{\mathrm{Na}} &= \bar g_{\mathrm{Na}}\, m^3 h\,(V-E_{\mathrm{Na}}) \\
I_{\mathrm{K}}  &= \bar g_{\mathrm{K}}\, n^4\,(V-E_{\mathrm{K}}) \\
I_{\mathrm{L}}  &= \bar g_{\mathrm{L}}\,(V-E_{\mathrm{L}})
\end{aligned}
\]

既定コンダクタンス・リーク:

| 記号 | 既定値 |
|------|--------|
| \(\bar g_{\mathrm{Na}}\) | \(120\,\mathrm{mS}/\mathrm{cm}^2\) |
| \(\bar g_{\mathrm{K}}\) | \(36\,\mathrm{mS}/\mathrm{cm}^2\) |
| \(\bar g_{\mathrm{L}}\) | \(0.3\,\mathrm{mS}/\mathrm{cm}^2\) |
| \(E_{\mathrm{L}}\) | \(-54.4\,\mathrm{mV}\) |
| \(C_m\) | \(1\,\mu\mathrm{F}/\mathrm{cm}^2\) |

---

## 5. Na/K ポンプ (E3)

Ellingsrud 系に倣った簡約形（Phase 1 はこの式に固定）:

\[
I_{\mathrm{pump}}
=
\bar I_{\mathrm{pump}}\, a\,
\frac{1}{1+(K_{\mathrm{Na}}/\mathrm{Na}_i)^3}\,
\frac{1}{1+(K_{\mathrm{K}}/\mathrm{K}_o)^2}
\]

| 記号 | 推奨レンジ / 既定 |
|------|-------------------|
| \(K_{\mathrm{Na}}\) | \(15\)–\(20\,\mathrm{mM}\)（既定 \(15\)） |
| \(K_{\mathrm{K}}\) | \(3\)–\(5\,\mathrm{mM}\)（既定 \(3.5\)） |
| \(\bar I_{\mathrm{pump}}\) | チューニング（スモーク既定 \(0.6\,\mu\mathrm{A}/\mathrm{cm}^2\)） |
| \(\gamma_{\mathrm{Na}}\) | スモーク既定 \(4\times 10^{-4}\) |
| \(\eta_{\mathrm{prod}}\) | スモーク既定 \(2.5\times 10^{-4}\,\mathrm{ms}^{-1}\) |
| \(\eta_{\mathrm{cost}}\) | スモーク既定 \(0.05\,(\mu\mathrm{A}/\mathrm{cm}^2)^{-1}\mathrm{ms}^{-1}\) |

ポンプ電流は外向き正（正味の外向き陽電荷流）。1 サイクルあたり \(3\,\mathrm{Na}^+\) 外向き・\(2\,\mathrm{K}^+\) 内向き。

---

## 6. 膜電位 ODE (E4)

\[
C_m\,\frac{dV}{dt}
=
-I_{\mathrm{Na}}-I_{\mathrm{K}}-I_{\mathrm{L}}-I_{\mathrm{pump}}-I_{\mathrm{KATP}}+I_{\mathrm{ext}}(t)+I_{\mathrm{syn}}(t)
\]

単細胞 Phase では \(I_{\mathrm{syn}}\equiv 0\)。既定 \(\bar g_{\mathrm{KATP}}=0\) により \(I_{\mathrm{KATP}}\equiv 0\)（下記 §8c）。電流はすべて**外向き正**（\(I_{\mathrm{ext}},I_{\mathrm{syn}}\) は内向き駆動を正とする実装慣習に合わせ、RHS で \(+I_{\mathrm{ext}}+I_{\mathrm{syn}}\)）。

外部刺激はステップ電流:

\[
I_{\mathrm{ext}}(t)=
\begin{cases}
I_0 & t_{\mathrm{on}}\le t < t_{\mathrm{off}} \\
0 & \text{otherwise}
\end{cases}
\]

---

## 7. 細胞内 Na (E5)

\[
\frac{d\mathrm{Na}_i}{dt}
=
-\gamma_{\mathrm{Na}}\,(I_{\mathrm{Na}}+3\,I_{\mathrm{pump}})
\]

符号チェック: スパイク時 \(I_{\mathrm{Na}}<0\)（内向き）→ \(-\gamma_{\mathrm{Na}} I_{\mathrm{Na}}>0\) → \(\mathrm{Na}_i\) が増加する。

実装クリップ: \(\mathrm{Na}_i \leftarrow \mathrm{clip}(\mathrm{Na}_i,\,5,\,80)\).

---

## 8. ATP (E6)

\[
\frac{da}{dt}
=
\eta_{\mathrm{prod}}(a_{\max}-a)
-
\eta_{\mathrm{cost}}\,\max(I_{\mathrm{pump}},\,0)
\]

実装クリップ: \(a\leftarrow\mathrm{clip}(a,\,10^{-3},\,a_{\max})\)（供給ストレス時は \(a_{\max}<1\)）。

---

## 8b. 任意: ATP→Na 興奮性ゲート（RQ3 予兆）

Phase-1 スケールでは \(a_{\max}\to\)ポンプ\(\to\mathrm{Na}_i\to E_{\mathrm{Na}}\) だけでは発火率がほぼ動かないことがある。RQ3 最小ではオプションで

\[
I_{\mathrm{Na}}
\leftarrow
I_{\mathrm{Na}}\cdot\bigl(f+(1-f)\,a\bigr),\qquad
f=\texttt{atp\_gNa\_floor}
\]

を掛ける（\(f\le 0\) または未設定で無効、既定オフ）。現象論であり、チャネル ATP 依存の主張ではない。

---

## 8c. 簡約 K_ATP（Phase 4-Bio）

ATP 低下で開く K コンダクタンスの**トイ簡約**（分子忠実ではない）。採用ゲート（案1・Hill）:

\[
\sigma(a)
=
\frac{1}{1+(a/a_{1/2})^{n}}
\qquad
(a\downarrow \Rightarrow \sigma\uparrow)
\]

\[
I_{\mathrm{KATP}}
=
\bar g_{\mathrm{KATP}}\,
\sigma(a)\,
(V-E_{\mathrm{K}})
\]

| 記号 | 意味 | 既定 / RQ3-C 例 |
|------|------|----------------|
| \(\bar g_{\mathrm{KATP}}\) | 最大コンダクタンス | 既定 **0**（無効）; C 条件 **0.3** mS/cm² |
| \(a_{1/2}\) | 半開 ATP | **0.2** |
| \(n\) | Hill 係数 | **2** |

実装: `ions_atp.I_KATP_current` / `sigma_KATP`。主張上限: 速い代謝ブレーキのトイ経路まで。本物の Kir6.x/SUR チャネル再現ではない。

---

## 8d. 局所代謝3因子学習則（Phase 4-Learn）

チャネル \(i\in\{\mathrm{sig},\mathrm{noise}\}\) の eligibility と ATP ゲート（大域スカラー \(M\equiv 1\) 可）:

\[
\dot e_i = -\frac{e_i}{\tau_e}+\frac{I_i(t)}{I_{\mathrm{ref}}}\,s(t),\qquad
s=\sigma\bigl(k(V-V_{\mathrm{th}})\bigr)
\]

\[
g(a)=\tanh\bigl((a-a_{\mathrm{crit}})/\beta\bigr),\qquad
\Delta w_i \propto M\,\langle e_i\,g(a)\rangle_t
\]

実装: `jnpnp.numerics.local_metabolic`。Adam/BPTT 対照は捨てない。脳の学習則再現の主張はしない。

### 8d-A1. 区間 S 発火率ホメオスタシス（Local+homeo）

純局所ではない。\(w_1\) のみ:

\[
\mathrm{err}_S=\mathrm{mean}_S-s^\star,\qquad
\Delta w_1^{\mathrm{homeo}}=-\eta_h\,\mathrm{err}_S,\qquad
\Delta w_2^{\mathrm{homeo}}=0
\]

\[
\Delta w=\Delta w^{\mathrm{met}}+\Delta w^{\mathrm{homeo}}
\]

例: \(s^\star=0.08\), \(\eta_h=8\)（`scripts/09_rq4_local_homeo.py`）。皮質ホメオスタシス可塑性の再現ではない。

---

## 9. 数値スキーム（Phase 1）

| 項目 | 選択 |
|------|------|
| 時間刻み | 固定 \(\Delta t = 0.01\)–\(0.025\,\mathrm{ms}\)（スモーク既定 \(0.025\)） |
| \(V\) | 前進 Euler |
| ゲート \(m,h,n\) | 指数 Euler（\(\tau=1/(\alpha+\beta)\)） |
| \(\mathrm{Na}_i,a\) | 前進 Euler + clip |
| 精度 | `jax.config.update("jax_enable_x64", True)` |
| Diffrax | 必須ではない（後で差し替え可能な固定刻みループ） |

---

## 10. スモーク実験の定性目標

電流刺激 → HH スパイク連発 → \(\mathrm{Na}_i\) 上昇 → ポンプ稼働 → \(a\) 低下 → 後半の発火抑制。

対照: \(a\equiv 1\)（ポンプ ATP 依存を切った固定 ATP）を同一図に重ねる。

段階チューニング手順:

1. \(\gamma_{\mathrm{Na}}=0\), \(\bar I_{\mathrm{pump}}=0\), \(a\) 固定 → 古典 HH 発火
2. \(\gamma_{\mathrm{Na}}>0\) → 連発で \(\mathrm{Na}_i\) が数 mM 動く
3. ポンプ + ATP を入れ、後半抑制が見えるよう \(\eta_{\mathrm{prod}},\eta_{\mathrm{cost}},\bar I_{\mathrm{pump}}\) を調整

---

## 11. 記号一覧（クイック）

| 記号 | 意味 |
|------|------|
| \(C_m\) | 膜容量 |
| \(\bar g_{\mathrm{Na}},\bar g_{\mathrm{K}},\bar g_{\mathrm{L}}\) | 最大コンダクタンス |
| \(E_{\mathrm{Na}},E_{\mathrm{K}},E_{\mathrm{L}}\) | 反転電位 |
| \(I_{\mathrm{pump}},\bar I_{\mathrm{pump}}\) | ポンプ電流とその最大スケール |
| \(K_{\mathrm{Na}},K_{\mathrm{K}}\) | ポンプ半飽和定数 |
| \(\gamma_{\mathrm{Na}}\) | 電流→濃度変換係数 |
| \(\eta_{\mathrm{prod}},\eta_{\mathrm{cost}}\) | ATP 産生・消費係数 |
| \(I_{\mathrm{ext}},I_0,t_{\mathrm{on}},t_{\mathrm{off}}\) | 外部刺激 |
| \(K_e,K_{e,\mathrm{bath}},\tau_K,\gamma_{Ke}\) | 局所細胞外 K（Phase 3b） |
| \(a_{\max}\), `atp_gNa_floor` | ATP 供給上限; 任意 Na 興奮性ゲート（RQ3） |
| \(I_{\mathrm{KATP}},\bar g_{\mathrm{KATP}},a_{1/2},n\) | 簡約 K_ATP（Phase 4-Bio; 既定 off） |
| \(I_{\mathrm{syn}},w_{i\leftarrow j},\tau_{\mathrm{syn}}\) | 2細胞ソフトスパイク・シナプス（Phase B1） |
| \(D\)（\(K_e\) 2点拡散） | Phase B2: \(dK_{e,i}/dt += D(K_{e,j}-K_{e,i})\) |

---

## 12. Phase 3b — 局所細胞外 \(K_e\)（RQ2 最小）

狭い細胞外プール（1 コンパートメント近似）。\(K_i\) は固定のまま。

\[
\frac{d K_e}{dt}
=
\gamma_{Ke}\,(I_{\mathrm{K}}-2 I_{\mathrm{pump}})
-
\frac{K_e-K_{e,\mathrm{bath}}}{\tau_K}
\]

- \(K_{e,\mathrm{bath}}=4\,\mathrm{mM}\)、実装では \(K_e\in[2,20]\) に clip
- \(E_{\mathrm{K}}=26.7\ln(K_e/K_i)\)（動的）
- ポンプの K 飽和も \(K_e\) を使用
- **ion-off アブレーション:** \(K_e\) を bath にクランプ（痕跡なし）
- スモーク既定例: \(\gamma_{Ke}=5\times10^{-4}\), \(\tau_K=1000\,\mathrm{ms}\)（`scripts/05_rq2_ke_memory.py`）

主張上限: 遅い \(K_e\) が短期痕跡として効く**予兆**まで。ワーキングメモリ実装の主張はしない。

---

## 13. 2細胞シナプス結合（Phase B1）

細胞 \(i=1,2\) は各々独立な \((V,m,h,n,\mathrm{Na}_i,a)\)。拡散 PDE・ギャップジャンクションは使わない。

\[
s_j=\frac{1}{1+\exp\bigl(-k(V_j-V_{\mathrm{th}})\bigr)},\qquad
\dot{\bar s}_j=-\frac{\bar s_j}{\tau_{\mathrm{syn}}}+s_j
\]

\[
I_{\mathrm{syn},i}=w_{i\leftarrow j}\,\bar s_j
\]

- \(w>0\): 興奮、\(w<0\): 抑制。実装: `jnpnp.models.two_cell`
- B1 既定例: 共有 \(I_{\mathrm{drive}}\), \(\tau_{\mathrm{syn}}=8\,\mathrm{ms}\), 弱い相互抑制＋構造非対称 \(w_{12}\neq w_{21}\), 微小 \(V(0)\) シード
- 主張上限: 発火非対称／さぼりの**予兆**まで（ネットワーク知能・リハビリ・脳波は主張しない）

---

## 14. 2細胞局所 \(K_e\) 拡散（Phase B2）

各細胞が独立な \(K_{e,i}\) を持ち、2点拡散のみで結合（フル1D格子 PDE ではない）。

\[
\frac{d K_{e,i}}{dt}
=
\gamma_{Ke}\,(I_{\mathrm{K},i}-2 I_{\mathrm{pump},i})
-
\frac{K_{e,i}-K_{e,\mathrm{bath}}}{\tau_K}
+
D\,(K_{e,j}-K_{e,i})
\]

- \(D=0\): 拡散オフ対照。\(D>0\): 隣へ遅いイオン痕跡が染みる
- \(E_{\mathrm{K},i}=26.7\ln(K_{e,i}/K_i)\)。実装: `simulate_two_cell_ke`
- B2 既定: シナプス \(w=0\)、K_ATP / gNa オフ。例 \(D=5\times10^{-3}\,\mathrm{ms}^{-1}\), \(\gamma_{Ke},\tau_K\) は RQ2 近傍
- 主張上限: 隣細胞への遅い \(K_e\) 痕跡の**予兆**まで。電場結合・脳波・組織拡散の忠実再現は主張しない
- **Phase C′:** 同一 `simulate_two_cell_ke` で \(w\neq0\) と \(D>0\) を同時にオン可（速いシナプス電流＋遅い \(K_e\) 拡散の共存トイ）
