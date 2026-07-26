# Proyek Eksperimental: Dari SPSA ke Low-Rank ke Neuro-Symbolic LLM

> **12 eksperimen dalam 3 fase:** Kegagalan SPSA → Kompresi Low-Rank → Arsitektur Transformer untuk Penalaran Aritmetika

---

## Ringkasan Per-Jalur

### Jalur A (ricci1–9): Optimasi Forward-Only & Kompresi Wavelet
| # | Nama | Fokus | Hasil |
|---|---|---|---|
| **ricci1** | SPSA vs AdamW | Forward-only SPSA pada regresi kompleks | SPSA lambat (0.78×), AdamW underfit |
| **ricci2** | RCSSG Complex-Step | Estimasi gradien via complex-step | **NaN** — gagal total |
| **ricci3** | Fractal Subspace SPSA | Wavelet Haar 1D + SPSA | Stagnan di ~0.25 (tidak konvergen) |
| **ricci4** | SynapticLinear | Representasi polar + SPSA | **NaN** — gagal total |
| **ricci5** | Fisher Diagonal Probe | Whitening → diagonal Fisher | off-diag ratio 0.175→0.147 (16%↓) |
| **ricci6** | Whitening Impact | Whitening pada SPSA/AdamW | SPSA membaik (0.523→0.325) tapi gagal |
| **ricci7** | Grid Search SPSA | 15 konfigurasi × 3 seed | **Semua gagal** (terbaik 0.256 vs AdamW 0.020) |
| **ricci8** | FractalMLP v1 (salah) | Wavelet 2D + AdamW, MNIST | DWT stacking blok (salah), 98.1% vs 96.4% |
| **ricci9** | FractalMLP v2 (benar) | DWT interleaving + baseline 52K | Wavelet netral: 97.4–97.7% semua model |

### Jalur B (lr10–12): Kompresi Low-Rank
| # | Nama | Fokus | Hasil |
|---|---|---|---|
| **lr10** | Low-Rank Fashion-MNIST | Faktorisasi U@V pada MLP | LR32: 88.1% (−82% param) vs Reg256: 88.7% |
| **lr11** | Arithmetic 2-Digit | Penjumlahan 0–99 (10K kombinasi) | **Ceiling effect**: semua model 100% |
| **lr12** | Arithmetic 3-Digit | Penjumlahan 0–999 (1M kombinasi) | Reg128 (100%) > LR32 (99.7%), Reg64 (98.8%) > LR16 (98.5%) |

### Jalur C (mbmd13–16): Transformer untuk Penalaran Aritmetika
| # | Nama | Fokus | Model |
|---|---|---|---|
| **mbmd13** | Arithmetic LLM Mini | Transformer decoder untuk aritmetika | MiniGPT (d_model=128, 4 layer) |
| **mbmd14** | Reasoning-Enhanced | Penalaran langkah-demi-langkah | MiniGPT + <step> tokens |
| **mbmd15** | Neuro-Symbolic | Tool-use (<python>) + variasi bahasa alami | MiniGPT + eksekusi Python |
| **mbmd16** | Clean Solver | Tanpa bahasa alami, template matematika murni | MiniGPT + tool threshold |

---

## Output Eksperimental

### ricci1 — SPSA vs AdamW

**Dataset:** Regresi `x → exp(ix)`, 2000 sampel, model ComplexLinearMLP (hidden=64, aktivasi identitas)

```
=== AdamW ===
Epoch   0 | Train Loss: 3.011036e+00 | Val Loss: 9.641457e-01 | Time: 0.0383s
...
Epoch 190 | Train Loss: 9.447788e-01 | Val Loss: 1.931022e+00 | Time: 0.0358s

=== SPSA ===
Epoch   0 | Train Loss: 4.827568e-01 | Val Loss: 1.956406e+00 | Time: 0.0588s
...
Epoch 190 | Train Loss: 1.955449e-01 | Val Loss: 2.758420e-01 | Time: 0.0410s
```

| Metrik | AdamW | SPSA |
|---|---|---|
| Val Loss akhir | ~1.93 | ~0.276 |
| Rata-rata waktu/epoch | 0.0411 s | 0.0527 s |
| Rasio kecepatan (SPSA vs AdamW) | 1.00× | 0.78× |

---

### ricci2 — RCSSG Complex-Step

**Metode:** Random Coordinate Stochastic Subspace Gradient, complex-step differentiation, num_dirs=2

```
=== AdamW ===
Epoch   0 | Train Loss: 3.011036e+00 | Val Loss: 9.641457e-01 | Time: 0.0383s
Epoch 190 | Train Loss: 9.447788e-01 | Val Loss: 1.931022e+00 | Time: 0.0358s

=== RCSSG (num_dirs=2) ===
Epoch   0 | Train Loss: nan | Val Loss: nan | Time: 0.1682s
Epoch 190 | Train Loss: nan | Val Loss: nan | Time: 0.0675s

Cosine similarity antara gradien estimasi (2 arah) dan gradien asli: nan
```

**Kesimpulan:** RCSSG gagal total. NaN di semua metrik dari epoch pertama.

---

### ricci3 — Fractal Subspace SPSA

**Dataset:** Regresi `sin(x₁)cos(x₂)`, 2000 sampel, model MLP 3-layer (hidden=32, ReLU)

```
Training AdamW...
AdamW Epoch   0 | Train: 0.225094 | Val: 0.207187 | Time: 0.1885s
AdamW Epoch 140 | Train: 0.003387 | Val: 0.003973 | Time: 0.0344s

Training SPSA (all params)...
SPSA Epoch   0 | Train: 0.260673 | Val: 0.262652 | Time: 0.0302s
SPSA Epoch 140 | Train: 0.246985 | Val: 0.250615 | Time: 0.0291s

Training Fractal SPSA (low-frequency only)...
FracSPSA Epoch   0 | Train: 0.232493 | Val: 0.242712 | Time: 0.0429s
FracSPSA Epoch 140 | Train: 0.232311 | Val: 0.242543 | Time: 0.0408s
```

| Metrik | AdamW | SPSA | FracSPSA |
|---|---|---|---|
| Val Loss akhir | **0.00397** | 0.25062 | 0.24254 |
| Waktu/epoch | 0.0347 s | 0.0295 s | 0.0417 s |
| Konvergensi | Sempurna | Stagnan | Stagnan |

---

### ricci4 — SynapticLinear (Diagonal Fisher)

**Model:** SynapticMLP dengan parameterisasi polar `W_real = exp(a)·cos(ϕ)`

```
=== AdamW (RegularMLP) ===
Epoch   0 | Train: 0.237823 | Val: 0.226424 | Time: 0.0366s
Epoch 140 | Train: 0.002453 | Val: 0.003666 | Time: 0.0332s

=== SPSA (RegularMLP) ===
Epoch   0 | Train: 0.241461 | Val: 0.260414 | Time: 0.0278s
Epoch 140 | Train: 0.239272 | Val: 0.258555 | Time: 0.0271s

=== SPSA (SynapticMLP) ===
Epoch   0 | Train: nan | Val: nan | Time: 0.6154s
Epoch 140 | Train: nan | Val: nan | Time: 0.0381s
```

**Kesimpulan:** SynapticMLP + SPSA = NaN total. `exp(a)` menyebabkan ledakan numerik.

---

### ricci5 — Fisher Diagonalization Probe

**Model:** Linear 4→2, Fisher empiris dari 500 sampel

```
Off-diagonal ratio (Raw): 0.1747
Off-diagonal ratio (Whitened): 0.1473
```

| Metrik | Raw | Whitened | Delta |
|---|---|---|---|
| Off-diagonal ratio | 0.175 | 0.147 | **−16.0%** |

**Kesimpulan:** Whitening mengurangi off-diagonal Fisher 16% — efek positif tapi kecil.

---

### ricci6 — Dampak Whitening pada Optimasi

**Dataset:** `sin(x₁)cos(x₂)`, model MLP 2-layer (hidden=16), 4 kondisi

```
=== AdamW Raw ===
Epoch 140 | Train: 0.01489 | Val: 0.01989 | Time: 0.024s

=== AdamW White ===
Epoch 140 | Train: 0.01440 | Val: 0.01976 | Time: 0.026s

=== SPSA Raw ===
Epoch 140 | Train: 0.53704 | Val: 0.52315 | Time: 0.019s

=== SPSA White ===
Epoch 140 | Train: 0.34269 | Val: 0.32512 | Time: 0.019s
```

| Metrik | AdamW Raw | AdamW White | SPSA Raw | SPSA White |
|---|---|---|---|---|
| Val Loss | 0.0199 | 0.0198 | 0.523 | **0.325** |
| Waktu/epoch | 0.025 s | 0.025 s | 0.019 s | 0.019 s |

**Kesimpulan:** Whitening memperbaiki SPSA 38% — tapi masih 16× lebih buruk dari AdamW.

---

### ricci7 — Grid Search SPSA Hyperparameters

**Grid:** a ∈ {1e-2, 5e-3, 1e-3, 5e-4, 1e-4}, c ∈ {1e-2, 1e-3, 1e-4}, 3 seed, input whitened

```
a=1e-02, c=1e-03 : 0.2558 ± 0.0053   ← TERBAIK
a=5e-03, c=1e-03 : 0.2735 ± 0.0184
a=1e-03, c=1e-03 : 0.3373 ± 0.0765
a=5e-04, c=1e-03 : 0.3547 ± 0.0918
a=1e-04, c=1e-03 : 0.3714 ± 0.1064   ← TERBURUK
```

**Temuan kritis:** Parameter `c` tidak memiliki efek — hasil identik untuk semua nilai c pada a yang sama. **SPSA gagal di semua konfigurasi.**

---

### ricci8 — FractalMLP vs RegularMLP (DWT Salah)

**Dataset:** MNIST, model MLP 3-layer, AdamW

```
Regular MLP:
  Trainable params: 235146
Epoch 10 | Loss: 0.0141 | Test Acc: 0.9807 | Time: 12.944s

Fractal MLP (details frozen):
  Trainable params: 60042 (total: 235146)
Epoch 10 | Loss: 0.0634 | Test Acc: 0.9638 | Time: 13.401s
```

| Model | Params | Test Acc | Waktu/epoch |
|---|---|---|---|
| Regular MLP | 235K | **98.07%** | 13.1 s |
| Fractal MLP (frozen) | 60K | 96.38% | 13.4 s |

**Masalah:** Inverse DWT 2D salah (stacking blok, bukan interleaving).

---

### ricci9 — FractalMLP v2 (DWT Diperbaiki)

**Dataset:** MNIST, 4 kondisi, AdamW, 20 epoch

```
=== Regular MLP (235K) ===
Epoch 20 | Loss: 0.0099 | Test Acc: 0.9773 | Time: 13.33s

=== Regular MLP Small (52K) ===
Epoch 20 | Loss: 0.0168 | Test Acc: 0.9745 | Time: 13.58s

=== Fractal MLP (frozen, 60K) ===
Epoch 20 | Loss: 0.0080 | Test Acc: 0.9742 | Time: 13.48s

=== Fractal MLP (all trainable, 235K) ===
Epoch 20 | Loss: 0.0106 | Test Acc: 0.9769 | Time: 13.55s
```

| Model | Params | Test Acc | Kesimpulan |
|---|---|---|---|
| Regular 235K | 235K | **97.73%** | Baseline |
| Regular 52K | 52K | 97.45% | −0.28% |
| Fractal 60K (frozen) | 60K | 97.42% | ≈ Reg52K |
| Fractal 235K | 235K | 97.69% | ≈ Reg235K |

**Kesimpulan:** Wavelet netral — format representasi tidak memengaruhi performa pada kapasitas setara.

---

### lr10 — Low-Rank MLP pada Fashion-MNIST

**Dataset:** Fashion-MNIST, model 2-layer (784→hidden→10), AdamW

```
=== Regular 256 (params: 203530) ===
Ep 20 | Loss: 0.1763 | Test Acc: 0.8873 | Time: 7.80s

=== Regular 64 (params: 50890) ===
Ep 20 | Loss: 0.2419 | Test Acc: 0.8764 | Time: 7.71s

=== LowRank r=8 (params: 11146) ===
Ep 20 | Loss: 0.3142 | Test Acc: 0.8591 | Time: 7.87s

=== LowRank r=16 (params: 19466) ===
Ep 20 | Loss: 0.2535 | Test Acc: 0.8782 | Time: 7.82s

=== LowRank r=32 (params: 36106) ===
Ep 20 | Loss: 0.2186 | Test Acc: 0.8809 | Time: 7.74s
```

| Model | Params | vs Reg256 | Test Acc |
|---|---|---|---|
| Regular 256 | 203.530 | — | **88.73%** |
| LowRank r=32 | 36.106 | −82% | 88.09% (−0.64%) |
| LowRank r=16 | 19.466 | −90% | 87.82% (−0.91%) |
| Regular 64 | 50.890 | −75% | 87.64% (−1.09%) |
| LowRank r=8 | 11.146 | −95% | 85.91% (−2.82%) |

**Kesimpulan:** LR32 kompresi terbaik (−82% param, −0.64% acc). LR16 mengalahkan Reg64 dengan 62% param lebih sedikit.

---

### lr11 — Arithmetic 2-Digit (0–99)

**Dataset:** 60K sampel, 4 digit input → 3 digit output, model 2-hidden MLP

```
=== Regular 256 (params: 106846) ===
Ep 7 | Loss: 0.0060 | Test Acc: 1.0000

=== LowRank r=16 (params: 22878) ===
Ep 7 | Loss: 0.0087 | Test Acc: 1.0000
Ep 17 | Loss: 0.1608 | Test Acc: 0.9463  ← ketidakstabilan

=== LowRank r=32 (params: 37214) ===
Ep 6 | Loss: 0.0030 | Test Acc: 1.0000

=== Regular 80 (params: 19550) ===
Ep 12 | Loss: 0.0090 | Test Acc: 1.0000
```

**Kesimpulan:** Ceiling effect — semua model 100%. Regular 80 (19.550 param) adalah yang paling efisien.

---

### lr12 — Arithmetic 3-Digit (0–999)

**Dataset:** 60K sampel, 1M kemungkinan input, 6 digit input → 4 digit output

```
=== Regular 256 (params: 125800) ===
Ep 15 | Loss: 0.0012 | Test Acc: 1.0000

=== LowRank r=32 (params: 41832) ===
Ep 30 | Loss: 0.0204 | Test Acc: 0.9965

=== LowRank r=16 (params: 26472) ===
Ep 10 | Loss: 2.4494 | Test Acc: 0.1613  ← konvergensi sangat lambat
Ep 30 | Loss: 0.0384 | Test Acc: 0.9846

=== Regular 128 (params: 46696) ===
Ep 30 | Loss: 0.0018 | Test Acc: 0.9996

=== Regular 64 (params: 19432) ===
Ep 30 | Loss: 0.0501 | Test Acc: 0.9879
```

| Model | Params | Acc | Kesimpulan |
|---|---|---|---|
| Regular 256 | 125.800 | **100.0%** | Baseline |
| Regular 128 | 46.696 | **100.0%** | Pemenang |
| LowRank r=32 | 41.832 | 99.65% | Fluktuatif |
| Regular 64 | 19.432 | 98.79% | Paling efisien |
| LowRank r=16 | 26.472 | 98.46% | Konvergensi lambat |

**Kesimpulan:** Reg128 lebih unggul dari LR32. Reg64 lebih unggul dari LR16. Low-rank kalah dalam perbandingan parameter-matched.

---

### mbmd13 — Arithmetic LLM Mini (Transformer)

**Model:** MiniGPT (d_model=128, nhead=4, num_layers=4, ff=512). Dataset: 2M sampel aritmetika (+, -, *, 4 digit)

```
Total parameters: 2,173,540

Ep 1 Step 2000 | Loss: 2.1129 | Acc: 0.1085
Ep 1 Step 4000 | Loss: 1.7545 | Acc: 0.1085
...
Ep 15 Step 7000 | Loss: 1.4700 | Acc: 0.1085
=== Epoch 15 | Loss: 1.5189 | Token Acc: 0.1204 | Time: 1653.4s
```

**Test Exact Match Accuracy:** 0.0000 (model gagal generalisasi)

---

### mbmd14 — Reasoning-Enhanced

**Model:** MiniGPT dengan <step>...</step><result>...</result> format. Dataset: 200K sampel, 3-digit, tanpa perkalian

```
Total parameters: 2,199,750

Ep 1 Step 1000 | Loss: 1.2045 | Acc: 0.5782
Ep 20 Step 1000 | Loss: 0.0811 | Acc: 0.9720
=== Epoch 20 | Loss: 0.0954 | Token Acc: 0.9631 | Time: 347.4s
```

Exact result match: ~97% (dari 200 sampel)

---

### mbmd15 — Neuro-Symbolic

**Model:** MiniGPT + tool-use (<tool><python>print(...)</python></tool>). Dataset: 200K sampel, variasi bahasa alami.

**Fitur:** Tool threshold (≥3 digit → pakai tool). Template bahasa: 6 variasi. Dataset: 200K, max_digits=4.

```
Total parameters: 2,347,177

Ep 1 Step 1000 | Loss: 1.3276 | Acc: 0.5406
...
Ep 25 Step 1000 | Loss: 0.0826 | Acc: 0.9707
=== Epoch 25 | Loss: 0.1022 | Token Acc: 0.9628 | Time: 1451.2s
```

Neuro-Symbolic Exact Result Accuracy: ~95–97%

---

### mbmd16 — Clean Solver

**Model:** MiniGPT, template matematika murni (tanpa bahasa alami). Hanya token digit, operator, special.

```
Total parameters: 2,266,052

Epoch  1 | Loss: 1.1638 | Time: 485.5s
Epoch  5 | Loss: 0.1731 | Time: 483.1s
Epoch 10 | Loss: 0.0549 | Time: 480.5s
Epoch 15 | Loss: 0.0261 | Time: 479.5s

Prompt: <s>24*9=</s>
True : 24*9 = 216
Pred : <step>24*9=?</step><result>216</result>

Prompt: <s>683-158=</s>
True : 683-158 = 525
Pred : <step></step></step>13-5=8 borrow1</step>...<result>525</result>

Prompt: <s>442+120=</s>
True : 442+120 = 562
Pred : <step>2+0+0=2 carry0</step><step>4+2+0=6 carry0</step><step>4+1+0=5 carry0</step><result>562</result>
```

---

## Ringkasan Fase

| Fase | Eksperimen | Target | Metrik Utama | Temuan Kunci |
|---|---|---|---|---|
| **Fase 1: SPSA** | ricci1–7 | Mengganti backprop dengan forward-only | Val loss, kecepatan | **SPSA gagal total** — semua varian (SPSA, RCSSG, FractalSPSA, Synaptic) tidak konvergen. Grid search membuktikan SPSA tidak bisa bersaing. |
| **Fase 2: Whitening** | ricci5–6 | Memperbaiki conditioning | Off-diag Fisher, konvergensi | Whitening mengurangi off-diag Fisher 16% dan memperbaiki SPSA 38% — tapi tidak cukup. |
| **Fase 3: Kompresi** | ricci8–9, lr10–12 | Mengurangi parameter | Akurasi vs param count | Low-rank LR32 (−82% param, −0.6% acc) adalah kompresi terbaik. Wavelet netral. |
| **Fase 4: Transformer** | mbmd13–16 | Penalaran aritmetika | Exact match accuracy | MiniGPT 4-layer dapat mencapai 95–97% exact match pada aritmetika 3–4 digit dengan tool-use. |

---

## Kesimpulan Akhir

1. **SPSA tidak viable** untuk optimasi neural network — 7 eksperimen, semua gagal.
2. **Whitening membantu** tapi tidak menyelamatkan SPSA — efek positif marjinal.
3. **Low-rank decomposition (r=32)** memberikan kompresi terbaik: −82% parameter dengan −0.6% akurasi.
4. **Wavelet Haar** netral — tidak meningkatkan atau menurunkan performa pada kapasitas setara.
5. **Transformer mini (4-layer, d_model=128)** mampu mempelajari aritmetika 3–4 digit dengan tool-use hingga 97% exact match.
6. **Proyek bergeser** dari klaim Ricci-Kähler Flow (tidak terbukti) ke kompresi low-rank (terbukti) ke arsitektur Transformer untuk penalaran (terbukti).
