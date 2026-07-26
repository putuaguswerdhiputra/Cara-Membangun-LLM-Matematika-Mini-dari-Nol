# Experiments: Perjalanan Membangun LLM Matematika Mini

Folder ini berisi seluruh kode eksperimen yang dilakukan selama riset **"Dari Kegagalan ke Neuro-Symbolic: Perjalanan Membangun LLM Matematika Mini dengan Bantuan AI dan Sumber Daya Terbatas"** (Juli 2026). Setiap eksperimen dirancang, dilatih, dan dievaluasi di Kaggle Notebook dengan 2× NVIDIA T4 GPU.

## Daftar Eksperimen

| No | Kode | Nama Eksperimen | Deskripsi Singkat | Status | File Kunci |
|----|------|-----------------|-------------------|--------|------------|
| 1 | #001 | SPSA vs AdamW | Optimasi zeroth‑order pada regresi kompleks | ❌ Gagal | `ricci1_spsa_adamw.py` |
| 2 | #002 | RCSSG | Estimasi gradien dengan complex step | ❌ NaN | `ricci2_complex_step.py` |
| 3 | #003 | Fractal SPSA | Kompresi wavelet + SPSA | ❌ Stagnan | `ricci3_fractal_spsa.py` |
| 4 | #004 | SynapticLinear | Representasi polar untuk diagonal Fisher | ❌ NaN | `ricci4_synaptic.py` |
| 5 | #005 | Fisher Probe | Pengaruh whitening pada FIM | ⚠️ Marginal | `ricci5_fisher_probe.py` |
| 6 | #006 | Whitening Impact | Whitening input + SPSA | ❌ Gagal | `ricci6_whitening_impact.py` |
| 7 | #007 | Grid Search SPSA | Tuning hyperparameter SPSA | ❌ Gagal | `ricci7_grid_search.py` |
| 8 | #008 | FractalMLP v1 | Kompresi wavelet di MNIST | ⚠️ Bug DWT | `ricci8_fractal_mlp.py` |
| 9 | #009 | FractalMLP v2 | Koreksi DWT, perbandingan adil | ⚖️ Netral | `ricci9_fractal_v2.py` |
| 10 | #010 | Low‑Rank MLP | Low‑rank pada Fashion‑MNIST | ⚖️ Setara | `lr10_lowrank_fashionmnist.py` |
| 11 | #011 | LR Arithmetic 2d | Penjumlahan 2‑digit (ceiling effect) | ❌ Tak informatif | `lr11_arithmetic_2d.py` |
| 12 | #012 | LR Arithmetic 3d | Penjumlahan 3‑digit | ⚖️ Kalah | `lr12_arithmetic_3d.py` |
| 13 | MBMD‑13 | Transformer‑langsung | MiniGPT soal→jawaban | ❌ Exact 0% | `mbmd13_arithmetic_llm.py` |
| 14 | MBMD‑14 | Reasoning‑Enhanced | MiniGPT + jejak penalaran | ❌ Exact rendah | `mbmd14_reasoning.py` |
| 15 | MBMD‑15 | Neuro‑Symbolic | MiniGPT + tool‑use, variasi bahasa | ❌ Mode collapse | `mbmd15_neuro_symbolic.py` |
| 16 | MBMD‑16 | Clean Solver | Tokenizer bersih, tanpa bahasa alami | ❌ Overfitting | `mbmd16_clean_solver.py` |

## Cara Menjalankan

1. Unggah file `.py` ke Kaggle Notebook (Python 3, GPU T4×2).
2. Jalankan seluruh sel. Setiap kode akan melatih model dari nol.
3. Hasil pelatihan (model `.pt`) akan tersimpan di `/kaggle/working`.

**Catatan:** Beberapa eksperimen membutuhkan waktu pelatihan yang cukup lama (hingga 2–3 jam). Pastikan sesi Kaggle tetap aktif.

## Referensi

- [Laporan lengkap](../README.md) (makalah utama)
- [Repositori GitHub](https://github.com/putuaguswerdhiputra)

---

*Dokumentasi ini adalah bukti bahwa kegagalan yang tercatat dengan baik adalah fondasi pengetahuan yang kokoh.*
