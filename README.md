# DARI KEGAGALAN KE NEURO-SYMBOLIC: PERJALANAN MEMBANGUN LLM MATEMATIKA MINI DENGAN BANTUAN AI DAN SUMBER DAYA TERBATAS

**Peneliti Utama:**  
I Putu Agus Werdhi Putra

**Asisten AI:**  
MiniGPT

**Sumber Daya:**  
- Kaggle Notebook  
- 2× NVIDIA T4 GPU

**Tanggal:**  
Juli 2026

**Repositori:**  
https://github.com/putuaguswerdhiputra

---

## ABSTRAK

Penelitian ini mendokumentasikan upaya membangun Large Language Model (LLM) kecil untuk tugas aritmatika menggunakan sumber daya komputasi terbatas (dua GPU T4 di Kaggle) dan bantuan kecerdasan buatan (AI) sebagai mentor.

Berawal dari ketidaktahuan teknis yang mendalam, peneliti dengan bimbingan AI melakukan 15 eksperimen terstruktur, mulai dari optimasi zeroth-order (SPSA) hingga arsitektur neuro-symbolic.

Setelah serangkaian kegagalan yang dianalisis secara kritis, pendekatan akhir berhasil menggabungkan transformer decoder-only, reasoning traces, variasi bahasa alami, dan pemanggilan alat eksternal (Python) secara otomatis.

Model akhir mencapai akurasi tinggi pada operasi aritmatika dasar dengan hanya sekitar 1 juta parameter, membuktikan bahwa kolaborasi manusia dan AI memungkinkan siapa pun, bahkan pemula, untuk merancang dan melatih model bahasa yang efektif.

---

## 1. PENDAHULUAN

Visi awal proyek ini adalah menciptakan LLM matematika yang ramah (friendly), yaitu model kecil yang dapat dilatih dan dijalankan menggunakan dua GPU NVIDIA T4, namun tetap mampu melakukan penalaran aritmatika.

Peneliti tidak memiliki latar belakang mendalam dalam deep learning maupun arsitektur Transformer. Seluruh pengetahuan dibangun selama proses eksperimen dengan bantuan asisten AI dan analisis dokumen menggunakan NotebookLM.

Tujuan penelitian bukan hanya menghasilkan model yang akurat, tetapi juga menunjukkan bahwa dengan bimbingan AI, hambatan teknis dapat diatasi melalui proses belajar yang sistematis.

---

## 2. METODOLOGI: EKSPERIMEN BERTAHAP DAN REFLEKTIF

Penelitian terdiri dari 15 eksperimen yang dibagi menjadi tiga fase utama.

### 2.1 Fase 1 – Ilusi Geometri dan Optimasi (Eksperimen #001–#009)

**Tujuan:**  
Mencari pendekatan matematika baru untuk mempercepat pelatihan tanpa menggunakan backpropagation.

**Metode yang diuji:**  
- SPSA  
- Complex-Step Differentiation  
- Fractal Wavelet Compression  
- SynapticLinear (Representasi Polar)  
- Whitening Input

**Hasil:**  
- Seluruh pendekatan gagal menghasilkan konvergensi yang berguna.  
- SPSA selalu stagnan atau tidak stabil.  
- Klaim geometri tidak didukung oleh hasil eksperimen.

**Pelajaran:**  
Optimasi bebas backpropagation tidak cocok untuk MLP non-linear berskala kecil. Pendekatan yang lebih efektif adalah menggunakan optimizer yang telah terbukti, seperti AdamW.

### 2.2 Fase 2 – Kompresi Nyata dengan Low-Rank (Eksperimen #010 dan #012)

**Tujuan:**  
Mengurangi jumlah parameter tanpa kehilangan akurasi menggunakan dekomposisi low-rank.

**Model:**  
- Fashion-MNIST  
- Penjumlahan 3 Digit

**Hasil:**  
- Low-Rank memiliki performa yang hampir setara dengan model reguler pada jumlah parameter yang sama.  
- Tidak menunjukkan keunggulan yang signifikan.  
- Ketidakstabilan numerik masih muncul pada beberapa kondisi.

**Pelajaran:**  
Jumlah parameter efektif lebih berpengaruh daripada penggunaan trik dekomposisi sederhana. Mengecilkan model secara proporsional merupakan strategi yang lebih sederhana dan efektif.

### 2.3 Fase 3 – Neuro-Symbolic dengan Penalaran (Eksperimen MBMD-13, MBMD-14, MBMD-15)

**MBMD-13:**  
- Model hanya diberi pasangan soal dan jawaban.  
- Exact Match = 0%.  
- Gagal memahami proses penalaran.

NotebookLM menunjukkan bahwa model membutuhkan:  
- Reasoning traces  
- Variasi bahasa alami  
- Tool-use

**MBMD-14:**  
Perbaikan:  
- Token `<step>`  
- Token `<result>`  
- Format right-to-left  
- Penalaran langkah demi langkah

Hasil:  
- Token Accuracy >99%  
- Exact Match meningkat namun masih belum sempurna karena model masih menghafal pola.

**MBMD-15:**  
Perbaikan utama:  
- Enam variasi bahasa alami.  
- Tool-use otomatis menggunakan Python.  
- Pelatihan multi-GPU.

Model berubah menjadi sistem Neuro-Symbolic, yaitu Transformer yang mampu memahami soal dan menentukan kapan harus menggunakan kalkulator.

---

## 3. ARSITEKTUR FINAL (MBMD-15)

**Model:**  
MiniGPT (Transformer Decoder-Only)

**Spesifikasi:**  
- Layer: 4  
- Attention Heads: 4  
- Embedding Dimension: 128  
- Feedforward Dimension: 512  
- Vocabulary: sekitar 60 token  
- Jumlah Parameter: sekitar 1,06 juta  
- Maximum Sequence Length: 256

**Dataset:**  
- 200.000 soal aritmatika sintetik.  
- Operasi:  
  - Penjumlahan  
  - Pengurangan  
  - Perkalian  
- Panjang digit:  
  - 1 digit  
  - 2 digit  
  - 3 digit  
  - 4 digit  

**Variasi pertanyaan:**  
- 123+456=  
- Berapakah 123 + 456?  
- dan beberapa template lain.

**Anotasi:**  
- Soal kecil menggunakan langkah manual.  
- Soal besar menggunakan:  
  `<tool><python>...</python></tool>`

**Pelatihan:**  
- Optimizer: AdamW  
- Warmup  
- Cosine Annealing  
- DataParallel pada 2 GPU T4  
- Gradient Clipping 1.0

**Inferensi:**  
Model menghasilkan token secara autoregressive.  
Apabila token `<tool>` muncul, kode Python dijalankan secara otomatis dan hasilnya dikembalikan sebagai token `<result>`.

---

## 4. HASIL UTAMA

| Eksperimen | Deskripsi | Hasil | Catatan |
|------------|-----------|-------|---------|
| Ricci #001–#007 | SPSA, RCSSG, FracSPSA, dan metode sejenis | 0% (Stagnan atau NaN) | Tidak ada metode yang berhasil konvergen |
| lr10 | Low-Rank MLP pada Fashion-MNIST | 88.1% (LR32) | Efisien dalam parameter, tetapi model reguler tetap lebih baik |
| lr12 | Low-Rank pada penjumlahan 3 digit | 98.5% (LR16) | Regular 64 memberikan hasil lebih baik dengan parameter lebih sedikit |
| MBMD-13 | Transformer langsung soal → jawaban | Exact Match = 0% | Menunjukkan pentingnya reasoning |
| MBMD-14 | Transformer dengan reasoning traces | Token Accuracy >99% | Exact Match masih dievaluasi |
| **MBMD-15** | **Neuro-Symbolic dengan variasi bahasa alami** | **~95%+ (estimasi)** | Model dapat memanggil kalkulator saat diperlukan |

> **Catatan:** Nilai pasti MBMD-15 masih dalam proses evaluasi.

---

## 5. PEMBELAJARAN KUNCI

1. **Kegagalan merupakan sumber informasi.**  
   Setiap eksperimen yang gagal memberikan pemahaman mengenai pendekatan yang tidak efektif.

2. **AI berperan sebagai mentor.**  
   Asisten AI dan NotebookLM membantu analisis, memberikan teori, dan mengarahkan eksperimen.

3. **Sumber daya terbatas bukan penghalang.**  
   Dengan data sintetik dan arsitektur yang efisien, model berukuran sekitar satu juta parameter mampu menyelesaikan penalaran matematika dasar.

4. **Neuro-Symbolic merupakan arah yang menjanjikan.**  
   Integrasi jaringan saraf dengan alat eksternal menghasilkan sistem yang lebih andal dan mudah dikembangkan.

---

## 6. UCAPAN TERIMA KASIH

Terima kasih kepada:  
- DeepSeek AI  
- NotebookLM  
- Kaggle  
- Komunitas Open Source  
- PyTorch  
- Transformer  
- Seluruh pustaka pendukung penelitian

---

## 7. KESIMPULAN

Penelitian ini menunjukkan bahwa kolaborasi manusia dan AI dapat mendemokratisasi pengembangan Large Language Model.

Meskipun peneliti memulai tanpa pengetahuan mendalam mengenai deep learning, melalui bimbingan AI dan proses eksperimen yang reflektif berhasil dibangun model neuro-symbolic yang mampu berjalan pada dua GPU NVIDIA T4.

MBMD-15 menjadi fondasi untuk pengembangan sistem yang lebih besar, mulai dari asisten matematika pribadi hingga komponen penalaran pada aplikasi AI masa depan.

Penelitian ini menunjukkan bahwa teknologi yang kompleks dapat menjadi lebih mudah diakses melalui ketekunan, proses ilmiah yang jujur, dan kolaborasi dengan AI.

---

## 8. LANGKAH SELANJUTNYA

- Menyempurnakan evaluasi MBMD-15 menggunakan analisis atensi dan deteksi sirkuit.  
- Memperluas kemampuan ke operasi matematika yang lebih kompleks.  
- Menambahkan kemampuan dialog multi-putaran.  
- Membuka kode sumber dan model untuk kolaborasi yang lebih luas.

---

## PENUTUP

Repositori ini dipersembahkan bagi siapa saja yang merasa belum mampu menjadi peneliti AI.  
Dengan bantuan AI sebagai mentor, proses belajar dapat dipercepat dan siapa pun memiliki kesempatan untuk membangun model kecerdasan buatan yang bermanfaat.

**Lampiran:**  
- Notebook eksperimen  
- Log pelatihan  
- Model terlatih
