# Crowd Counting with CSRNet 

Proyek ini saya dan rekan saya Arifa kerjakan dalam kompetisi **Hology 8.0 Data Mining 2025**.  
Meskipun tidak berhasil masuk final, saya dokumentasikan karena saya ingin menunjukkan bagaimana kami memahami masalah, menganalisis data, dan membangun solusi deep learning dari awal.  

---

## Memahami Masalah
Tugas kompetisi ini adalah **menghitung jumlah orang dalam citra kerumunan**.  
Tantangannya cukup nyata:
- Kepala bisa sangat kecil, terutama pada crowd yang padat.
- Kondisi pencahayaan dan latar belakang bervariasi.
- Beberapa objek non-manusia bisa “menyerupai” kepala dan membingungkan model.

Karena alasan itu, sekadar memprediksi angka total secara langsung biasanya tidak cukup. Kami memilih pendekatan berbasis **density map regression**: model dilatih untuk menghasilkan peta kepadatan orang, lalu jumlah diperoleh dengan menjumlahkan semua nilai density.

---

## Eksplorasi Data (EDA)

Sebelum membangun model, kami melihat seperti apa data train (1900 gambar + JSON titik kepala) dan test (500 gambar).

### 1. Distribusi Jumlah Kepala
Kami hitung jumlah titik kepala di setiap JSON untuk melihat sebarannya.  
Hasil histogram memperlihatkan variasi yang cukup besar: ada gambar dengan puluhan orang, ada juga yang lebih dari seribu.  


### 2. Visualisasi Titik Kepala
Untuk verifikasi label, kami overlay titik dari JSON di atas gambar.  
Hasilnya terlihat cukup konsisten: setiap titik mewakili posisi kepala.  
data-with-label.png

### 3. Dari Titik ke Density Map
Label JSON kemudian kami ubah ke **density map** menggunakan metode **NormDM** (Normalised Density Map) dengan sigma adaptif berbasis jarak tetangga terdekat.  
Density ini bersifat “preserve count”: integral dari peta ≈ jumlah titik.  
density.png

EDA ini membuat kami yakin bahwa pendekatan density regression memang cocok untuk masalah ini.

---

## Ide Solusi

Berdasarkan pemahaman masalah, kami merancang solusi dengan poin utama:

1. **Model CSRNet**  
   - Backbone: VGG16-BN pretrained ImageNet, tanpa pool5.  
   - Backend: dilated convolution stack.  
   - Output: 1 channel density map, resolusi 1/8 dari input.

2. **Loss Function Combo**  
   - MSE pada density map (untuk detail lokal).  
   - L1 loss pada jumlah total (untuk kalibrasi global).  
   - Opsional: SSIM loss kecil (untuk kualitas struktur).

3. **Training Strategy**  
   - Freeze backbone pada 5 epoch awal → stabilisasi.  
   - **Progressive patch training**: 384 → 512 → 640.  
   - Augmentasi: scale, crop, flip horizontal, brightness/contrast.

4. **Inference**  
   - Resize gambar dengan batas sisi maksimum.  
   - **Test-time augmentation (TTA)** multi-scale, prediksi akhir = rata-rata integral density.

---

## Eksperimen & Proses Belajar

Kami membagi proses ke dua tahap:

### Tahap 1 – Training Utama
- Training dengan progressive patch.  
- Hasil validasi awal menunjukkan MAE masih tinggi (~31–32).  
- Namun sudah ada indikasi bahwa model belajar pola crowd secara umum.  

### Tahap 2 – Fine-tuning
- Menyesuaikan loss: meningkatkan bobot L1 count, menonaktifkan SSIM.  
- Menggunakan patch lebih besar (512–640).  
- Menurunkan learning rate dan evaluasi lebih sering.  
- MAE pada validasi membaik signifikan, mencapai ~15–18.  


---

## Analisis Error

Kami tidak hanya berhenti di angka. Kami juga lihat prediksi pada gambar validasi:

- **Undercount**: biasanya terjadi pada crowd sangat padat di area gelap/blur.  
- **Overcount**: sering muncul di pola berulang (poster, background dengan tekstur).  
- Visualisasi GT vs pred membantu mengidentifikasi pola ini.  
predict.png

---

## Insight & Pembelajaran

Dari proyek ini, kami belajar beberapa hal penting:
- **EDA menentukan arah solusi**: dengan melihat distribusi count & visualisasi titik, kami tahu direct regression tidak ideal.  
- **Density regression + count supervision** jauh lebih robust untuk crowd padat.  
- **Dokumentasi eksperimen** (EDA, visualisasi, error analysis) membantu menjelaskan proses berpikir, bukan sekadar angka leaderboard.  

Walaupun tidak memenangkan kompetisi, kami bangga karena kami bisa menyusun pipeline end-to-end yang rapi dan reproducible.

---

## Next Steps
Beberapa ide perbaikan yang kami catat:
- Tiling inference untuk gambar beresolusi sangat besar.  
- Augmentasi khusus crowd (misalnya cutmix berbasis head region).  
- Model alternatif: SCARNet, DM-Count, atau backbone lebih ringan.  
- Post-calibration di domain count (misalnya isotonic regression).  

---

👤 **Muhammad Naufal Aqil**  | **Arifa Alayya Firajabi**
📌 Machine Learning • Computer Vision • Medical Informatics  
🔗 [LinkedIn](https://www.linkedin.com/in/muhammad-naufal-aqil-b6114424a/)

