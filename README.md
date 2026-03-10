# 🦴 Bone Fracture Detection System

Sistem deteksi fraktur tulang pada citra X-Ray menggunakan model Deep Learning berbasis **DETR** (*Detection Transformer*) dari HuggingFace, dengan antarmuka web interaktif yang dibangun menggunakan **Streamlit**.

---

## ✨ Fitur

| Fitur | Deskripsi |
|---|---|
| **Deteksi Otomatis** | Deteksi area fraktur pada citra X-Ray menggunakan model DETR |
| **Confidence Threshold** | Slider untuk mengatur ambang batas minimum confidence score |
| **Non-Maximum Suppression** | Menghilangkan deteksi duplikat yang saling tumpang-tindih |
| **Perbandingan Citra** | Tab perbandingan side-by-side antara citra asli dan hasil deteksi |
| **Detail Deteksi** | Tabel detail setiap fraktur dengan confidence, koordinat, dan luas area |
| **Metrik Ringkasan** | Kartu metrik: jumlah fraktur, rata-rata confidence, tertinggi, terendah |
| **Metadata Gambar** | Informasi dimensi, ukuran file, format, dan rasio aspek |
| **Unduh Hasil** | Unduh gambar hasil anotasi dalam format PNG atau JPEG |
| **Riwayat Sesi** | Sidebar menampilkan riwayat deteksi selama sesi berjalan |
| **UI Responsif** | Tampilan gelap modern dengan layout wide dan sidebar pengaturan |

---

## 🚀 Instalasi & Menjalankan

### Prasyarat
- Python 3.9+
- File checkpoint model DETR (`model.ckpt`) di direktori root proyek

### Langkah Instalasi

```bash
# Clone repository
git clone https://github.com/Nantarachma/BoneFractureDetectionSystem.git
cd BoneFractureDetectionSystem

# Buat virtual environment (opsional, disarankan)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependensi
pip install -r requirements.txt
```

### Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan berjalan di `http://localhost:8501`.

---

## 📦 Dependensi

| Paket | Versi | Fungsi |
|---|---|---|
| `streamlit` | ≥1.32.0 | Framework antarmuka web |
| `torch` | ≥2.0.0 | Deep learning framework |
| `torchvision` | ≥0.15.0 | Utilitas computer vision & NMS |
| `transformers` | ≥4.35.0 | Model DETR dari HuggingFace |
| `Pillow` | ≥9.4.0 | Pemrosesan & visualisasi gambar |

---

## 🏗️ Arsitektur

```
Input Citra X-Ray (JPG/PNG)
        │
        ▼
┌─────────────────┐
│ DetrImageProcessor │  Preprocessing: resize, normalisasi, padding
└────────┬────────┘
         ▼
┌─────────────────┐
│ DetrForObjectDetection │  Inferensi: deteksi objek berbasis Transformer
└────────┬────────┘
         ▼
┌─────────────────┐
│ Post-Processing  │  Konversi koordinat, filter threshold, NMS
└────────┬────────┘
         ▼
┌─────────────────┐
│  Visualisasi     │  Bounding box + label confidence pada gambar
└────────┬────────┘
         ▼
   Hasil Deteksi (Annotated Image + Metrik)
```

---

## 📁 Struktur Proyek

```
BoneFractureDetectionSystem/
├── app.py              # Aplikasi utama (Streamlit)
├── requirements.txt    # Dependensi Python
├── model.ckpt          # Checkpoint model DETR (tidak di-commit)
├── .gitignore          # File yang dikecualikan dari Git
└── README.md           # Dokumentasi proyek
```

---

## 🎯 Cara Penggunaan

1. **Buka aplikasi** di browser (`http://localhost:8501`)
2. **Sesuaikan parameter** di sidebar (Confidence Threshold, NMS IoU)
3. **Unggah citra X-Ray** melalui area upload (format JPG/PNG)
4. **Klik tombol** "🔍 Proses Deteksi Fraktur"
5. **Lihat hasil** pada tab Perbandingan Citra dan Detail Deteksi
6. **Unduh gambar** hasil anotasi dalam format PNG atau JPEG

---

## 🛠️ Teknologi

- **Model**: DETR (*DEtection TRansformer*) — arsitektur end-to-end object detection
- **Framework**: PyTorch + HuggingFace Transformers
- **UI**: Streamlit dengan CSS kustom
- **Visualisasi**: Pillow (PIL)