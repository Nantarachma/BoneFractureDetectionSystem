# 🦴 Bone Fracture Detection System

Sistem deteksi fraktur tulang pada citra X-Ray menggunakan model Deep Learning berbasis **DETR** (*Detection Transformer*) dari HuggingFace, dengan antarmuka web interaktif yang dibangun menggunakan **Streamlit**.

---

## ✨ Fitur

| Fitur | Deskripsi |
|---|---|
| **Deteksi Otomatis** | Deteksi area fraktur pada citra X-Ray menggunakan model DETR |
| **Confidence Threshold** | Slider untuk mengatur ambang batas minimum confidence score |
| **Non-Maximum Suppression** | Menghilangkan deteksi duplikat yang saling tumpang-tindih (otomatis) |
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
- Koneksi internet (model akan diunduh otomatis dari HuggingFace Hub: [`nantarach/bone-fracture-detr`](https://huggingface.co/nantarach/bone-fracture-detr))

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
├── .gitignore          # File yang dikecualikan dari Git
└── README.md           # Dokumentasi proyek
```

---

## 🎯 Cara Penggunaan

1. **Buka aplikasi** di browser setelah menjalankan `streamlit run app.py` (default: `http://localhost:8501`)
2. **Atur Confidence Threshold** pada sidebar di sebelah kiri — geser slider untuk menentukan ambang batas minimum confidence score deteksi (default: 0.5)
3. **Unggah citra X-Ray** melalui area upload di halaman utama (format: JPG, JPEG, atau PNG)
4. **Periksa preview** — gambar yang diunggah akan tampil bersama metadata (dimensi, ukuran file, rasio aspek)
5. **Klik tombol** "🔍 Proses Deteksi Fraktur" untuk memulai analisis
6. **Lihat ringkasan** — kartu metrik menampilkan jumlah fraktur, rata-rata confidence, confidence tertinggi, dan confidence terendah
7. **Bandingkan citra** — buka tab "Perbandingan Citra" untuk melihat citra asli dan hasil deteksi secara berdampingan
8. **Cek detail** — buka tab "Detail Deteksi" untuk melihat tabel lengkap setiap fraktur (confidence, koordinat, luas area)
9. **Unduh hasil** — klik tombol unduh untuk menyimpan gambar hasil anotasi (tersedia format PNG dan JPEG)
10. **Pantau riwayat** — sidebar mencatat riwayat setiap deteksi yang dijalankan selama sesi

---

## 🔧 Konfigurasi untuk Developer

Berikut adalah variabel-variabel *hardcoded* di bagian atas file `app.py` yang dapat Anda ubah sesuai kebutuhan:

| Variabel | Default | Deskripsi | Cara Mengubah |
|---|---|---|---|
| `MODEL_CHECKPOINT` | `"nantarach/bone-fracture-detr"` | ID model HuggingFace Hub atau path checkpoint lokal | Ganti dengan model ID HuggingFace atau path lokal, misalnya `"facebook/detr-resnet-50"` |
| `DEFAULT_CONFIDENCE` | `0.5` | Nilai default slider confidence threshold | Ubah ke angka antara `0.0`-`1.0`; nilai lebih rendah = lebih banyak deteksi (lebih sensitif), nilai lebih tinggi = lebih sedikit deteksi (lebih presisi) |
| `NMS_IOU_THRESHOLD` | `0.5` | Ambang batas IoU untuk Non-Maximum Suppression | Ubah ke angka antara `0.0`-`1.0`; nilai lebih rendah = penyaringan lebih ketat (lebih sedikit overlap), nilai lebih tinggi = mengizinkan lebih banyak overlap |
| `MAX_IMAGE_SIDE` | `800` | Ukuran sisi terpanjang gambar saat preprocessing (piksel) | Naikkan untuk resolusi input lebih tinggi (membutuhkan lebih banyak memori), turunkan untuk inferensi lebih cepat |
| `BOX_COLOR` | `(99, 102, 241)` | Warna bounding box dalam format RGB | Ganti tuple RGB, misalnya `(255, 0, 0)` untuk merah |
| `TEXT_BG_COLOR` | `(99, 102, 241)` | Warna latar belakang label teks (RGB) | Sesuaikan dengan `BOX_COLOR` agar konsisten |
| `TEXT_COLOR` | `(255, 255, 255)` | Warna teks label (RGB) | Pastikan kontras yang cukup terhadap `TEXT_BG_COLOR` |
| `FRACTURE_LABEL` | `"fracture"` | Label kelas fraktur untuk filtering hasil deteksi | Ubah jika model Anda menggunakan label kelas yang berbeda |

### Contoh Mengubah Konfigurasi

```python
# ---- Di bagian atas app.py ----

# Menggunakan model dari HuggingFace Hub
MODEL_CHECKPOINT = "nantarach/bone-fracture-detr"

# Meningkatkan sensitivitas deteksi (lebih banyak hasil, termasuk yang kurang yakin)
DEFAULT_CONFIDENCE = 0.3

# Membuat NMS lebih ketat (menghilangkan lebih banyak deteksi yang overlap)
NMS_IOU_THRESHOLD = 0.3

# Menggunakan bounding box warna merah
BOX_COLOR = (239, 68, 68)
TEXT_BG_COLOR = (239, 68, 68)
```

---

## 🛠️ Teknologi

- **Model**: DETR (*DEtection TRansformer*) — arsitektur end-to-end object detection
- **Framework**: PyTorch + HuggingFace Transformers
- **UI**: Streamlit dengan CSS kustom
- **Visualisasi**: Pillow (PIL)