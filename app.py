"""
Sistem Deteksi Fraktur Tulang pada Citra X-Ray
Menggunakan model Deep Learning berbasis PyTorch dengan arsitektur
Detection Transformer (DETR) dari HuggingFace.
"""

import io
import torch
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from transformers import DetrForObjectDetection, DetrImageProcessor

# ---------------------------------------------------------------------------
# Konfigurasi halaman Streamlit (wajib dipanggil sebelum elemen UI lainnya)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Deteksi Fraktur Tulang",
    page_icon="🦴",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Injeksi CSS untuk tampilan minimalis dan modern
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Warna latar utama */
        .stApp {
            background-color: #0f1117;
            color: #e0e0e0;
        }

        /* Judul utama */
        h1 {
            text-align: center;
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: 0.5px;
        }

        /* Sub-judul / deskripsi */
        .subtitle {
            text-align: center;
            color: #9ca3af;
            font-size: 0.95rem;
            margin-bottom: 2rem;
        }

        /* Kartu pembungkus konten */
        .card {
            background-color: #1c1f26;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            border: 1px solid #2d3141;
        }

        /* Label kecil di atas gambar */
        .section-label {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #6b7280;
            margin-bottom: 0.5rem;
        }

        /* Tombol utama */
        div.stButton > button {
            width: 100%;
            background-color: #3b82f6;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s ease;
        }
        div.stButton > button:hover {
            background-color: #2563eb;
        }

        /* Sembunyikan toolbar gambar bawaan Streamlit */
        [data-testid="stImage"] > div {
            border-radius: 8px;
            overflow: hidden;
        }

        /* Divider tipis */
        hr {
            border-color: #2d3141;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
MODEL_CHECKPOINT = "model.ckpt"   # Checkpoint lokal model DETR
CONFIDENCE_THRESHOLD = 0.5         # Ambang batas confidence score
MAX_IMAGE_SIDE = 800               # Panjang sisi terpanjang untuk resize (px)
BOX_COLOR = (59, 130, 246)         # Warna bounding box (biru, RGB)
TEXT_BG_COLOR = (59, 130, 246)     # Warna latar label teks
TEXT_COLOR = (255, 255, 255)       # Warna teks label

# Indeks kelas fraktur sesuai label_id model DETR yang dilatih.
# Model ini menggunakan dua kelas: 0 = background/"no-object", 1 = fraktur.
FRACTURE_CLASS_INDEX = 1


# ===========================================================================
# 1. FUNGSI MUAT MODEL
#    Menggunakan @st.cache_resource agar model hanya di-load sekali.
# ===========================================================================
@st.cache_resource(show_spinner="Memuat model DETR…")
def load_model():
    """
    Memuat DetrImageProcessor dan DetrForObjectDetection dari checkpoint lokal.
    Model di-set ke mode evaluasi untuk inferensi.
    """
    # Muat processor untuk preprocessing gambar (resize, normalisasi, padding)
    processor = DetrImageProcessor.from_pretrained(MODEL_CHECKPOINT)

    # Muat bobot model DETR dari checkpoint lokal
    model = DetrForObjectDetection.from_pretrained(MODEL_CHECKPOINT)

    # Set ke mode evaluasi: menonaktifkan dropout dan batch norm tracking
    model.eval()

    return processor, model


# ===========================================================================
# 2. FUNGSI PREPROCESSING & INFERENSI
#    Menangani konversi gambar → tensor, inferensi model, dan post-processing.
# ===========================================================================
def run_inference(image: Image.Image, processor, model):
    """
    Melakukan inferensi DETR pada gambar PIL yang diberikan.

    Args:
        image: Gambar asli dalam format PIL.Image (RGB).
        processor: DetrImageProcessor yang sudah di-load.
        model: DetrForObjectDetection yang sudah di-load.

    Returns:
        List of dict dengan kunci 'box' (xyxy piksel) dan 'score' (float).
    """
    orig_width, orig_height = image.size

    # --- Preprocessing ---
    # DetrImageProcessor secara otomatis menangani:
    #   • Resize sisi terpanjang ke MAX_IMAGE_SIDE
    #   • Normalisasi nilai piksel menggunakan statistik ImageNet
    #   • Padding agar ukuran seragam dalam satu batch
    #   • Konversi ke tensor PyTorch
    inputs = processor(images=image, return_tensors="pt")

    # --- Inferensi ---
    # torch.no_grad() menonaktifkan perhitungan gradien → hemat memori & lebih cepat
    with torch.no_grad():
        outputs = model(**inputs)

    # Ambil logits (prediksi kelas mentah) dan pred_boxes (koordinat ternormalisasi)
    logits = outputs.logits       # shape: (1, num_queries, num_classes + 1)
    pred_boxes = outputs.pred_boxes  # shape: (1, num_queries, 4) – format cxcywh [0,1]

    # Hitung confidence score menggunakan Softmax pada dimensi kelas
    # Kolom terakhir (-1) adalah kelas "no-object"; kita ambil skor kelas fraktur (indeks 1)
    probs = torch.nn.functional.softmax(logits, dim=-1)  # (1, num_queries, num_classes+1)

    # Ambil skor kelas fraktur pada batch pertama menggunakan FRACTURE_CLASS_INDEX
    scores = probs[0, :, FRACTURE_CLASS_INDEX]   # (num_queries,)
    boxes = pred_boxes[0]     # (num_queries, 4) – cxcywh ternormalisasi

    # --- Post-Processing ---
    detections = []
    for score, box in zip(scores, boxes):
        conf = score.item()

        # Lewati prediksi di bawah ambang batas confidence
        if conf < CONFIDENCE_THRESHOLD:
            continue

        # Konversi format cxcywh (ternormalisasi) → xyxy (piksel gambar asli)
        cx, cy, w, h = box.tolist()
        x1 = (cx - w / 2) * orig_width
        y1 = (cy - h / 2) * orig_height
        x2 = (cx + w / 2) * orig_width
        y2 = (cy + h / 2) * orig_height

        # Klem koordinat agar tidak keluar dari batas gambar
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(orig_width, int(x2))
        y2 = min(orig_height, int(y2))

        detections.append({"box": (x1, y1, x2, y2), "score": conf})

    return detections


# ===========================================================================
# 3. FUNGSI VISUALISASI
#    Menggambar bounding box dan label pada gambar asli menggunakan PIL.
# ===========================================================================
def draw_detections(image: Image.Image, detections: list) -> Image.Image:
    """
    Menggambar bounding box beserta label confidence pada gambar.

    Args:
        image: Gambar asli dalam format PIL.Image.
        detections: List hasil dari run_inference().

    Returns:
        Gambar PIL dengan anotasi bounding box.
    """
    # Buat salinan agar gambar asli tidak termodifikasi
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    # Tentukan ukuran font proporsional terhadap gambar
    font_size = max(14, int(min(annotated.size) * 0.025))

    # Daftar jalur font lintas-platform (Linux, macOS, Windows)
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",       # Linux (Debian/Ubuntu)
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",                 # Linux (Fedora/RHEL)
        "/Library/Fonts/Arial Bold.ttf",                               # macOS
        "C:/Windows/Fonts/arialbd.ttf",                                # Windows
    ]
    font = None
    for fp in font_candidates:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except IOError:
            continue
    if font is None:
        # Fallback ke font default bawaan Pillow jika tidak ada font yang tersedia
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["score"]

        # Ketebalan garis bounding box
        line_width = max(2, int(min(annotated.size) * 0.003))

        # Gambar bounding box
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=line_width)

        # Buat teks label dengan nilai confidence
        label_text = f"Fraktur {conf * 100:.1f}%"

        # Hitung ukuran kotak teks menggunakan textbbox
        text_bbox = draw.textbbox((x1, y1), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Posisi latar belakang teks (di atas bounding box)
        padding = 3
        bg_y1 = max(0, y1 - text_h - padding * 2)
        bg_y2 = y1

        # Gambar latar belakang teks
        draw.rectangle(
            [x1, bg_y1, x1 + text_w + padding * 2, bg_y2],
            fill=TEXT_BG_COLOR,
        )

        # Gambar teks label
        draw.text((x1 + padding, bg_y1 + padding), label_text, fill=TEXT_COLOR, font=font)

    return annotated


# ===========================================================================
# ANTARMUKA PENGGUNA (UI)
# ===========================================================================

# Judul dan deskripsi singkat
st.markdown("<h1>🦴 Deteksi Fraktur Tulang</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Sistem deteksi fraktur tulang berbasis DETR (Detection Transformer) '
    "pada citra X-Ray</p>",
    unsafe_allow_html=True,
)

st.divider()

# --- Komponen 1: File Uploader ---
st.markdown('<p class="section-label">Input Citra X-Ray</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    label="Input Citra X-Ray",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",   # Label tersembunyi karena sudah ditampilkan di atas
    help="Unggah file citra X-Ray dalam format JPG, JPEG, atau PNG.",
)

# --- Komponen 2: Preview gambar asli ---
if uploaded_file is not None:
    # Baca dan tampilkan gambar asli yang diunggah
    original_image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

    st.markdown('<p class="section-label">Preview Citra</p>', unsafe_allow_html=True)
    st.image(original_image, use_container_width=True)

st.divider()

# --- Komponen 3: Tombol Proses Deteksi ---
process_button = st.button("🔍 Proses Deteksi Fraktur", use_container_width=True)

# ===========================================================================
# LOGIKA BACKEND – dijalankan hanya saat tombol ditekan
# ===========================================================================
if process_button:
    # Validasi: pastikan pengguna sudah mengunggah gambar
    if uploaded_file is None:
        st.warning("⚠️ Harap unggah citra X-Ray terlebih dahulu sebelum memproses.")
    else:
        # Muat model (di-cache sehingga hanya di-load sekali)
        processor, model = load_model()

        # Jalankan preprocessing dan inferensi
        with st.spinner("Menganalisis citra…"):
            detections = run_inference(original_image, processor, model)

        # Gambar bounding box pada gambar asli
        result_image = draw_detections(original_image, detections)

        st.divider()

        # Tampilkan gambar hasil deteksi
        st.markdown('<p class="section-label">Hasil Deteksi</p>', unsafe_allow_html=True)
        st.image(result_image, use_container_width=True)

        # Tampilkan ringkasan deteksi
        if detections:
            st.success(f"✅ Ditemukan **{len(detections)}** area fraktur.")
            for i, det in enumerate(detections, start=1):
                x1, y1, x2, y2 = det["box"]
                st.markdown(
                    f"- **Fraktur #{i}** — Confidence: `{det['score'] * 100:.1f}%` "
                    f"| Koordinat: `({x1}, {y1}) → ({x2}, {y2})`"
                )
        else:
            st.info("ℹ️ Tidak ada fraktur terdeteksi pada citra ini dengan threshold saat ini.")
