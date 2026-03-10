"""
Sistem Deteksi Fraktur Tulang pada Citra X-Ray
Menggunakan model Deep Learning berbasis PyTorch dengan arsitektur
Detection Transformer (DETR) dari HuggingFace.

Fitur:
- Deteksi fraktur otomatis pada citra X-Ray
- Pengaturan confidence threshold dan NMS via sidebar
- Visualisasi bounding box dengan perbandingan Original vs Annotated
- Unduh hasil anotasi sebagai file gambar
- Riwayat deteksi per sesi
- Informasi metadata gambar
"""

import io
import logging
from math import gcd
from typing import List, Dict, Tuple, Optional

import torch
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms as torchvision_nms
from transformers import DetrForObjectDetection, DetrImageProcessor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfigurasi halaman Streamlit (wajib dipanggil sebelum elemen UI lainnya)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Deteksi Fraktur Tulang",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Injeksi CSS untuk tampilan modern dan responsif
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* ---- Latar utama ---- */
        .stApp {
            background-color: #0f1117;
            color: #e0e0e0;
        }

        /* ---- Judul utama ---- */
        h1 {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: 0.5px;
        }

        /* ---- Sub-judul / deskripsi ---- */
        .subtitle {
            text-align: center;
            color: #9ca3af;
            font-size: 0.95rem;
            margin-bottom: 2rem;
            line-height: 1.6;
        }

        /* ---- Kartu pembungkus konten ---- */
        .card {
            background-color: #1c1f26;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            border: 1px solid #2d3141;
            transition: box-shadow 0.3s ease;
        }
        .card:hover {
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.12);
        }

        /* ---- Label kecil di atas gambar ---- */
        .section-label {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #6b7280;
            margin-bottom: 0.5rem;
        }

        /* ---- Tombol utama ---- */
        div.stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            font-size: 1.05rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.25s ease;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            box-shadow: 0 4px 16px rgba(59, 130, 246, 0.45);
            transform: translateY(-1px);
        }

        /* ---- Gambar & border radius ---- */
        [data-testid="stImage"] > div {
            border-radius: 10px;
            overflow: hidden;
        }

        /* ---- Divider tipis ---- */
        hr {
            border-color: #2d3141;
        }

        /* ---- Metrik kartu ---- */
        .metric-card {
            background: linear-gradient(135deg, #1e2330 0%, #1c1f26 100%);
            border: 1px solid #2d3141;
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
            transition: box-shadow 0.3s ease;
        }
        .metric-card:hover {
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.12);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #3b82f6;
            line-height: 1;
        }
        .metric-label {
            font-size: 0.75rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 0.4rem;
        }

        /* ---- Tabel deteksi ---- */
        .det-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 0.5rem;
            font-size: 0.9rem;
        }
        .det-table th {
            background-color: #1e2330;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.6rem 0.8rem;
            text-align: left;
            border-bottom: 2px solid #2d3141;
        }
        .det-table td {
            padding: 0.55rem 0.8rem;
            border-bottom: 1px solid #2d3141;
            color: #d1d5db;
        }
        .det-table tr:hover td {
            background-color: #1e2330;
        }

        /* ---- Confidence bar ---- */
        .conf-bar-bg {
            background-color: #1e2330;
            border-radius: 6px;
            height: 8px;
            width: 100%;
            overflow: hidden;
        }
        .conf-bar-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 0.4s ease;
        }

        /* ---- Sidebar styling ---- */
        section[data-testid="stSidebar"] {
            background-color: #141720;
            border-right: 1px solid #2d3141;
        }

        /* ---- Sidebar header ---- */
        .sidebar-header {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #6b7280;
            margin-bottom: 0.5rem;
            margin-top: 1rem;
        }

        /* ---- Footer ---- */
        .footer {
            text-align: center;
            color: #4b5563;
            font-size: 0.78rem;
            margin-top: 3rem;
            padding: 1.5rem 0;
            border-top: 1px solid #2d3141;
        }

        /* ---- Image info badge ---- */
        .img-info {
            display: inline-block;
            background-color: #1e2330;
            border: 1px solid #2d3141;
            border-radius: 8px;
            padding: 0.3rem 0.7rem;
            font-size: 0.78rem;
            color: #9ca3af;
            margin-right: 0.5rem;
            margin-bottom: 0.3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Konstanta default
# ---------------------------------------------------------------------------
MODEL_CHECKPOINT = "model.ckpt"   # Checkpoint lokal model DETR
DEFAULT_CONFIDENCE = 0.5           # Ambang batas confidence default
DEFAULT_NMS_IOU = 0.5              # Ambang batas IoU untuk NMS default
MAX_IMAGE_SIDE = 800               # Panjang sisi terpanjang untuk resize (px)
BOX_COLOR = (59, 130, 246)         # Warna bounding box (biru, RGB)
TEXT_BG_COLOR = (59, 130, 246)     # Warna latar label teks
TEXT_COLOR = (255, 255, 255)       # Warna teks label

# Indeks kelas fraktur sesuai label_id model DETR yang dilatih.
# Model ini menggunakan dua kelas: 0 = background/"no-object", 1 = fraktur.
FRACTURE_CLASS_INDEX = 1


# ---------------------------------------------------------------------------
# Inisialisasi session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history: List[Dict] = []


# ===========================================================================
# 1. FUNGSI MUAT MODEL
#    Menggunakan @st.cache_resource agar model hanya di-load sekali.
# ===========================================================================
@st.cache_resource(show_spinner="Memuat model DETR…")
def load_model() -> Tuple[DetrImageProcessor, DetrForObjectDetection]:
    """
    Memuat DetrImageProcessor dan DetrForObjectDetection dari checkpoint lokal.
    Model di-set ke mode evaluasi untuk inferensi.

    Returns:
        Tuple berisi (processor, model).

    Raises:
        FileNotFoundError: Jika file checkpoint tidak ditemukan.
        RuntimeError: Jika terjadi kegagalan saat memuat model.
    """
    try:
        logger.info("Memuat model DETR dari checkpoint: %s", MODEL_CHECKPOINT)

        # Muat processor untuk preprocessing gambar (resize, normalisasi, padding)
        processor = DetrImageProcessor.from_pretrained(MODEL_CHECKPOINT)

        # Muat bobot model DETR dari checkpoint lokal
        model = DetrForObjectDetection.from_pretrained(MODEL_CHECKPOINT)

        # Set ke mode evaluasi: menonaktifkan dropout dan batch norm tracking
        model.eval()

        logger.info("Model DETR berhasil dimuat.")
        return processor, model

    except Exception as exc:
        logger.error("Gagal memuat model: %s", exc)
        raise


# ===========================================================================
# 2. FUNGSI UTILITAS — Non-Maximum Suppression (NMS)
# ===========================================================================
def apply_nms(
    detections: List[Dict], iou_threshold: float = 0.5
) -> List[Dict]:
    """
    Menerapkan Non-Maximum Suppression untuk menyaring deteksi yang
    saling tumpang-tindih, sehingga hanya deteksi terbaik yang tersisa.

    Args:
        detections: List of dict dengan kunci 'box' (x1,y1,x2,y2) dan 'score'.
        iou_threshold: Batas IoU di atas mana deteksi dianggap duplikat.

    Returns:
        List deteksi setelah NMS.
    """
    if len(detections) == 0:
        return detections

    boxes = torch.tensor([d["box"] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d["score"] for d in detections], dtype=torch.float32)

    keep_indices = torchvision_nms(boxes, scores, iou_threshold)
    return [detections[i] for i in keep_indices.tolist()]


# ===========================================================================
# 3. FUNGSI PREPROCESSING & INFERENSI
#    Menangani konversi gambar → tensor, inferensi model, dan post-processing.
# ===========================================================================
def run_inference(
    image: Image.Image,
    processor: DetrImageProcessor,
    model: DetrForObjectDetection,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
    nms_iou_threshold: float = DEFAULT_NMS_IOU,
) -> List[Dict]:
    """
    Melakukan inferensi DETR pada gambar PIL yang diberikan.

    Args:
        image: Gambar asli dalam format PIL.Image (RGB).
        processor: DetrImageProcessor yang sudah di-load.
        model: DetrForObjectDetection yang sudah di-load.
        confidence_threshold: Ambang batas minimum confidence score.
        nms_iou_threshold: Ambang batas IoU untuk NMS.

    Returns:
        List of dict dengan kunci 'box' (xyxy piksel) dan 'score' (float),
        sudah difilter oleh threshold dan NMS, terurut berdasarkan score.
    """
    orig_width, orig_height = image.size
    logger.info(
        "Menjalankan inferensi — ukuran gambar: %dx%d, threshold: %.2f, NMS IoU: %.2f",
        orig_width,
        orig_height,
        confidence_threshold,
        nms_iou_threshold,
    )

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
    logits = outputs.logits          # shape: (1, num_queries, num_classes + 1)
    pred_boxes = outputs.pred_boxes  # shape: (1, num_queries, 4) – format cxcywh [0,1]

    # Hitung confidence score menggunakan Softmax pada dimensi kelas
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Ambil skor kelas fraktur pada batch pertama
    scores = probs[0, :, FRACTURE_CLASS_INDEX]
    boxes = pred_boxes[0]

    # --- Post-Processing ---
    detections: List[Dict] = []
    for score, box in zip(scores, boxes):
        conf = score.item()

        # Lewati prediksi di bawah ambang batas confidence
        if conf < confidence_threshold:
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

    # Terapkan NMS untuk menghilangkan deteksi duplikat
    detections = apply_nms(detections, iou_threshold=nms_iou_threshold)

    # Urutkan berdasarkan score tertinggi
    detections.sort(key=lambda d: d["score"], reverse=True)

    logger.info("Inferensi selesai — %d fraktur terdeteksi.", len(detections))
    return detections


# ===========================================================================
# 4. FUNGSI VISUALISASI
#    Menggambar bounding box dan label pada gambar asli menggunakan PIL.
# ===========================================================================
def _load_font(font_size: int) -> ImageFont.FreeTypeFont:
    """Memuat font TrueType dari berbagai lokasi lintas-platform."""
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for fp in font_candidates:
        try:
            return ImageFont.truetype(fp, font_size)
        except IOError:
            continue
    return ImageFont.load_default()


def draw_detections(image: Image.Image, detections: List[Dict]) -> Image.Image:
    """
    Menggambar bounding box beserta label confidence pada gambar.

    Args:
        image: Gambar asli dalam format PIL.Image.
        detections: List hasil dari run_inference().

    Returns:
        Gambar PIL dengan anotasi bounding box.
    """
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    # Tentukan ukuran font proporsional terhadap gambar
    font_size = max(14, int(min(annotated.size) * 0.025))
    font = _load_font(font_size)

    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["box"]
        conf = det["score"]

        # Ketebalan garis bounding box
        line_width = max(2, int(min(annotated.size) * 0.003))

        # Gambar bounding box
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=line_width)

        # Buat teks label: nomor urut + confidence
        label_text = f"#{idx} Fraktur {conf * 100:.1f}%"

        # Hitung ukuran kotak teks menggunakan textbbox
        text_bbox = draw.textbbox((x1, y1), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Posisi latar belakang teks (di atas bounding box)
        padding = 4
        bg_y1 = max(0, y1 - text_h - padding * 2)
        bg_y2 = y1

        # Gambar latar belakang teks
        draw.rectangle(
            [x1, bg_y1, x1 + text_w + padding * 2, bg_y2],
            fill=TEXT_BG_COLOR,
        )

        # Gambar teks label
        draw.text(
            (x1 + padding, bg_y1 + padding),
            label_text,
            fill=TEXT_COLOR,
            font=font,
        )

    return annotated


# ===========================================================================
# 5. FUNGSI UTILITAS — Konversi gambar ke bytes untuk unduhan
# ===========================================================================
def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Mengonversi PIL.Image ke bytes buffer untuk unduhan."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def get_file_size_str(size_bytes: int) -> str:
    """Mengonversi ukuran bytes ke string yang mudah dibaca."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def get_confidence_color(score: float) -> str:
    """Mengembalikan warna hex berdasarkan tingkat confidence."""
    if score >= 0.8:
        return "#22c55e"  # hijau
    if score >= 0.6:
        return "#f59e0b"  # kuning/orange
    return "#ef4444"      # merah


# ===========================================================================
# SIDEBAR — Pengaturan & Informasi
# ===========================================================================
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")

    st.markdown('<p class="sidebar-header">Parameter Deteksi</p>', unsafe_allow_html=True)

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.95,
        value=DEFAULT_CONFIDENCE,
        step=0.05,
        help="Ambang batas minimum confidence score. Deteksi di bawah nilai ini akan diabaikan.",
    )

    nms_iou_threshold = st.slider(
        "NMS IoU Threshold",
        min_value=0.1,
        max_value=0.95,
        value=DEFAULT_NMS_IOU,
        step=0.05,
        help=(
            "Non-Maximum Suppression: menghilangkan deteksi duplikat yang saling tumpang-tindih. "
            "Nilai lebih rendah = penyaringan lebih ketat."
        ),
    )

    st.divider()

    st.markdown('<p class="sidebar-header">Tentang Aplikasi</p>', unsafe_allow_html=True)
    st.markdown(
        """
        **Deteksi Fraktur Tulang** menggunakan arsitektur
        *Detection Transformer* (DETR) untuk mendeteksi area fraktur
        pada citra X-Ray secara otomatis.

        **Teknologi:**
        - 🧠 PyTorch + HuggingFace Transformers
        - 🖼️ Pillow untuk visualisasi
        - 🎨 Streamlit untuk antarmuka web

        **Cara Penggunaan:**
        1. Unggah citra X-Ray (JPG/PNG)
        2. Sesuaikan parameter jika diperlukan
        3. Klik **Proses Deteksi Fraktur**
        4. Lihat hasil dan unduh gambar anotasi
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Riwayat deteksi
    st.markdown('<p class="sidebar-header">Riwayat Sesi</p>', unsafe_allow_html=True)
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history), start=1):
            count = entry["count"]
            avg = entry["avg_conf"]
            icon = "🔴" if count > 0 else "🟢"
            st.markdown(
                f"{icon} **#{len(st.session_state.history) - i + 1}** — "
                f"{count} fraktur (avg {avg:.0%})"
            )
    else:
        st.caption("Belum ada deteksi yang dijalankan.")


# ===========================================================================
# ANTARMUKA PENGGUNA — Konten Utama
# ===========================================================================

# Judul dan deskripsi singkat
st.markdown("<h1>🦴 Deteksi Fraktur Tulang</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">'
    "Sistem deteksi fraktur tulang berbasis <b>DETR</b> "
    "(<i>Detection Transformer</i>) pada citra X-Ray.<br>"
    "Unggah citra, sesuaikan parameter di sidebar, lalu klik tombol deteksi."
    "</p>",
    unsafe_allow_html=True,
)

st.divider()

# --- Komponen 1: File Uploader ---
st.markdown('<p class="section-label">📂 Input Citra X-Ray</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    label="Input Citra X-Ray",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    help="Unggah file citra X-Ray dalam format JPG, JPEG, atau PNG.",
)

# --- Komponen 2: Preview gambar asli + info metadata ---
original_image: Optional[Image.Image] = None

if uploaded_file is not None:
    original_image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    img_w, img_h = original_image.size
    file_size = uploaded_file.size

    st.markdown('<p class="section-label">🖼️ Preview Citra & Informasi</p>', unsafe_allow_html=True)

    col_img, col_info = st.columns([3, 1])

    with col_img:
        st.image(original_image, use_container_width=True)

    with col_info:
        st.markdown(
            f"""
            <div class="card">
                <p class="sidebar-header" style="margin-top:0">Metadata</p>
                <span class="img-info">📐 {img_w} × {img_h} px</span><br>
                <span class="img-info">📁 {get_file_size_str(file_size)}</span><br>
                <span class="img-info">🎨 {original_image.mode}</span><br>
                <span class="img-info">📄 {uploaded_file.type or 'N/A'}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Info rasio aspek
        g = gcd(img_w, img_h)
        st.markdown(
            f"""
            <div class="card">
                <p class="sidebar-header" style="margin-top:0">Detail</p>
                <span class="img-info">📏 Rasio {img_w // g}:{img_h // g}</span><br>
                <span class="img-info">🔢 {img_w * img_h:,} piksel</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# --- Komponen 3: Tombol Proses Deteksi ---
process_button = st.button("🔍 Proses Deteksi Fraktur", use_container_width=True)

# ===========================================================================
# LOGIKA BACKEND — dijalankan hanya saat tombol ditekan
# ===========================================================================
if process_button:
    # Validasi: pastikan pengguna sudah mengunggah gambar
    if original_image is None:
        st.warning("⚠️ Harap unggah citra X-Ray terlebih dahulu sebelum memproses.")
    else:
        try:
            # Muat model (di-cache sehingga hanya di-load sekali)
            processor, model = load_model()
        except Exception as exc:
            st.error(f"❌ Gagal memuat model: {exc}")
            st.stop()

        try:
            # Jalankan preprocessing dan inferensi
            with st.spinner("🔄 Menganalisis citra…"):
                detections = run_inference(
                    original_image,
                    processor,
                    model,
                    confidence_threshold=confidence_threshold,
                    nms_iou_threshold=nms_iou_threshold,
                )
        except Exception as exc:
            st.error(f"❌ Terjadi kesalahan saat inferensi: {exc}")
            logger.exception("Inference error")
            st.stop()

        # Gambar bounding box pada gambar asli
        result_image = draw_detections(original_image, detections)

        # Simpan ke riwayat sesi
        avg_conf = (
            sum(d["score"] for d in detections) / len(detections)
            if detections
            else 0.0
        )
        st.session_state.history.append(
            {"count": len(detections), "avg_conf": avg_conf}
        )

        st.divider()

        # ----- Ringkasan Metrik -----
        st.markdown('<p class="section-label">📊 Ringkasan Deteksi</p>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)

        max_conf = max((d["score"] for d in detections), default=0.0)
        min_conf = min((d["score"] for d in detections), default=0.0)

        with m1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{len(detections)}</div>'
                f'<div class="metric-label">Fraktur Terdeteksi</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{avg_conf:.0%}</div>'
                f'<div class="metric-label">Rata-rata Confidence</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{max_conf:.0%}</div>'
                f'<div class="metric-label">Confidence Tertinggi</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{min_conf:.0%}</div>'
                f'<div class="metric-label">Confidence Terendah</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")  # spacing

        # ----- Tabs: Perbandingan & Detail -----
        tab_compare, tab_detail = st.tabs(["🖼️ Perbandingan Citra", "📋 Detail Deteksi"])

        with tab_compare:
            col_orig, col_result = st.columns(2)
            with col_orig:
                st.markdown(
                    '<p class="section-label">Citra Asli</p>', unsafe_allow_html=True
                )
                st.image(original_image, use_container_width=True)
            with col_result:
                st.markdown(
                    '<p class="section-label">Hasil Deteksi</p>', unsafe_allow_html=True
                )
                st.image(result_image, use_container_width=True)

        with tab_detail:
            if detections:
                st.success(f"✅ Ditemukan **{len(detections)}** area fraktur.")

                # Tabel deteksi
                table_rows = ""
                for i, det in enumerate(detections, start=1):
                    x1, y1, x2, y2 = det["box"]
                    conf = det["score"]
                    color = get_confidence_color(conf)
                    width = int(conf * 100)
                    area = (x2 - x1) * (y2 - y1)

                    table_rows += f"""
                    <tr>
                        <td><b>#{i}</b></td>
                        <td>
                            <div style="display:flex;align-items:center;gap:8px;">
                                <span style="color:{color};font-weight:600;">{conf*100:.1f}%</span>
                                <div class="conf-bar-bg" style="width:80px;">
                                    <div class="conf-bar-fill" style="width:{width}%;background-color:{color};"></div>
                                </div>
                            </div>
                        </td>
                        <td>({x1}, {y1}) → ({x2}, {y2})</td>
                        <td>{area:,} px²</td>
                    </tr>"""

                st.markdown(
                    f"""
                    <table class="det-table">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Confidence</th>
                                <th>Koordinat (x1,y1)→(x2,y2)</th>
                                <th>Luas Area</th>
                            </tr>
                        </thead>
                        <tbody>{table_rows}</tbody>
                    </table>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info(
                    "ℹ️ Tidak ada fraktur terdeteksi pada citra ini. "
                    "Coba turunkan *Confidence Threshold* di sidebar."
                )

        # ----- Tombol unduh gambar hasil -----
        st.divider()
        dl_col1, dl_col2, _ = st.columns([1, 1, 2])
        with dl_col1:
            st.download_button(
                label="⬇️ Unduh Hasil (PNG)",
                data=image_to_bytes(result_image, "PNG"),
                file_name="hasil_deteksi_fraktur.png",
                mime="image/png",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                label="⬇️ Unduh Hasil (JPEG)",
                data=image_to_bytes(result_image, "JPEG"),
                file_name="hasil_deteksi_fraktur.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )

# ===========================================================================
# FOOTER
# ===========================================================================
st.markdown(
    '<div class="footer">'
    "🦴 Bone Fracture Detection System — "
    "Dibangun dengan Streamlit, PyTorch, dan HuggingFace Transformers"
    "</div>",
    unsafe_allow_html=True,
)
