import os
import io
import uuid
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory, send_file
import fitz  # PyMuPDF
from simple_lama_inpainting import SimpleLama
from PIL import Image

app = Flask(__name__, static_folder="static")
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

print("Loading LaMa inpainting model...")
lama = SimpleLama()
print("LaMa model ready.")

RENDER_DPI = 200
PDF_DPI = 72
DPI_SCALE = RENDER_DPI / PDF_DPI

# Cache rendered pages: "file_id:page_num" -> numpy_array
page_cache = {}
# Undo / redo history: file_id -> list of (page_num, image_snapshot)
undo_history = {}
redo_history = {}
MAX_UNDO = 50


def _get_page_image(file_id, page_num):
    cache_key = f"{file_id}:{page_num}"
    if cache_key in page_cache:
        return page_cache[cache_key]

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_id}.pdf")
    doc = fitz.open(filepath)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=RENDER_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n).copy()
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    doc.close()

    page_cache[cache_key] = img
    return img


def _save_undo(file_id, page_num, img_before):
    if file_id not in undo_history:
        undo_history[file_id] = []
    undo_history[file_id].append((page_num, img_before.copy()))
    if len(undo_history[file_id]) > MAX_UNDO:
        undo_history[file_id] = undo_history[file_id][-MAX_UNDO:]
    redo_history.pop(file_id, None)


def _update_page_cache(file_id, page_num, img):
    page_cache[f"{file_id}:{page_num}"] = img


# ---------------------------------------------------------------------------
# Background analysis + inpainting
# ---------------------------------------------------------------------------

def _analyze_background(img_region, mask):
    outside = mask == 0
    if not np.any(outside):
        return True, np.array([255, 255, 255], dtype=np.uint8)
    bg_pixels = img_region[outside]
    avg_std = np.std(bg_pixels.astype(np.float32), axis=0).mean()
    median_color = np.median(bg_pixels, axis=0).astype(np.uint8)
    return avg_std < 15, median_color


def _build_glyph_mask(img_region, bbox_mask, bg_color):
    gray = cv2.cvtColor(img_region, cv2.COLOR_RGB2GRAY)
    bg_gray = int(np.mean(cv2.cvtColor(bg_color.reshape(1, 1, 3), cv2.COLOR_RGB2GRAY)))
    diff = np.abs(gray.astype(np.int16) - bg_gray)
    threshold = max(25, int(np.std(gray[bbox_mask > 0]) * 0.8)) if np.any(bbox_mask > 0) else 30
    text_pixels = (diff > threshold) & (bbox_mask > 0)
    glyph_mask = text_pixels.astype(np.uint8) * 255
    glyph_mask = cv2.dilate(glyph_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    if np.sum(glyph_mask > 0) / max(1, np.sum(bbox_mask > 0)) < 0.05:
        return bbox_mask
    return glyph_mask


def _detect_lines(gray_img):
    h, w = gray_img.shape
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 10), 1))
    h_lines = cv2.dilate(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, h_kernel),
                         cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 10)))
    v_lines = cv2.dilate(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, v_kernel),
                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    line_mask = cv2.bitwise_or(h_lines, v_lines)
    hough = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=max(20, min(w, h) // 5), maxLineGap=5)
    if hough is not None:
        for line in hough:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 5 or angle > 175 or 85 < angle < 95:
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)
    return line_mask


# Font flag bitmasks
FLAG_BOLD = 16
FLAG_ITALIC = 2
FLAG_SERIF = 4
FLAG_MONO = 8


def _map_font(font_name, flags):
    """Map PDF font name + flags to closest font code."""
    name = font_name.lower() if font_name else ""
    bold = bool(flags & FLAG_BOLD) or any(k in name for k in ("bold", "black", "heavy"))
    italic = bool(flags & FLAG_ITALIC) or any(k in name for k in ("italic", "oblique"))

    if flags & FLAG_MONO or any(k in name for k in ("courier", "mono", "consol", "menlo")):
        if bold and italic: return "cobi"
        if bold: return "cobo"
        if italic: return "coit"
        return "cour"
    elif flags & FLAG_SERIF or any(k in name for k in ("times", "serif", "georgia", "garamond", "cambria", "palatino")):
        if bold and italic: return "tibi"
        if bold: return "tibo"
        if italic: return "tiit"
        return "tiro"
    else:
        if bold and italic: return "hebi"
        if bold: return "hebo"
        if italic: return "heit"
        return "helv"


_PIL_FONT_MAP = {
    "tiro": "/usr/share/fonts/gsfonts/NimbusRoman-Regular.otf",
    "tibo": "/usr/share/fonts/gsfonts/NimbusRoman-Bold.otf",
    "tiit": "/usr/share/fonts/gsfonts/NimbusRoman-Italic.otf",
    "tibi": "/usr/share/fonts/gsfonts/NimbusRoman-BoldItalic.otf",
    "helv": "/usr/share/fonts/gsfonts/NimbusSans-Regular.otf",
    "hebo": "/usr/share/fonts/gsfonts/NimbusSans-Bold.otf",
    "heit": "/usr/share/fonts/gsfonts/NimbusSans-Italic.otf",
    "hebi": "/usr/share/fonts/gsfonts/NimbusSans-BoldItalic.otf",
    "cour": "/usr/share/fonts/gsfonts/NimbusMonoPS-Regular.otf",
    "cobo": "/usr/share/fonts/gsfonts/NimbusMonoPS-Bold.otf",
    "coit": "/usr/share/fonts/gsfonts/NimbusMonoPS-Italic.otf",
    "cobi": "/usr/share/fonts/gsfonts/NimbusMonoPS-BoldItalic.otf",
}


def _get_spans_in_rect(file_id, page_num, rx0, ry0, rx1, ry1):
    """Extract text spans from the PDF that overlap with the given rect (PDF points)."""
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_id}.pdf")
    doc = fitz.open(filepath)
    page = doc[page_num]
    blocks = page.get_text("dict")["blocks"]
    spans = []
    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if not span["text"].strip():
                    continue
                sb = span["bbox"]  # (x0, y0, x1, y1)
                # Check overlap with selection rect
                if sb[2] > rx0 and sb[0] < rx1 and sb[3] > ry0 and sb[1] < ry1:
                    spans.append({
                        "text": span["text"],
                        "bbox": list(sb),
                        "font": span["font"],
                        "size": span["size"],
                        "color": span["color"],
                        "flags": span["flags"],
                    })
    doc.close()
    return spans


def _render_spans_on_image(img, spans):
    """Render text spans onto a numpy image using Pillow."""
    from PIL import ImageDraw, ImageFont

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    for span in spans:
        text = span["text"]
        bbox = span["bbox"]
        fontname = _map_font(span.get("font", ""), span.get("flags", 0))
        pdf_fontsize = span.get("size", 11)
        color_int = span.get("color", 0)
        color_rgb = (
            (color_int >> 16) & 255,
            (color_int >> 8) & 255,
            color_int & 255,
        )

        px0 = bbox[0] * DPI_SCALE
        py0 = bbox[1] * DPI_SCALE
        px1 = bbox[2] * DPI_SCALE
        target_w = px1 - px0

        font_path = _PIL_FONT_MAP.get(fontname, _PIL_FONT_MAP["tiro"])
        pixel_fontsize = pdf_fontsize * DPI_SCALE

        try:
            pil_font = ImageFont.truetype(font_path, int(round(pixel_fontsize)))
        except Exception:
            pil_font = ImageFont.load_default()

        # Auto-scale to fit bbox width
        text_bbox = pil_font.getbbox(text)
        rendered_w = text_bbox[2] - text_bbox[0]
        if rendered_w > target_w * 1.02 and rendered_w > 0:
            scale = target_w / rendered_w
            pixel_fontsize *= scale
            try:
                pil_font = ImageFont.truetype(font_path, int(round(pixel_fontsize)))
            except Exception:
                pass
            text_bbox = pil_font.getbbox(text)

        text_y = py0 - text_bbox[1]
        draw.text((px0, text_y), text, fill=color_rgb, font=pil_font)

    return np.array(pil_img)


def _erase_region(full_img, px0, py0, px1, py1):
    """Erase a rectangular region and reconstruct the background."""
    h, w = full_img.shape[:2]
    text_h, text_w = py1 - py0, px1 - px0
    ctx_pad = max(text_h * 3, text_w * 2, 80)
    crop_x0 = max(0, px0 - ctx_pad)
    crop_y0 = max(0, py0 - ctx_pad)
    crop_x1 = min(w, px1 + ctx_pad)
    crop_y1 = min(h, py1 + ctx_pad)
    crop = full_img[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    crop_h, crop_w = crop.shape[:2]

    mask_y0, mask_x0 = py0 - crop_y0, px0 - crop_x0
    mask_y1, mask_x1 = py1 - crop_y0, px1 - crop_x0
    bbox_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    bbox_mask[mask_y0:mask_y1, mask_x0:mask_x1] = 255

    is_solid, bg_color = _analyze_background(crop, bbox_mask)

    if is_solid:
        glyph_mask = _build_glyph_mask(crop, bbox_mask, bg_color)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        line_mask = _detect_lines(gray)
        glyph_mask = cv2.bitwise_and(glyph_mask, cv2.bitwise_not(line_mask))
        fill_region = glyph_mask > 0
        if np.any(fill_region):
            result = crop.copy()
            result[fill_region] = bg_color
            blur_mask = cv2.GaussianBlur(glyph_mask.astype(np.float32) / 255.0, (5, 5), 0)
            blend_3ch = blur_mask[:, :, np.newaxis]
            blended = (crop.astype(np.float32) * (1 - blend_3ch) +
                       result.astype(np.float32) * blend_3ch).astype(np.uint8)
        else:
            blended = crop
    else:
        inpaint_mask = bbox_mask.copy()
        result_pil = lama(Image.fromarray(crop), Image.fromarray(inpaint_mask))
        result = np.array(result_pil)
        if result.shape[0] != crop_h or result.shape[1] != crop_w:
            result = cv2.resize(result, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
        blend_mask = cv2.GaussianBlur(
            cv2.dilate(inpaint_mask.astype(np.float32) / 255.0,
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1),
            (9, 9), 0)
        blend_3ch = blend_mask[:, :, np.newaxis]
        blended = (crop.astype(np.float32) * (1 - blend_3ch) +
                   result.astype(np.float32) * blend_3ch).astype(np.uint8)

    return blended, crop_x0, crop_y0, crop_w, crop_h


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    file_id = str(uuid.uuid4())
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_id}.pdf")
    file.save(filepath)

    doc = fitz.open(filepath)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pages.append({"width": page.rect.width, "height": page.rect.height})
    doc.close()
    return jsonify({"file_id": file_id, "pages": pages})


@app.route("/pdf/<file_id>")
def serve_pdf(file_id):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_id}.pdf")
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    return send_file(filepath, mimetype="application/pdf")


@app.route("/erase", methods=["POST"])
def erase():
    """Erase a user-selected rectangular region from a page."""
    data = request.json
    file_id = data["file_id"]
    page_num = data["page"]
    rect = data["rect"]  # {x, y, w, h} in PDF points

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_id}.pdf")
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    full_img = _get_page_image(file_id, page_num)
    h, w = full_img.shape[:2]

    _save_undo(file_id, page_num, full_img)

    # Convert PDF points to pixel coords
    px0 = max(0, int(rect["x"] * DPI_SCALE))
    py0 = max(0, int(rect["y"] * DPI_SCALE))
    px1 = min(w, int((rect["x"] + rect["w"]) * DPI_SCALE))
    py1 = min(h, int((rect["y"] + rect["h"]) * DPI_SCALE))

    if px1 <= px0 or py1 <= py0:
        return jsonify({"error": "Invalid selection"}), 400

    blended, crop_x0, crop_y0, crop_w, crop_h = _erase_region(full_img, px0, py0, px1, py1)

    full_img[crop_y0:crop_y0 + crop_h, crop_x0:crop_x0 + crop_w] = blended
    _update_page_cache(file_id, page_num, full_img)

    # Return full page image
    pil_img = Image.fromarray(full_img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)

    return jsonify({
        "image": base64.b64encode(buf.getvalue()).decode(),
        "page": page_num,
        "undo_remaining": len(undo_history.get(file_id, [])),
    })


@app.route("/undo", methods=["POST"])
def undo():
    data = request.json
    file_id = data["file_id"]

    if file_id not in undo_history or not undo_history[file_id]:
        return jsonify({"error": "Nothing to undo"}), 400

    page_num, snapshot = undo_history[file_id].pop()

    current_img = page_cache.get(f"{file_id}:{page_num}")
    if current_img is not None:
        if file_id not in redo_history:
            redo_history[file_id] = []
        redo_history[file_id].append((page_num, current_img.copy()))

    _update_page_cache(file_id, page_num, snapshot)

    buf = io.BytesIO()
    Image.fromarray(snapshot).save(buf, format="JPEG", quality=92)

    return jsonify({
        "page": page_num,
        "image": base64.b64encode(buf.getvalue()).decode(),
        "undo_remaining": len(undo_history.get(file_id, [])),
        "redo_remaining": len(redo_history.get(file_id, [])),
    })


@app.route("/redo", methods=["POST"])
def redo():
    data = request.json
    file_id = data["file_id"]

    if file_id not in redo_history or not redo_history[file_id]:
        return jsonify({"error": "Nothing to redo"}), 400

    page_num, snapshot = redo_history[file_id].pop()

    current_img = page_cache.get(f"{file_id}:{page_num}")
    if current_img is not None:
        if file_id not in undo_history:
            undo_history[file_id] = []
        undo_history[file_id].append((page_num, current_img.copy()))

    _update_page_cache(file_id, page_num, snapshot)

    buf = io.BytesIO()
    Image.fromarray(snapshot).save(buf, format="JPEG", quality=92)

    return jsonify({
        "page": page_num,
        "image": base64.b64encode(buf.getvalue()).decode(),
        "undo_remaining": len(undo_history.get(file_id, [])),
        "redo_remaining": len(redo_history.get(file_id, [])),
    })


@app.route("/copy", methods=["POST"])
def copy_region():
    """Copy a rectangular region from a page, return as base64 PNG."""
    data = request.json
    file_id = data["file_id"]
    page_num = data["page"]
    rect = data["rect"]  # {x, y, w, h} in PDF points

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_id}.pdf")
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    full_img = _get_page_image(file_id, page_num)
    h, w = full_img.shape[:2]

    px0 = max(0, int(rect["x"] * DPI_SCALE))
    py0 = max(0, int(rect["y"] * DPI_SCALE))
    px1 = min(w, int((rect["x"] + rect["w"]) * DPI_SCALE))
    py1 = min(h, int((rect["y"] + rect["h"]) * DPI_SCALE))

    if px1 <= px0 or py1 <= py0:
        return jsonify({"error": "Invalid selection"}), 400

    snippet = full_img[py0:py1, px0:px1].copy()
    pil_img = Image.fromarray(snippet)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")

    return jsonify({
        "image": base64.b64encode(buf.getvalue()).decode(),
        "w": px1 - px0,
        "h": py1 - py0,
    })


@app.route("/paste", methods=["POST"])
def paste_region():
    """Paste a base64 image onto a page at a given position and size."""
    data = request.json
    file_id = data["file_id"]
    page_num = data["page"]
    image_b64 = data["image"]  # base64 PNG of the snippet
    dest = data["dest"]  # {x, y, w, h} in PDF points

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_id}.pdf")
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    full_img = _get_page_image(file_id, page_num)
    h, w = full_img.shape[:2]

    _save_undo(file_id, page_num, full_img)

    # Decode the snippet
    snippet_bytes = base64.b64decode(image_b64)
    snippet_pil = Image.open(io.BytesIO(snippet_bytes)).convert("RGB")

    # Destination in pixel coords
    dx0 = max(0, int(dest["x"] * DPI_SCALE))
    dy0 = max(0, int(dest["y"] * DPI_SCALE))
    dx1 = min(w, int((dest["x"] + dest["w"]) * DPI_SCALE))
    dy1 = min(h, int((dest["y"] + dest["h"]) * DPI_SCALE))
    dw, dh = dx1 - dx0, dy1 - dy0

    if dw <= 0 or dh <= 0:
        return jsonify({"error": "Invalid destination"}), 400

    # Resize snippet to destination size
    resized = snippet_pil.resize((dw, dh), Image.LANCZOS)
    snippet_np = np.array(resized)

    # Composite onto page
    full_img[dy0:dy1, dx0:dx1] = snippet_np
    _update_page_cache(file_id, page_num, full_img)

    pil_img = Image.fromarray(full_img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)

    return jsonify({
        "image": base64.b64encode(buf.getvalue()).decode(),
        "page": page_num,
        "undo_remaining": len(undo_history.get(file_id, [])),
    })


@app.route("/clean-bg", methods=["POST"])
def clean_background():
    """Erase background in a region but re-render text/numbers on top."""
    data = request.json
    file_id = data["file_id"]
    page_num = data["page"]
    rect = data["rect"]  # {x, y, w, h} in PDF points

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_id}.pdf")
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    full_img = _get_page_image(file_id, page_num)
    h, w = full_img.shape[:2]

    _save_undo(file_id, page_num, full_img)

    # 1. Extract text spans in this region from the original PDF
    rx0, ry0 = rect["x"], rect["y"]
    rx1, ry1 = rx0 + rect["w"], ry0 + rect["h"]
    spans = _get_spans_in_rect(file_id, page_num, rx0, ry0, rx1, ry1)

    # 2. Erase the entire region (text + background)
    px0 = max(0, int(rx0 * DPI_SCALE))
    py0 = max(0, int(ry0 * DPI_SCALE))
    px1 = min(w, int(rx1 * DPI_SCALE))
    py1 = min(h, int(ry1 * DPI_SCALE))

    if px1 <= px0 or py1 <= py0:
        return jsonify({"error": "Invalid selection"}), 400

    blended, crop_x0, crop_y0, crop_w, crop_h = _erase_region(full_img, px0, py0, px1, py1)
    full_img[crop_y0:crop_y0 + crop_h, crop_x0:crop_x0 + crop_w] = blended

    # 3. Re-render text spans on top of the cleaned background
    if spans:
        full_img = _render_spans_on_image(full_img, spans)

    _update_page_cache(file_id, page_num, full_img)

    pil_img = Image.fromarray(full_img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)

    return jsonify({
        "image": base64.b64encode(buf.getvalue()).decode(),
        "page": page_num,
        "undo_remaining": len(undo_history.get(file_id, [])),
    })


@app.route("/download", methods=["POST"])
def download():
    """Build final PDF with erased regions baked in."""
    data = request.json
    file_id = data["file_id"]
    modified_pages = set(data.get("modified_pages", []))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_id}.pdf")
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    doc = fitz.open(filepath)

    for page_num in modified_pages:
        cache_key = f"{file_id}:{page_num}"
        if cache_key not in page_cache:
            continue

        img = page_cache[cache_key]
        page = doc[page_num]
        rect = page.rect

        page.clean_contents()
        for xref in page.get_contents():
            doc.update_stream(xref, b" ")

        pil_img = Image.fromarray(img)
        img_buf = io.BytesIO()
        pil_img.save(img_buf, format="PNG")
        page.insert_image(rect, stream=img_buf.getvalue())

    save_id = str(uuid.uuid4())
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{save_id}.pdf")
    doc.save(save_path, garbage=4, deflate=True)
    doc.close()

    return jsonify({"file_id": save_id})


@app.route("/download/<file_id>")
def download_pdf(file_id):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_id}.pdf")
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    return send_file(filepath, as_attachment=True, download_name="erased.pdf")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
