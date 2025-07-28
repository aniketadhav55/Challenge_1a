import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import joblib
import os
import json

#Load Model
model = joblib.load(r"Trained_Model_Round1A_Aniket.joblib")
label_encoder = joblib.load(r"Trained_label_encoder_Round1A_Aniket.joblib")
feature_names = list(model.feature_names_in_)

# Font Threshold 
def estimate_font_threshold(doc, percentile=0.90):
    sizes = [span["size"]
             for page in doc
             for block in page.get_text("dict")["blocks"]
             if "lines" in block
             for line in block["lines"]
             for span in line["spans"]]
    return float(np.quantile(sizes, percentile)) if sizes else 12.0

# Font Stats Per Page
def get_page_font_stats(doc):
    stats = {}
    for page_num, page in enumerate(doc):
        sizes = [span["size"]
                 for block in page.get_text("dict")["blocks"]
                 if "lines" in block
                 for line in block["lines"]
                 for span in line["spans"]]
        if not sizes:
            sizes = [10.0]
        stats[page_num + 1] = {
            "median": float(np.median(sizes)),
            "max": float(np.max(sizes)),
            "std": float(np.std(sizes))
        }
    return stats

# Heuristic Filters 
def is_likely_paragraph(text):
    words = text.split()
    if len(words) > 10: return True
    if text.endswith((".", "?")): return True
    if text and text[0].islower(): return True
    if sum(c in text for c in ".;,") >= 2: return True
    avg_word_len = sum(len(w) for w in words) / max(1, len(words))
    return avg_word_len < 3.5

def is_numbering_only(text):
    stripped = text.strip().replace(".", "").replace("-", "").replace("–", "")
    if not stripped: return True
    if stripped.isdigit(): return True
    if len(stripped) < 4 and all(c.isdigit() or c == '.' for c in text): return True
    if len(text.split()) <= 2 and all(w.strip().isdigit() or w.strip().replace(".", "").isdigit() for w in text.split()):
        return True
    return False

def is_bad_h3(text):
    too_long = len(text.split()) > 10
    ends_like_sentence = text.endswith(".") or text.endswith("?")
    lowercase_start = text and text[0].islower()
    few_caps = sum(c.isupper() for c in text) < 2
    too_much_punct = sum(c in text for c in ".,;:!?") >= 3
    return any([too_long, ends_like_sentence, lowercase_start, few_caps, too_much_punct])

# Feature Extraction 
def extract_features(lines_with_meta, font_stats, doc):
    features = []
    metadata = []

    for line in lines_with_meta:
        text, font_size, is_bold, is_centered, is_italic, page, x0, _, _, _, y0 = line
        stats = font_stats[page]
        line_position = y0 / doc[page - 1].rect.height
        font_size_ratio = font_size / max(1e-3, stats["median"])
        font_size_zscore = (font_size - stats["median"]) / (stats["std"] or 1e-5)
        font_percentile = font_size / stats["max"] if stats["max"] else 0
        char_count = len(text)
        word_count = len(text.split())

        row = {
            "font_size": font_size,
            "is_bold": int(is_bold),
            "is_italic": int(is_italic),
            "is_centered": int(is_centered),
            "x0": x0,
            "y0": y0,
            "page": page,
            "font_size_ratio": font_size_ratio,
            "font_size_zscore": font_size_zscore,
            "font_percentile": font_percentile,
            "line_position": line_position,
            "char_count": char_count,
            "word_count": word_count
        }

        feature_row = [row.get(col, 0) for col in feature_names]
        features.append(feature_row)

        metadata.append({
            "text": text,
            "font_size": font_size,
            "page": page
        })

    return features, metadata

# PDF Processor 
def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    FONT_SIZE_THRESHOLD = estimate_font_threshold(doc)
    font_stats = get_page_font_stats(doc)
    lines_with_meta = []
    prev_bottom = 0

    for page_num, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text, max_font_size, is_bold, is_italic = "", 0, False, False
                x0 = line["bbox"][0]
                y0, y1 = line["bbox"][1], line["bbox"][3]
                for span in line["spans"]:
                    line_text += span["text"].strip() + " "
                    max_font_size = max(max_font_size, span["size"])
                    if "bold" in span["font"].lower(): is_bold = True
                    if "italic" in span["font"].lower(): is_italic = True
                text = line_text.strip()
                if not text:
                    continue
                is_centered = abs(x0 - (page.rect.width / 2)) < 50
                gap_above = y0 - prev_bottom if prev_bottom else 0
                lines_with_meta.append([
                    text, max_font_size, is_bold, is_centered, is_italic,
                    page_num + 1, x0, gap_above, 0, x0, y0
                ])
                prev_bottom = y1

    features, meta = extract_features(lines_with_meta, font_stats, doc)

    if not features or len(features[0]) != len(feature_names):
        raise ValueError(" Feature mismatch. Check feature extraction.")

    df = pd.DataFrame(features, columns=feature_names)
    preds = model.predict(df)
    labels = label_encoder.inverse_transform(preds)

    outline = []
    current_heading = None

    for i, label in enumerate(labels):
        text = meta[i]["text"]
        font_size = meta[i]["font_size"]
        page = meta[i]["page"]

        # Extra strict filter for H3
        if label == "H3" and is_bad_h3(text):
            continue

        if label in ["H1", "H2", "H3"]:
            if (
                font_size < FONT_SIZE_THRESHOLD
                or is_likely_paragraph(text)
                or is_numbering_only(text)
            ):
                continue

            if current_heading and current_heading["page"] == page and abs(current_heading["font_size"] - font_size) <= 0.5:
                current_heading["text"] += " " + text
            else:
                if current_heading:
                    outline.append({
                        "level": current_heading["label"],
                        "text": current_heading["text"].strip(),
                        "page": current_heading["page"]
                    })
                current_heading = {
                    "label": label,
                    "text": text,
                    "page": page,
                    "font_size": font_size
                }
        else:
            if current_heading:
                outline.append({
                    "level": current_heading["label"],
                    "text": current_heading["text"].strip(),
                    "page": current_heading["page"]
                })
                current_heading = None

    if current_heading:
        outline.append({
            "level": current_heading["label"],
            "text": current_heading["text"].strip(),
            "page": current_heading["page"]
        })

    return {
        "title": os.path.basename(pdf_path),
        "outline": outline
    }


if __name__ == "__main__":
    input_dir = r"input"  # folder containing multiple PDFs
    output_dir = r"output"  # folder to store the JSON outputs
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(" No PDF files found in input directory.")
    else:
        for pdf_file in pdf_files:
            input_pdf_path = os.path.join(input_dir, pdf_file)
            try:
                print(f"🔄 Processing: {pdf_file}")
                result = process_pdf(input_pdf_path)
                output_json = os.path.join(output_dir, os.path.splitext(pdf_file)[0] + "_outline.json")
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"✅ Saved: {output_json}")
            except Exception as e:
                print(f" Failed to process {pdf_file}: {str(e)}")
