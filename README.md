# 📘 Adobe Round 1A - Heading Structure Extractor

This project extracts structured heading outlines (`H1`, `H2`, `H3`) from English PDF documents using a trained ML model.

---

## 📁 Project Structure

```
├── input/                       # Folder containing input PDFs
├── output/                      # Output folder with extracted outline JSONs
├── dataset/                     # Training data (CSV or annotated PDFs)
├── Trained_Model.joblib         # Trained classifier model
├── Trained_label_encoder.joblib# Label encoder used during training
├── script.py                    # Main script for PDF processing
├── Dockerfile                   # Docker image configuration
├── requirements.txt             # Required Python dependencies
└── README.md                    # Project instructions
```

---

## 🚀 How to Run with Docker

### 🔨 1. Build Docker Image

From the root project directory:

```bash
docker build -t round1a-parser .
```

---

### 📂 2. Place PDFs to Test

Put your English PDFs into the `input/` folder.

---

### ▶️ 3. Run the Container

```bash
docker run --rm   -v "$(pwd)/input:/app/input"   -v "$(pwd)/output:/app/output"   --network none   round1a-parser
```

This will:
- Process all PDFs in `/input`
- Save structured outlines as JSON in `/output`

---

## 📝 Output Example

Each JSON output looks like:

```json
{
  "title": "sample.pdf",
  "outline": [
    {
      "level": "H1",
      "text": "Introduction",
      "page": 1
    },
    {
      "level": "H2",
      "text": "Installation Steps",
      "page": 2
    }
  ]
}
```

---

## 📦 Dependencies

All dependencies are handled inside Docker via `requirements.txt`. No need to install manually.

---

## 🧠 Notes
- H3 headings are filtered with extra rules to reduce false positives.
- The model is specifically trained on English structure data for accurate classification.