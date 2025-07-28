🚀 Challenge 1a: PDF Heading Extraction Solution
🧠 Overview
This repository contains our submission for Challenge 1a of the Adobe India Hackathon 2025. The task is to develop a high-accuracy, ML-based PDF processing system to extract a structured outline (headings and subheadings) from diverse PDF documents.

The system uses a trained ML classifier with rule-based heuristics to robustly extract and classify headings (H1, H2, H3) while filtering out noise, body paragraphs, and irrelevant text.

📦 Folder Structure
graphql
Copy
Edit
project/
├── process_pdf.py           # Core implementation
├── heading_classifier.joblib        # Trained model (≤200MB)
├── heading_label_encoder.joblib     # Label encoder for headings
├── Dockerfile               # For containerized execution
├── input/                   # Input PDF folder (read-only)
└── output/                  # Output JSON folder
🧰 Technologies Used
Python 3.10

PyMuPDF (fitz) — for PDF parsing

joblib — for model serialization

scikit-learn — for ML inference

pandas, numpy — for feature processing

re — for regex-based text cleaning

🧠 Core Approach
🎯 Objective
Extract structured section headings (H1, H2, H3) from PDFs and export an outline in the format:

json
Copy
Edit
{
  "title": "document_name.pdf",
  "outline": [
    {
      "level": "H1",
      "text": "Introduction",
      "page": 1
    },
    ...
  ]
}
⚙ Processing Pipeline
PDF Parsing
Text is extracted from each line in each PDF page using PyMuPDF. Each line is converted into a feature vector including:

Font size

Bold, Italic, Centered

Character/Word count

Position on page (x0/y0, relative height)

Font percentiles (z-score, ratio)

Dynamic Thresholding
A 90th percentile font size is computed dynamically for each PDF to distinguish actual headings from body text.

ML Classification
A pre-trained classifier (RandomForest or similar) predicts H1, H2, H3, or None.

Heuristic Filtering
We apply smart filters to eliminate misclassified or noisy headings:

Reject likely paragraphs (long sentences, lowercase start)

Reject numbering-only lines (e.g., "1.2.3")

Reject weak H3s (short words, low caps, too much punctuation)

Reject known noise words (e.g., "version", "remarks")

Outline Structuring
Consecutive headings of the same level and similar font size are merged. The result is structured and saved as JSON.

🧪 Heuristic Rules for H3 Classification
Custom filters applied to improve H3 classification accuracy:

Word count limit: max 12 words

Avoid sentences ending with . or ?

Capital letter ratio must be significant

Reject headings with too much punctuation (.,;:!?)

Ignore headings containing noise keywords:

arduino
Copy
Edit
{"overview", "version", "remarks", "table", "contents", "date"}
📤 Output Format
For each input.pdf, the script creates:

pgsql
Copy
Edit
input_outline.json
Example:

json
Copy
Edit
{
  "title": "sample.pdf",
  "outline": [
    { "level": "H1", "text": "Executive Summary", "page": 1 },
    { "level": "H2", "text": "Scope", "page": 2 },
    { "level": "H3", "text": "Limitations", "page": 2 }
  ]
}
🐳 Docker Instructions
✅ Build the Docker Image
bash
Copy
Edit
docker build -t round1a-parser .   
🚀 Run the Container
bash
Copy
Edit
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none round1a-parser
⏱ Performance & Constraints
Constraint	Status
≤ 10 sec for 50 pages	✅ Optimized
≤ 200 MB model size	✅ ~8 MB
No internet	✅ Offline
CPU-only (AMD64)	✅ Supported
RAM ≤ 16 GB	✅ Efficient

🧪 Testing Strategy
Simple PDFs: One-column documents

Complex PDFs: Multi-column, bold/italic/centered text

Large PDFs: 50+ pages

False Positive Checks: Against paragraphs wrongly predicted as headings

📝 How to Extend
Add multilingual support by training on more fonts/scripts

Add footnote/table detection (optional)

Improve section merging logic (e.g., based on layout or TOC matching)

✅ Final Notes
This solution balances ML classification with deterministic heuristics to maximize heading extraction accuracy while adhering to tight compute and runtime constraints.

For questions or feedback, feel free to connect!