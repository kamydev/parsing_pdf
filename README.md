# 🧾 PDF to Text/PDF OCR Pipeline

This project provides a full pipeline for extracting structured information (text and tables) from scanned PDF documents using PaddleOCR and layout-aware models. It’s built for local, offline usage — suitable for confidential or sensitive documents.

## 📌 Features

- Convert scanned PDFs into structured content
- Detect tables and extract them row-wise
- Export output as:
  - Clean `.txt` file with preserved layout
  - Styled `.pdf` with table borders
- Modular pipeline written in Python
- Easy to test, extend, or integrate

## 🧱 Project Structure

```
ocr_project/
│
├── ocr_pipeline.py         # Core pipeline class with all OCR logic
├── main_ocr_pipeline.ipynb     # Jupyter Notebook to test and run the pipeline step-by-step
├── output/                 # All outputs go here (.txt, .pdf, cropped images, etc.)
└── README.md               # You're reading it
```

## 🖥️ Requirements

- Python 3.8+
- PaddleOCR
- layoutparser
- OpenCV
- pdf2image
- fpdf (for PDF export)

### 🔧 Install Dependencies

```bash
pip install paddleocr layoutparser[layoutmodels,tesseract] opencv-python pdf2image fpdf
```

Make sure `Tesseract` and `Poppler` are installed and accessible in PATH (especially for Windows users).

## 🚀 How to Use

### 1. Place your scanned PDF in the `input_pdfs/` folder.

### 2. Run the Jupyter Notebook

Open `test_pipeline.ipynb` and execute it step-by-step. Key sections include:

- Mount paths
- Run the pipeline:
```python
ocr = PDFtoTxtOCR(input_pdf_path="input_pdfs/yourfile.pdf")
content_blocks = ocr.run_pipeline()
```

- Export results:
```python
ocr.export_to_txt(content_blocks, "output/extracted_content.txt")
ocr.export_structured_to_pdf(content_blocks, "output/extracted_content.pdf")
```

## 📄 Output Example

**TXT Output**
```
Page 1 :

A - Description du matériel pédagogique
Salle informatique
Ordinateur (PC)     180     Nombre total minimum
...
```

**PDF Output**
```
Tables are formatted with proper grid borders and wrapped text.
All Unicode characters (French, Arabic, symbols) are supported using system fonts.
```

## 🧠 Internals

The main class `PDFtoTxtOCR` performs the following:

- Uses `pdf2image` to convert PDF pages to images
- Uses `PaddleDetectionLayoutModel` to detect tables
- Crops each table and performs OCR using `PaddleOCR`
- Reconstructs tables by grouping bounding boxes into rows
- Stores everything in a structured list of blocks (text or table)
- Exports results to text and PDF

## ✅ Notes

- Tested on Windows with system fonts (`Arial.ttf`) to support Unicode.
- Works fully offline, no cloud processing needed.
- You can plug in your own model for layout detection if needed.

## 🛠️ TODO (Optional Enhancements)

- Export to `.xlsx` instead of just `.txt` or `.pdf`
- Support multi-column layouts
- Add CLI runner (optional)

## 👤 Author

Abdelkarim – Data Science & AI Engineer




