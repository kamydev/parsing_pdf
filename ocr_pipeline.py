import os
import cv2
import pytesseract
import pandas as pd
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from pytesseract import Output
from paddleocr import PaddleOCR
from fpdf import FPDF

class PDFtoExcelOCR:
    def __init__(self, pdf_path, output_dir, tesseract_cmd=None, poppler_path=None):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.poppler_path = poppler_path

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        os.makedirs(self.output_dir, exist_ok=True)

        self.ocr = PaddleOCR(lang='fr') 

    def convert_pdf_to_images(self):
        images = convert_from_path(self.pdf_path, dpi=300, poppler_path=self.poppler_path)
        image_paths = []

        for idx, img in enumerate(images):
            img_path = os.path.join(self.output_dir, f"page_{idx+1}.jpg")
            img.save(img_path, "JPEG")
            image_paths.append(img_path)

        return image_paths

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def run_pipeline(self):
        """Full pipeline: convert PDF to images, preprocess, extract text and table content as raw text."""
        images = self.convert_pdf_to_images()
        all_content_blocks = []

        for i, image_path in enumerate(images):
            print(f"\n📄 Processing page {i + 1}/{len(images)}")

            # Load image from path
            image = cv2.imread(image_path)

            # Preprocess for table detection (from image path)
            preprocessed = self.preprocess_image(image_path)

            # === Detect tables ===
            table_bboxes = self.detect_tables(preprocessed)
            table_bboxes = sorted(table_bboxes, key=lambda box: box[1])  # Sort top-to-bottom
            print(f"📐 Detected {len(table_bboxes)} table(s)")

            # Mask table areas to improve text OCR
            text_only = self.remove_regions(image, table_bboxes)

            # === Extract text blocks ===
            text_blocks = self.extract_layout_blocks(text_only)
            for tb in text_blocks:
                if tb["text"].strip():
                    all_content_blocks.append({
                        "page": i + 1,
                        "type": "text",
                        "content": tb["text"].strip(),
                        "bbox": tb["bbox"]
                    })

            # === Extract raw table text ===
            for j, bbox in enumerate(table_bboxes):
                table_rows = self.extract_table_from_region(image, bbox)
                if table_rows:
                    all_content_blocks.append({
                        "page": i + 1,
                        "type": "table",
                        "content": table_rows  # leave as list of rows
                    })


        return all_content_blocks

    def detect_tables(self, image):
        """Detect table-like regions using contour analysis and filter bad boxes."""
        gray = image.copy() if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 10)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []

        height, width = gray.shape
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Filter out noise and massive boxes
            if w * h < 5000:
                continue  # too small
            if w > 0.9 * width and h > 0.9 * height:
                continue  # likely whole page or major section
            if w < 50 or h < 20:
                continue  # too thin

            bboxes.append((x, y, x + w, y + h))

        return bboxes

    def extract_table_from_region(self, image, bbox):
        """
        Extract structured table rows from a cropped image region.
        """
        # Crop the region
        x1, y1, x2, y2 = map(int, bbox)
        cropped = image[y1:y2, x1:x2]
        temp_path = "temp_cropped_table.png"
        cv2.imwrite(temp_path, cropped)

        # Run OCR
        result = self.ocr.ocr(temp_path)[0]

        if not isinstance(result, dict):
            return []

        boxes = result.get("rec_boxes", [])
        texts = result.get("rec_texts", [])

        if len(boxes) != len(texts):
            return []

        # Utility functions
        def get_y_center(box):
            box = np.array(box).reshape(-1, 2)
            return float(box[:, 1].mean())

        def get_x_coord(box):
            box = np.array(box).reshape(-1, 2)
            return float(box[:, 0].min())

        # Pair boxes with their texts
        box_texts = [
            (boxes[i], texts[i]) for i in range(len(boxes)) if isinstance(texts[i], str)
        ]

        # Sort by vertical center (top to bottom)
        box_texts.sort(key=lambda x: get_y_center(x[0]))

        # Group by rows (based on y proximity)
        rows = []
        current_row = []
        row_threshold = 20  # pixels

        for box, text in box_texts:
            y = get_y_center(box)
            if not current_row or abs(y - get_y_center(current_row[-1][0])) <= row_threshold:
                current_row.append((box, text))
            else:
                current_row.sort(key=lambda x: get_x_coord(x[0]))
                rows.append([t for _, t in current_row])
                current_row = [(box, text)]

        if current_row:
            current_row.sort(key=lambda x: get_x_coord(x[0]))
            rows.append([t for _, t in current_row])

        return rows

    def extract_layout_blocks(self, image):
        """Extract coherent text blocks (lines or paragraphs) using pytesseract layout data."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        data = pytesseract.image_to_data(pil_image, output_type=Output.DATAFRAME)
        data = data.dropna(subset=["text"])

        grouped = data.groupby(["block_num", "par_num", "line_num"])

        blocks = []
        for (_, _, _), group in grouped:
            line_text = " ".join(group["text"].tolist()).strip()
            if line_text:
                x = group["left"].min()
                y = group["top"].min()
                w = group["left"].max() + group["width"].max() - x
                h = group["top"].max() + group["height"].max() - y
                blocks.append({
                    "type": "text",
                    "text": line_text,
                    "bbox": (x, y, x + w, y + h)
                })

        return blocks

    def remove_regions(self, image, bboxes):
        """Remove specified regions from an image by filling them with white rectangles."""
        masked_image = image.copy()
        for x0, y0, x1, y1 in bboxes:
            cv2.rectangle(masked_image, (x0, y0), (x1, y1), (255, 255, 255), thickness=-1)
        return masked_image

    def export_to_txt(self, content_blocks, output_path):
        def render_table_ascii(table_rows):
            # Handle multiline cell rows first
            merged_rows = []
            for row in table_rows:
                if len(row) == 1 and merged_rows:
                    merged_rows[-1][-1] += " " + row[0]
                else:
                    merged_rows.append(row)

            # Determine column widths
            num_cols = max(len(row) for row in merged_rows)
            col_widths = [0] * num_cols
            for row in merged_rows:
                for i, cell in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(cell))

            # Helper to format a row
            def format_row(row):
                padded = [row[i].ljust(col_widths[i]) if i < len(row) else " " * col_widths[i] for i in range(num_cols)]
                return "| " + " | ".join(padded) + " |"

            # Build table string
            top_border = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
            lines = [top_border]
            for idx, row in enumerate(merged_rows):
                lines.append(format_row(row))
                lines.append(top_border)
            return "\n".join(lines)

        # Main export logic
        with open(output_path, "w", encoding="utf-8") as f:
            current_page = None
            for block in content_blocks:
                if block["page"] != current_page:
                    current_page = block["page"]
                    f.write(f"\nPage {current_page} :\n")

                if block["type"] == "text":
                    f.write(block["content"].strip() + "\n")

                elif block["type"] == "table":
                    table = block["content"]
                    if isinstance(table, list):
                        table_str = render_table_ascii(table)
                        f.write(table_str + "\n")
                    else:
                        f.write(table + "\n")

    def export_structured_to_pdf(self, content_blocks, pdf_path):
        from fpdf import FPDF
        import os

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Use a Unicode-safe system font (regular only)
        font_path = "C:/Windows/Fonts/arial.ttf"  # system font
        pdf.add_font("CustomUnicode", "", font_path, uni=True)
        pdf.set_font("CustomUnicode", "", size=12)

        current_page = None

        for block in content_blocks:
            if block["page"] != current_page:
                current_page = block["page"]
                pdf.cell(0, 10, f"Page {current_page} :", ln=True)

            if block["type"] == "text":
                lines = block["content"].splitlines()
                for line in lines:
                    if line.strip():  # skip empty lines
                        pdf.cell(0, 10, txt=line.strip(), ln=True)

            elif block["type"] == "table":
                table_data = block["content"]
                if not table_data:
                    continue
                col_count = max(len(row) for row in table_data)
                col_width = pdf.w / col_count - 10
                for row in table_data:
                    for cell in row:
                        cell = cell if isinstance(cell, str) else str(cell)
                        pdf.cell(col_width, 10, txt=cell.strip(), border=1)
                    pdf.ln()

        pdf.output(pdf_path)
