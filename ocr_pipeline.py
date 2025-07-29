# ocr_pipeline.py

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


class PDFtoExcelOCR:
    def __init__(self, pdf_path, output_dir, tesseract_cmd=None, poppler_path=None):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.poppler_path = poppler_path

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        os.makedirs(self.output_dir, exist_ok=True)

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

    def extract_text(self, image):
        text = pytesseract.image_to_string(image, lang='eng')
        return text

    def export_to_excel(self, text_list, excel_path):
        df = pd.DataFrame({'Page': list(range(1, len(text_list)+1)), 'Content': text_list})
        df.to_excel(excel_path, index=False)

    def run_pipeline(self):
        """Full pipeline: convert PDF to images, preprocess, extract text and tables."""
        images = self.convert_pdf_to_images()
        all_content_blocks = []

        for i, image_path in enumerate(images):
            image = cv2.imread(image_path)
            print(f"\n📄 Processing page {i + 1}/{len(images)}")

            # Preprocessing
            preprocessed = self.preprocess_image(image_path)

            # === Detect tables ===
            table_bboxes = self.detect_tables(preprocessed)
            print(f"📐 Detected {len(table_bboxes)} table(s)")

            # Mask out tables from image to focus on text
            text_only = self.remove_regions(image, table_bboxes)

            # === Extract text blocks ===
            text_blocks = self.extract_layout_blocks(text_only)
            for tb in text_blocks:
                if tb["text"].strip():
                    all_content_blocks.append({
                        "page": i + 1,
                        "type": "text",
                        "content": tb["text"],
                        "bbox": tb["bbox"]
                    })

            # === Extract table content ===
            for bbox in table_bboxes:
                table = self.extract_table_from_region(image, bbox)
                if isinstance(table, pd.DataFrame) and not table.empty and table.shape[1] > 1:
                    all_content_blocks.append({
                        "page": i + 1,
                        "type": "table",
                        "content": table
                    })

        return all_content_blocks

    def detect_tables(self, image):
        # Detects rectangular boxes indicating tables
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))

        horizontal = cv2.erode(image, kernel_h, iterations=1)
        horizontal = cv2.dilate(horizontal, kernel_h, iterations=1)

        vertical = cv2.erode(image, kernel_v, iterations=1)
        vertical = cv2.dilate(vertical, kernel_v, iterations=1)

        table_mask = cv2.add(horizontal, vertical)

        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 1000]

        return boxes

    def extract_table_from_region(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        table_img = image[y1:y2, x1:x2]

        # Convert to gray + threshold
        gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        # Find lines (simplified)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

        # Combine line images
        table_structure = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        table_structure = cv2.dilate(table_structure, np.ones((2, 2), np.uint8), iterations=1)

        # OCR
        table_text = pytesseract.image_to_string(table_img, config="--psm 6")
        lines = [line.strip() for line in table_text.split("\n") if line.strip()]
        df = pd.DataFrame(lines)
        return df

    def export_structured_to_excel(self, content_blocks, excel_path):
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            for i, block in enumerate(content_blocks):
                if block["type"] == "text":
                    df = pd.DataFrame({"Content": [block["content"]]})
                    df.to_excel(writer, sheet_name=f"Page{block['page']}_text", index=False)
                elif block["type"] == "table":
                    df = block["content"]
                    df.to_excel(writer, sheet_name=f"Page{block['page']}_table_{i}", index=False)

    def export_structured_to_pdf(self, blocks, output_path):
        doc = SimpleDocTemplate(output_path)
        flowables = []
        styles = getSampleStyleSheet()

        for block in blocks:
            if block["type"] == "text":
                text = block["content"].replace("\n", "<br/>")
                para = Paragraph(text, styles["Normal"])
                flowables.append(para)
                flowables.append(Spacer(1, 12))
            elif block["type"] == "table":
                try:
                    df = block["content"]
                    table_data = [df.columns.tolist()] + df.fillna("").astype(str).values.tolist()
                    t = Table(table_data)
                    flowables.append(t)
                    flowables.append(Spacer(1, 12))
                except Exception as e:
                    error_para = Paragraph(f"<b>⚠️ Table parse error:</b> {e}", styles["Normal"])
                    flowables.append(error_para)

        doc.build(flowables)
 
    def extract_layout_blocks(self, image):
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        data = pytesseract.image_to_data(pil_img, output_type=Output.DATAFRAME)
        data = data.dropna(subset=["text"])
        blocks = []

        for _, row in data.iterrows():
            text = row["text"].strip()
            if not text:
                continue

            x, y, w, h = row["left"], row["top"], row["width"], row["height"]
            bbox = (int(x), int(y), int(x + w), int(y + h))
            blocks.append({"type": "text", "text": text, "bbox": bbox})

        return blocks

    def classify_block_type(self, df):
        if len(df) < 4:
            return "text"  # too small to be a table

        # Count how many lines have multiple words horizontally aligned (table-like)
        y_rounded = df["top"].round(-1)
        line_counts = y_rounded.value_counts()

        # If there are at least 2 lines with 3+ words horizontally aligned → table
        potential_table_rows = (line_counts >= 3).sum()
        if potential_table_rows >= 2:
            return "table"

        return "text"

    def detect_layout_correct(page):
        elements = []
        for el in page.layout:
            if hasattr(el, "get_text"):
                # You can optionally filter out whitespace or very small regions
                x0, y0, x1, y1 = map(int, [el.x0, el.top, el.x1, el.bottom])
                elements.append(("text", (x0, y0, x1, y1)))
        return elements

    def remove_regions(self, image, bboxes):
        """Remove specified regions from an image by filling them with white rectangles."""
        masked_image = image.copy()
        for x0, y0, x1, y1 in bboxes:
            cv2.rectangle(masked_image, (x0, y0), (x1, y1), (255, 255, 255), thickness=-1)
        return masked_image

    def export_to_text(self, blocks, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            for block in blocks:
                f.write(f"--- Page {block['page']} | Type: {block['type']} ---\n")
                if block["type"] == "text":
                    f.write(block["content"].strip() + "\n\n")
                elif block["type"] == "table":
                    if isinstance(block["content"], pd.DataFrame):
                        f.write(block["content"].to_string(index=False) + "\n\n")
                    else:
                        f.write("[⚠️ Invalid table content]\n\n")
        print(f"✅ Exported structured content to text: {output_path}")
