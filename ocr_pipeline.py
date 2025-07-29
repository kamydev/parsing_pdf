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

    def extract_text(self, image):
        text = pytesseract.image_to_string(image, lang='eng')
        return text

    def export_to_excel(self, text_list, excel_path):
        df = pd.DataFrame({'Page': list(range(1, len(text_list)+1)), 'Content': text_list})
        df.to_excel(excel_path, index=False)

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
                table_text = self.extract_table_from_region(image, bbox, index=f"{i+1}_{j+1}")
                if table_text.strip():
                    all_content_blocks.append({
                        "page": i + 1,
                        "type": "table",
                        "content": table_text.strip()
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

    def extract_table_from_region(self, image, bbox, index=0):
        """
        Extract text from a detected table region using PaddleOCR.
        Save the crop to disk and OCR it via image path (more stable).
        """
        import cv2
        import numpy as np

        x1, y1, x2, y2 = map(int, bbox)
        table_image = image[y1:y2, x1:x2]

        if table_image.shape[0] < 10 or table_image.shape[1] < 10:
            return "[⚠️ Skipped tiny region]"

        # Save to disk for OCR
        crop_path = os.path.join(self.output_dir, f"page_table_{index}.png")
        cv2.imwrite(crop_path, table_image)

        try:
            result = self.ocr.ocr(crop_path)

            if not result or not isinstance(result[0], list):
                return "[⚠️ No OCR result]"

            texts = []
            for line in result[0]:
                if (
                    isinstance(line, list)
                    and len(line) == 2
                    and isinstance(line[1], (list, tuple))
                    and len(line[1]) == 2
                ):
                    text = line[1][0].strip()
                    if text:
                        texts.append(text)
                else:
                    print(f"⚠️ Skipped malformed OCR line: {line}")

            return "\n".join(texts) if texts else "[⚠️ OCR found no valid text]"

        except Exception as e:
            print(f"⚠️ OCR result parsing failed: {e}")
            return "[⚠️ OCR result parsing failed]"

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

    def export_to_txt(self, content_blocks, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            current_page = None
            for block in content_blocks:
                if block["page"] != current_page:
                    current_page = block["page"]
                    f.write(f"\n--- Page {current_page} ---\n")

                if block["type"] == "text":
                    f.write(block["content"] + "\n")
                elif block["type"] == "table":
                    f.write("[Table detected]\n")
                    f.write(block["content"] + "\n")
