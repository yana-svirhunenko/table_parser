import os
from typing import List, Tuple
from PIL import Image
import numpy as np
import cv2
import pytesseract
from pdf2image import convert_from_path
import torch
from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection

from .utils import image_to_base64, generate_table_name

import warnings
warnings.filterwarnings("ignore")


class TableDetector:
    def __init__(
        self,
        detection_model_name: str = "microsoft/table-transformer-detection",
        max_side: int = 2000,
        threshold: float = 0.2
    ):
        self.detection_model_name = detection_model_name
        self.max_side = max_side
        self.threshold = threshold
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Load detection model and set device."""
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.feature_extractor = DetrFeatureExtractor(
            do_resize=True,
            size=self.max_side,
            max_size=self.max_side
        )
        self.detection_model = TableTransformerForObjectDetection.from_pretrained(
            self.detection_model_name
        ).to(self.device)

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image if larger than max_side, keeping aspect ratio."""
        w, h = image.size
        scale = self.max_side / max(w, h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
        return image

    def _detect_table_bounds(self, image: Image.Image) -> List[Tuple[float, float, float, float]]:
        """
        Detect bounding boxes of tables in the image.
        Returns list of boxes [x1, y1, x2, y2].
        """
        encoding = self.feature_extractor(image, return_tensors="pt")
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.detection_model(**encoding)

        width, height = image.size
        results = self.feature_extractor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=[(height, width)]
        )[0]

        return [
            box.tolist()
            for label, box in zip(results["labels"], results["boxes"])
            if self.detection_model.config.id2label[label.item()] == "table"
        ]

    def extract_tables(self, file_path: str):
        """
        Extract tables from a PDF or image file.
        Returns a list of (PIL image, list of bounding boxes).
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            images = convert_from_path(file_path)
        else:
            images = [Image.open(file_path).convert("RGB")]

        results = []
        for img in images:
            img = self._resize_image(img)
            boxes = self._detect_table_bounds(img.convert("RGB"))
            results.append((img, boxes))
        return results


def detect_tables(detector: TableDetector, file_path: str):
    """
    High-level function to detect tables in a file and annotate images.
    Returns dictionary with:
      - num_tables
      - tables (name and bbox)
      - annotated_image (cv2 image)
      - annotated_image_base64 (str)
    """
    results = detector.extract_tables(file_path)

    final_results = []
    annotated_images = []
    annotated_images_b64 = []

    for img, boxes in results:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_copy = img_cv.copy()

        for j, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            w, h = x2 - x1, y2 - y1

            # Extract header ROI and recognize text using OCR
            table_roi = img_cv[y1:y2, x1:x2]
            header_roi = table_roi[0:max(1, h // 4), :]
            header_text = pytesseract.image_to_string(header_roi, config="--psm 6").strip().replace("\n", " ")

            table_name = generate_table_name(header_text) if header_text else f"Table {j+1}"
            final_results.append({
                "name": table_name,
                "bbox": [x1, y1, x2, y2]
            })

            # Annotate image
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img_copy, table_name, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        annotated_images.append(img_copy)
        annotated_images_b64.append(image_to_base64(img_copy))

    return {
        "num_tables": len(final_results),
        "tables": final_results,
        "annotated_image": annotated_images[0] if annotated_images else None,
        "annotated_image_base64": annotated_images_b64[0] if annotated_images_b64 else None
    }
