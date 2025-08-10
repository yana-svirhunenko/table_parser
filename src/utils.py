import cv2
import base64


def image_to_base64(image) -> str:
    """Encode a cv2 image to base64 string for embedding in HTML."""
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")


def generate_table_name(header_text: str) -> str:
    """
    Generate a human-readable table name based on the first few words
    of the detected header text.
    """
    words = [
        w.strip() for w in header_text.split()
        if w.strip() and all(c.isalnum() or c in (".", "-") for c in w)
    ]
    return " ".join(words[:5]) if words else "Table"
