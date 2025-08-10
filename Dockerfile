FROM python:3.10-slim

# Install system dependencies needed for pdf2image, pytesseract, opencv, tesseract-ocr, and OpenCV's libGL
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    libsm6 libxext6 libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY uploads ./uploads
COPY output ./output

# Expose port the app runs on
EXPOSE 5001

CMD ["python", "-m", "src.app"]
