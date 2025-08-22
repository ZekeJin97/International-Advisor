FROM python:3.11-slim

# Install system dependencies for Tesseract and PDF processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    tesseract-ocr-chi-sim \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads vectorstores

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "main_langchain:app", "--host", "0.0.0.0", "--port", "8080"]