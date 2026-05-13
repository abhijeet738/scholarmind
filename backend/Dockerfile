# Use official Python lightweight image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# We use the CPU-only version of PyTorch to save space (~200MB vs ~2GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose port (HF Spaces uses 7860 by default)
EXPOSE 7860

# Create a non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
# Give the user ownership of the /app directory so it can write model caches
RUN chown -R user:user /app
USER user

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
