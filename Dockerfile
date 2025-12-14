FROM python:3.10-slim

WORKDIR /app

# Install only minimal OpenCV dependencies (NO Vulkan)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "predict.py"]

