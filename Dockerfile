# FROM python:3.11-slim

# WORKDIR /app

# RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# FROM python:3.11-slim

# WORKDIR /app

# RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# EXPOSE 8501

# CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (curl for uv, tesseract-ocr for OCR, libgomp1 for FAISS)
RUN apt-get update && apt-get install -y \
    curl \
    tesseract-ocr \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "rag_system.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
