FROM python:3.11-slim

# Update package lists and install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-dri \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    libgtk-3-0 \
    pkg-config \
    libhdf5-dev \
    gcc \
    g++ \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml* ./

# Install Python dependencies
RUN if [ -f "pyproject.toml" ]; then \
        uv venv && \
        . .venv/bin/activate && \
        uv sync; \
    else \
        pip install -r requirements.txt; \
    fi

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "main.py"]