FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PyTensor/PyMC
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY setup.py .
COPY src/ ./src/
COPY requirements.txt .

# Install the package and dependencies
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY services/predictor/predictor.py .

# Create data directory
RUN mkdir -p data

EXPOSE 8956

CMD ["python", "predictor.py"]
