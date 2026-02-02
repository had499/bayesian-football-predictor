FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PyTensor/PyMC
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (copy entire src with all subdirectories)
COPY src/ /app/src/
COPY services/predictor/predictor.py /app/

# Create data directory for model persistence
RUN mkdir -p /app/data

# Set Python path to include src directory
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose port
EXPOSE 8956

# Run the application
CMD ["python", "/app/predictor.py"]
