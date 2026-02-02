FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY services/predictor/predictor.py ./services/predictor/

# Create data directory for model persistence
RUN mkdir -p data

# Expose port
EXPOSE 8956

# Run the application
CMD ["python", "services/predictor/predictor.py"]
