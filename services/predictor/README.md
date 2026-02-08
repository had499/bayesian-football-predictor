# Football Predictor API - Docker

## Quick Start

### Using Docker Compose (Recommended)

```bash
cd services/predictor
docker compose up --build
```

The API will be available at http://localhost:8956

### Using Docker directly

```bash
# Build the image
docker build -t football-predictor -f services/predictor/Dockerfile .

# Run the container
docker run -p 8956:8956 -v $(pwd)/services/predictor/data:/app/data football-predictor
```

## Endpoints

- `GET /` - API info
- `POST /train` - Train the model
- `GET /predict` - Get predictions for the next round
- `GET /status` - Check model status

## Documentation

Interactive API docs: http://localhost:8956/docs

## Data Persistence

Model data is stored in `services/predictor/data/` and persisted via Docker volume.
