# YBS410 - Image Processing Backend

This is a FastAPI backend for YBS410 project - image processing using YOLO model to detect defects and products.

## Features

- Image processing using YOLO model
- Automatic model loading at startup
- REST API endpoints for image processing
- Health check and model information endpoints
- Automatic cleanup of processed images

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python main.py
```

The server will start on http://localhost:8000

## API Endpoints

### POST /process-image/
Process an image and get predictions.
- Request: Multipart form data with image file
- Response: JSON with predictions including class labels, confidence scores, and bounding boxes

### GET /health
Check if the server and model are running properly.

### GET /model-info
Get information about the model and supported formats.

## API Documentation

Once the server is running, you can access:
- Interactive API docs: http://localhost:8000/docs
- OpenAPI specification: http://localhost:8000/openapi.json