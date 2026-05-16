#!/bin/bash
echo "Starting FastAPI Backend..."
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &

# Wait for API to load models
echo "Waiting for backend to initialize..."
sleep 15

echo "Starting Streamlit UI..."
streamlit run ui/app.py --server.port 7860 --server.address 0.0.0.0
