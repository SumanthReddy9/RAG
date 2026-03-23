#!/bin/bash

set -e

ollama serve & 

sleep 15

# Pull the model
ollama pull tinyllama || true

# Start FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000