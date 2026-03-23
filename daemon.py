#!/usr/bin/env python3
"""
Neuroboros Daemon Server - Serves the unified VIPER-NEUROBOROS substrate.
Port 8192 as per previous configuration.
"""
import argparse
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('hybrid_ouroboros')
sys.path.append('core')

app = FastAPI(title="Neuroboros Daemon", version="1.0.0")

# CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get("/health")
async def health_check() -> HealthResponse:
    return HealthResponse(status="healthy", message="Neuroboros Daemon running")

@app.get("/")
async def root():
    return {"message": "Neuroboros Daemon - VIPER-NEUROBOROS substrate server", "port": 8192}

# Placeholder endpoints for project components (extend as needed)
@app.get("/api/spiking/status")
async def spiking_status():
    return {"status": "ready", "components": ["viper", "fusion", "brian2"]}

@app.post("/api/inference/text")
async def text_inference(prompt: str):
    # Placeholder - integrate hybrid_ouroboros/inference/text_generate.py
    return {"input": prompt, "output": "[Placeholder generation]"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Neuroboros Daemon Server")
    parser.add_argument("--run-server", action="store_true", help="Run the FastAPI server")
    parser.add_argument("--host", default="127.0.0.1", help="Host IP")
    parser.add_argument("--port", type=int, default=8192, help="Port (default 8192)")
    args = parser.parse_args()

    if args.run_server:
        print(f"Starting Neuroboros Daemon on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        print("Use --run-server to start the server")
        parser.print_help()

