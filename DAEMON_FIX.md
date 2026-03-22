# Daemon Startup Fix

Fixed Pydantic-core version conflict (2.14.6/2.41.5) via pip environment alignment.

## Issue
Daemon startup was blocked by incompatible pydantic-core versions required by chromadb/langchain.

## Solution
Installed pydantic-core 2.41.5 to align with langchain/chromadb requirements while keeping FastAPI compatible.

## Result
Daemon now starts successfully at http://127.0.0.1:8192 with:
- FastAPI endpoints live
- File upload working
- Health checks passing
- UI accessible
