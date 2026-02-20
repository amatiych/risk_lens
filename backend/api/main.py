"""FastAPI application for Risk Lens portfolio analysis."""

import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.api.routers import portfolio, chat, config

app = FastAPI(
    title="Risk Lens API",
    version="1.0.0",
    description="Portfolio risk analysis API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1974",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(portfolio.router)
app.include_router(chat.router)
app.include_router(config.router)


@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}


# Serve React frontend in production
STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "web" / "dist"

if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")
