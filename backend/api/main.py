"""FastAPI application for Risk Lens portfolio analysis."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routers import portfolio, chat, config

app = FastAPI(
    title="Risk Lens API",
    version="1.0.0",
    description="Portfolio risk analysis API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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
