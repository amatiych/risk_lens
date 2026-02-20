"""LLM provider configuration endpoints."""

from fastapi import APIRouter

from backend.api.schemas import ProviderResponse, ProviderUpdateRequest
from backend.api.dependencies import get_provider, set_provider

router = APIRouter(prefix="/api/v1/config", tags=["config"])


@router.get("/provider", response_model=ProviderResponse)
async def get_current_provider():
    return ProviderResponse(provider=get_provider())


@router.put("/provider", response_model=ProviderResponse)
async def update_provider(request: ProviderUpdateRequest):
    set_provider(request.provider)
    return ProviderResponse(provider=request.provider)
