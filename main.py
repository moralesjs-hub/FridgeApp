"""Production-grade FastAPI backend service for FridgeApp Cloud Vision & Recipe Matching."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.cloud_vision import CloudVisionService
from src.nutrition import estimate_nutrition
from src.recipe_engine import RecipeMatcher

# Setup logging
logger = logging.getLogger("fridgeapp.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [FridgeApp.API] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

ROOT = Path(__file__).resolve().parent
RECIPE_PATH = ROOT / "data" / "processed" / "recipes.csv"

# Maximum upload payload size (15 MB)
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

app = FastAPI(
    title="FridgeApp Cloud Vision API",
    description="Multimodal food recognition, spatial bounding box detection, and recipe matching.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core services
vision_service = CloudVisionService()
recipe_matcher = RecipeMatcher(RECIPE_PATH)


class BoundingBoxResponse(BaseModel):
    ymin: float
    xmin: float
    ymax: float
    xmax: float


class EntityResponse(BaseModel):
    name: str
    normalized_name: str
    confidence: float
    category: str
    box: Optional[BoundingBoxResponse] = None


class DetectionResponse(BaseModel):
    success: bool
    provider: str
    model: str
    latency_ms: float
    item_count: int
    unique_ingredient_count: int
    unique_ingredients: list[str]
    entities: list[EntityResponse]
    error: Optional[str] = None


class MatchedRecipeResponse(BaseModel):
    id: str
    name: str
    match_score: float
    matched_ingredients: list[str]
    missing_ingredients: list[str]
    calories_per_serving: Optional[float] = None
    prep_time: Optional[float] = None
    instructions: list[str] = Field(default_factory=list)
    estimated_nutrition: dict[str, Optional[float]] = Field(default_factory=dict)


class AnalyzeAndMatchResponse(BaseModel):
    detection: DetectionResponse
    recipes: list[MatchedRecipeResponse]


async def _read_and_validate_upload(file: UploadFile) -> bytes:
    """Validate upload MIME type, size limit, and payload readability."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file must be an image. Received content type: {file.content_type}",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds the maximum allowed upload limit of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    return contents


@app.get("/")
def health_check() -> dict[str, Any]:
    """Service health check and provider info."""
    return {
        "status": "healthy",
        "service": "FridgeApp Cloud Vision & Recipe Engine",
        "version": "2.0.0",
        "provider": "Google Gemini Multimodal AI",
        "recipe_catalog_size": len(recipe_matcher.recipes),
    }


@app.post("/api/detect", response_model=DetectionResponse)
async def detect_ingredients(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(default=0.20),
) -> DetectionResponse:
    """Detect food ingredients with spatial bounding coordinates from an uploaded photo."""
    contents = await _read_and_validate_upload(file)

    try:
        result = vision_service.detect_ingredients(
            contents,
            confidence_threshold=confidence_threshold,
            annotate=False,
        )
    except Exception as exc:
        logger.error("Internal error during ingredient detection: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vision processing failed: {exc}",
        ) from exc

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cloud vision provider unavailable: {result.error}",
        )

    entity_list = [
        EntityResponse(
            name=e.name,
            normalized_name=e.normalized_name,
            confidence=e.confidence,
            category=e.category,
            box=BoundingBoxResponse(**e.box.model_dump()) if e.box else None,
        )
        for e in result.entities
    ]

    return DetectionResponse(
        success=result.success,
        provider=result.provider,
        model=result.model,
        latency_ms=result.latency_ms,
        item_count=len(entity_list),
        unique_ingredient_count=len(result.unique_ingredients),
        unique_ingredients=result.unique_ingredients,
        entities=entity_list,
        error=result.error,
    )


@app.post("/api/analyze-and-match", response_model=AnalyzeAndMatchResponse)
async def analyze_and_match(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(default=0.20),
    max_recipes: int = Form(default=6),
    min_match_percent: float = Form(default=20.0),
    servings: int = Form(default=1),
) -> AnalyzeAndMatchResponse:
    """Analyze image with cloud vision and immediately compute ranked recipe recommendations."""
    detection = await detect_ingredients(file, confidence_threshold)

    # Rank recipes using the detection's unique normalized ingredients
    ranked_recipes = recipe_matcher.match(detection.unique_ingredients)

    qualifying_recipes: list[MatchedRecipeResponse] = []
    for r in ranked_recipes:
        if r["match_score"] * 100.0 < min_match_percent:
            continue
        if len(qualifying_recipes) >= max_recipes:
            break

        nutrition = estimate_nutrition(r, servings=servings)
        qualifying_recipes.append(
            MatchedRecipeResponse(
                id=r["id"],
                name=r["name"],
                match_score=round(r["match_score"], 4),
                matched_ingredients=r["matched_ingredients"],
                missing_ingredients=r["missing_ingredients"],
                calories_per_serving=r["calories_per_serving"],
                prep_time=r["prep_time"],
                instructions=r["instructions"],
                estimated_nutrition=nutrition,
            )
        )

    return AnalyzeAndMatchResponse(
        detection=detection,
        recipes=qualifying_recipes,
    )
