"""Gemini-powered multimodal fridge image analyzer and recipe generator."""

from __future__ import annotations

import io
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class Recipe(BaseModel):
    title: str = Field(description="Name of the recipe")
    cook_time_minutes: int = Field(description="Total prep and cook time in minutes")
    ingredients_used: list[str] = Field(description="Detected fridge ingredients used")
    missing_staples_assumed: list[str] = Field(description="Basic pantry staples assumed (oil, salt, etc.)")
    instructions: list[str] = Field(description="Cooking steps")


class FridgeAnalysisResult(BaseModel):
    detected_ingredients: list[str] = Field(description="All visible foods, produce, dairy, and leftovers")
    recipes: list[Recipe] = Field(description="2-3 customized recipes based strictly on contents")


class FridgeAnalyzer:
    """Analyze fridge images using Google Gemini."""

    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(f"GEMINI_API_KEY not found in environment or at {ENV_PATH}")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _compress_image(
        self,
        source: str | os.PathLike | bytes | Image.Image,
        max_size=(1024, 1024),
        quality=75,
    ) -> bytes:
        img = Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()

    def analyze_fridge(self, image_source: str | os.PathLike | bytes | Image.Image) -> FridgeAnalysisResult:
        compressed = self._compress_image(image_source)
        prompt = (
            "Analyze this refrigerator image.\n"
            "1. Detect all visible food items, produce, condiments, dairy, and leftovers.\n"
            "2. Generate 2 to 3 creative, practical recipes utilizing these ingredients "
            "(you may assume basic pantry staples like salt, pepper, oil, and water).\n"
            "Return output strictly adhering to the schema."
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[types.Part.from_bytes(data=compressed, mime_type="image/jpeg"), prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FridgeAnalysisResult,
                temperature=0.2,
            ),
        )
        return FridgeAnalysisResult.model_validate_json(response.text)
