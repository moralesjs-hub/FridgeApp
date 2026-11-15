"""Cloud-based computer vision and ingredient detection service using multimodal AI."""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel, Field

from src.utils import normalize_ingredient, normalize_ingredients

# Load environment configuration
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

logger = logging.getLogger("fridgeapp.vision")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [FridgeApp.Vision] %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Category color map for visualization
CATEGORY_COLORS = {
    "produce": "#2ecc71",     # Emerald green
    "dairy": "#3498db",       # Blue
    "protein": "#e74c3c",     # Coral red
    "condiment": "#f39c12",   # Orange
    "beverage": "#9b59b6",    # Purple
    "pantry": "#1abc9c",      # Turquoise
    "leftover": "#e67e22",    # Amber
    "other": "#95a5a6",       # Gray
}


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates (0.0 to 1.0)."""

    ymin: float = Field(ge=0.0, le=1.0)
    xmin: float = Field(ge=0.0, le=1.0)
    ymax: float = Field(ge=0.0, le=1.0)
    xmax: float = Field(ge=0.0, le=1.0)

    def to_pixel_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert normalized coordinates to absolute pixel box (left, top, right, bottom)."""
        left = int(self.xmin * width)
        top = int(self.ymin * height)
        right = int(self.xmax * width)
        bottom = int(self.ymax * height)
        return left, top, max(right, left + 1), max(bottom, top + 1)


class RawDetection(BaseModel):
    """Raw entity detected by the multimodal cloud model."""

    name: str = Field(description="Name of detected food item (e.g. 'roma tomato', 'whole milk carton')")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence score")
    category: str = Field(
        description="Category: produce, dairy, protein, condiment, beverage, pantry, leftover, other",
    )
    box_2d: list[int] = Field(
        description="2D bounding box [ymin, xmin, ymax, xmax] normalized on a 0 to 1000 integer scale.",
    )


class DetectionSchema(BaseModel):
    """Root structured output schema for the cloud vision model."""

    ingredients: list[RawDetection] = Field(
        description="List of all visible food items, produce, containers, and ingredients detected in the image."
    )


@dataclass
class DetectedEntity:
    """Refined food entity with canonical naming and bounding coordinates."""

    name: str
    normalized_name: str
    confidence: float
    category: str
    box: BoundingBox | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "normalized_name": self.normalized_name,
            "confidence": round(self.confidence, 4),
            "category": self.category,
            "box": self.box.model_dump() if self.box else None,
        }


@dataclass
class VisionAnalysisResult:
    """Full outcome of cloud vision analysis."""

    success: bool
    provider: str
    model: str
    latency_ms: float
    entities: list[DetectedEntity]
    unique_ingredients: list[str]
    annotated_image: Image.Image | None = None
    error: str | None = None


class CloudVisionService:
    """Production-grade cloud vision service powered by Google Gemini multimodal API."""

    PRIMARY_MODELS = [
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-flash-latest",
    ]

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._client: Any = None
        self._cache: dict[str, VisionAnalysisResult] = {}

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. Add it to your .env file or environment variables."
            )
        try:
            from google import genai
            from google.genai import types
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=15000),
            )
            return self._client
        except Exception as exc:
            logger.error("Failed to initialize Google GenAI client: %s", exc)
            raise RuntimeError(f"Could not initialize Cloud Vision Client: {exc}") from exc

    @staticmethod
    def preprocess_image(
        source: bytes | str | Path | Image.Image,
        max_dimension: int = 1280,
    ) -> tuple[Image.Image, bytes]:
        """Validate, normalize orientation, and optimize image payload for cloud processing."""
        try:
            if isinstance(source, Image.Image):
                img = source
            elif isinstance(source, (str, Path)):
                img = Image.open(source)
            else:
                img = Image.open(io.BytesIO(source))

            # Transpose according to EXIF orientation
            img = ImageOps.exif_transpose(img) or img
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            elif img.mode == "L":
                img = img.convert("RGB")

            # Maintain aspect ratio thumbnailing
            if max(img.size) > max_dimension:
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85, optimize=True)
            compressed_bytes = buf.getvalue()
            return img, compressed_bytes
        except Exception as exc:
            raise ValueError(f"Invalid or unreadable image payload: {exc}") from exc

    def detect_ingredients(
        self,
        image_input: bytes | str | Path | Image.Image,
        confidence_threshold: float = 0.20,
        annotate: bool = True,
    ) -> VisionAnalysisResult:
        """Analyze a refrigerator/produce image and extract grounded ingredients with bounding boxes."""
        start_time = time.perf_counter()
        img, compressed_bytes = self.preprocess_image(image_input)
        digest = sha256(compressed_bytes).hexdigest()

        # Check local in-memory cache
        if digest in self._cache:
            logger.info("Serving cloud vision result from memory cache (digest=%s...)", digest[:8])
            return self._cache[digest]

        client = self._get_client()

        prompt = (
            "Analyze this refrigerator, pantry, or produce photo carefully.\n"
            "Identify ALL visible individual food items, produce, condiments, dairy products, "
            "containers, and beverages.\n"
            "For each detected food item, provide:\n"
            "1. A descriptive name (e.g. 'bell pepper', 'cherry tomatoes', 'egg carton', 'almond milk')\n"
            "2. Confidence score between 0.0 and 1.0\n"
            "3. Category: produce, dairy, protein, condiment, beverage, pantry, leftover, or other\n"
            "4. Accurate 2D bounding box [ymin, xmin, ymax, xmax] normalized on a 0 to 1000 integer scale.\n"
            "Distinguish individual clustered items on shelves or inside produce drawers.\n"
            "Adhere strictly to the structured JSON schema."
        )

        from google.genai import types

        last_error = None
        for model_name in self.PRIMARY_MODELS:
            try:
                logger.info("Dispatching image payload to cloud vision model: %s", model_name)
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=compressed_bytes, mime_type="image/jpeg"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=DetectionSchema,
                        temperature=0.1,
                    ),
                )

                parsed_json = response.text
                schema_result = DetectionSchema.model_validate_json(parsed_json)
                entities = self._process_raw_detections(
                    schema_result.ingredients,
                    confidence_threshold,
                )

                annotated = self.annotate_image(img, entities) if annotate else None
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                unique_ings = normalize_ingredients([e.normalized_name for e in entities])

                result = VisionAnalysisResult(
                    success=True,
                    provider="Google Gemini Multimodal Cloud Vision",
                    model=model_name,
                    latency_ms=round(elapsed_ms, 2),
                    entities=entities,
                    unique_ingredients=unique_ings,
                    annotated_image=annotated,
                )

                self._cache[digest] = result
                logger.info(
                    "Cloud vision successfully identified %d items (%d unique) in %.1f ms",
                    len(entities),
                    len(unique_ings),
                    elapsed_ms,
                )
                return result
            except Exception as exc:
                logger.warning("Cloud vision attempt failed with model %s: %s", model_name, exc)
                last_error = exc

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        error_msg = f"All cloud vision endpoints failed. Last error: {last_error}"
        logger.error(error_msg)
        return VisionAnalysisResult(
            success=False,
            provider="Google Gemini Multimodal Cloud Vision",
            model="failed",
            latency_ms=round(elapsed_ms, 2),
            entities=[],
            unique_ingredients=[],
            error=str(last_error),
        )

    @staticmethod
    def _process_raw_detections(
        raw_items: list[RawDetection],
        confidence_threshold: float,
    ) -> list[DetectedEntity]:
        """Convert raw detections to normalized entities with normalized bounding boxes."""
        entities: list[DetectedEntity] = []
        for item in raw_items:
            if item.confidence < confidence_threshold:
                continue

            normalized = normalize_ingredient(item.name)
            if not normalized:
                continue

            bbox = None
            if item.box_2d and len(item.box_2d) == 4:
                ymin, xmin, ymax, xmax = [v / 1000.0 for v in item.box_2d]
                # Clamp coordinates safely between 0.0 and 1.0
                ymin = max(0.0, min(1.0, ymin))
                xmin = max(0.0, min(1.0, xmin))
                ymax = max(0.0, min(1.0, ymax))
                xmax = max(0.0, min(1.0, xmax))
                if ymax > ymin and xmax > xmin:
                    bbox = BoundingBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)

            entities.append(
                DetectedEntity(
                    name=item.name.strip().lower(),
                    normalized_name=normalized,
                    confidence=item.confidence,
                    category=item.category.lower(),
                    box=bbox,
                )
            )
        return entities

    @staticmethod
    def annotate_image(
        image: Image.Image,
        entities: list[DetectedEntity],
    ) -> Image.Image:
        """Render sharp color-coded bounding boxes and label badges onto the image."""
        annotated = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw = ImageDraw.Draw(annotated)

        width, height = annotated.size

        # Try to load a clean truetype font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", size=max(12, int(height * 0.022)))
        except OSError:
            font = ImageFont.load_default()

        for entity in entities:
            if not entity.box:
                continue
            color_hex = CATEGORY_COLORS.get(entity.category, CATEGORY_COLORS["other"])
            color_rgb = ImageColor.getrgb(color_hex)
            left, top, right, bottom = entity.box.to_pixel_box(width, height)

            # Draw translucent fill on overlay
            fill_color = (*color_rgb, 40)
            draw_overlay.rectangle([left, top, right, bottom], fill=fill_color)

            # Draw thick bounding outline
            draw.rectangle([left, top, right, bottom], outline=color_hex, width=3)

            # Prepare badge text
            label_text = f"{entity.normalized_name} {entity.confidence:.0%}"

            # Calculate text dimensions
            if hasattr(font, "getbbox"):
                bbox = font.getbbox(label_text)
                t_width = bbox[2] - bbox[0]
                t_height = bbox[3] - bbox[1]
            else:
                t_width, t_height = 80, 16

            badge_padding = 4
            badge_left = left
            badge_top = max(0, top - t_height - (badge_padding * 2))
            badge_right = badge_left + t_width + (badge_padding * 2)
            badge_bottom = badge_top + t_height + (badge_padding * 2)

            # Draw label badge background
            draw.rectangle([badge_left, badge_top, badge_right, badge_bottom], fill=color_hex)
            # Draw label text in white
            draw.text(
                (badge_left + badge_padding, badge_top + badge_padding),
                label_text,
                fill="white",
                font=font,
            )

        # Composite translucent overlay onto main image
        final_image = Image.alpha_composite(annotated, overlay).convert("RGB")
        return final_image
