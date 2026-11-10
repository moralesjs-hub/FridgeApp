"""Focused tests for the Cloud Vision service, FastAPI endpoints, and image grounding."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app
from src.cloud_vision import (
    BoundingBox,
    CloudVisionService,
    DetectedEntity,
    RawDetection,
    VisionAnalysisResult,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE_PATH = ROOT / "data" / "raw" / "sample_produce.png"


def test_bounding_box_pixel_conversion() -> None:
    box = BoundingBox(ymin=0.1, xmin=0.2, ymax=0.8, xmax=0.9)
    left, top, right, bottom = box.to_pixel_box(width=1000, height=500)
    assert left == 200
    assert top == 50
    assert right == 900
    assert bottom == 400


def test_preprocess_image_validation_and_compression() -> None:
    # Test valid image
    sample_bytes = SAMPLE_IMAGE_PATH.read_bytes()
    img, compressed = CloudVisionService.preprocess_image(sample_bytes, max_dimension=800)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert max(img.size) <= 800
    assert len(compressed) > 0

    # Test corrupted image payload
    with pytest.raises(ValueError, match="Invalid or unreadable image payload"):
        CloudVisionService.preprocess_image(b"corrupted raw data")


def test_process_raw_detections_normalizes_and_clamps_coordinates() -> None:
    raw_items = [
        RawDetection(
            name="fresh sliced tomatoes",
            confidence=0.95,
            category="produce",
            box_2d=[100, 200, 500, 600],
        ),
        RawDetection(
            name="almond milk carton",
            confidence=0.10,  # Below threshold
            category="beverage",
            box_2d=[0, 0, 100, 100],
        ),
    ]

    entities = CloudVisionService._process_raw_detections(raw_items, confidence_threshold=0.25)
    assert len(entities) == 1
    assert entities[0].normalized_name == "tomato"
    assert entities[0].confidence == 0.95
    assert entities[0].category == "produce"
    assert entities[0].box is not None
    assert entities[0].box.ymin == pytest.approx(0.1)
    assert entities[0].box.xmin == pytest.approx(0.2)
    assert entities[0].box.ymax == pytest.approx(0.5)
    assert entities[0].box.xmax == pytest.approx(0.6)


def test_annotate_image_draws_bounding_boxes() -> None:
    test_img = Image.new("RGB", (400, 400), color="white")
    entity = DetectedEntity(
        name="bell pepper",
        normalized_name="bell pepper",
        confidence=0.92,
        category="produce",
        box=BoundingBox(ymin=0.2, xmin=0.2, ymax=0.6, xmax=0.6),
    )

    annotated = CloudVisionService.annotate_image(test_img, [entity])
    assert isinstance(annotated, Image.Image)
    assert annotated.size == (400, 400)
    # The canvas should no longer be completely plain white after drawing
    assert annotated.tobytes() != test_img.tobytes()


def test_fastapi_health_check() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Google Gemini" in data["provider"]
    assert data["recipe_catalog_size"] >= 25


def test_fastapi_detect_endpoint_validation_and_mock_response(monkeypatch) -> None:
    client = TestClient(app)

    # Rejects non-image upload
    bad_res = client.post(
        "/api/detect",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert bad_res.status_code == 400

    # Rejects empty file
    empty_res = client.post(
        "/api/detect",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert empty_res.status_code == 400

    # Mock CloudVisionService.detect_ingredients
    mock_entity = DetectedEntity(
        name="roma tomato",
        normalized_name="tomato",
        confidence=0.97,
        category="produce",
        box=BoundingBox(ymin=0.1, xmin=0.1, ymax=0.5, xmax=0.5),
    )
    mock_result = VisionAnalysisResult(
        success=True,
        provider="Mock Cloud Vision",
        model="gemini-mock",
        latency_ms=120.5,
        entities=[mock_entity],
        unique_ingredients=["tomato"],
    )

    import main
    monkeypatch.setattr(main.vision_service, "detect_ingredients", lambda *a, **k: mock_result)

    sample_bytes = SAMPLE_IMAGE_PATH.read_bytes()
    good_res = client.post(
        "/api/detect",
        files={"file": ("sample.png", sample_bytes, "image/png")},
    )
    assert good_res.status_code == 200
    payload = good_res.json()
    assert payload["success"] is True
    assert payload["model"] == "gemini-mock"
    assert payload["unique_ingredients"] == ["tomato"]
    assert len(payload["entities"]) == 1
    assert payload["entities"][0]["normalized_name"] == "tomato"


def test_fastapi_analyze_and_match_endpoint(monkeypatch) -> None:
    client = TestClient(app)

    mock_entity = DetectedEntity(
        name="egg",
        normalized_name="egg",
        confidence=0.99,
        category="dairy",
        box=BoundingBox(ymin=0.1, xmin=0.1, ymax=0.4, xmax=0.4),
    )
    mock_result = VisionAnalysisResult(
        success=True,
        provider="Mock Cloud Vision",
        model="gemini-mock",
        latency_ms=150.0,
        entities=[mock_entity],
        unique_ingredients=["egg", "cheese", "spinach"],
    )

    import main
    monkeypatch.setattr(main.vision_service, "detect_ingredients", lambda *a, **k: mock_result)

    sample_bytes = SAMPLE_IMAGE_PATH.read_bytes()
    response = client.post(
        "/api/analyze-and-match",
        files={"file": ("fridge.png", sample_bytes, "image/png")},
        data={"max_recipes": 3, "min_match_percent": 10.0, "servings": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["detection"]["success"] is True
    assert len(body["recipes"]) > 0
    assert all("egg" in r["matched_ingredients"] or "cheese" in r["matched_ingredients"] for r in body["recipes"])
    assert "estimated_nutrition" in body["recipes"][0]
