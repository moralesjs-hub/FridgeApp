# AI Coding Rules

## Project constraints

- Use traditional computer vision and explainable algorithms. Do not send images to a third-party vision-language model or use a pure API wrapper.
- Keep vision, recipe matching, nutrition, normalization, and interface code in their existing independent modules.
- Keep original/external inputs in `data/raw/` and cleaned or curated application data in `data/processed/`. Never silently overwrite source data.
- State dataset origin, license, preprocessing, and any synthetic/demo data decisions in project documentation.
- Do not describe the bundled demo detections as model predictions.

## Code quality

- Follow PEP 8, add type hints to public functions and methods, and document non-obvious public behavior.
- Validate external inputs and handle missing, malformed, or unavailable data with explicit user-facing errors.
- Keep the app usable without network access by preserving the bundled sample workflow.
- Do not commit credentials, downloaded model weights, user photos, or large raw datasets.
- Add or update focused tests when behavior changes, and verify imports and the documented launch command.

## Computer-vision limitations

- ImageNet labels are not a food-specific ingredient detector. Explain that mapping is limited and predictions can be wrong.
- Preserve confidence thresholds and allow people to review and edit recognized ingredients.
- Do not imply that an image classifier detects every food item, quantities, freshness, allergens, or nutritional content.
- Treat nutrition values as estimates, not health advice.
