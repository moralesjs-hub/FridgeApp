# Project Proposal: AI Food Recognition and Recipe Suggestions

**Course:** CSCI 270 — Applied AI, College of Charleston  
**Team:** Philip Pagana, Snovi Patel, Jaden Morales, Amelia West

## Problem statement

People often have ingredients at home but are unsure what to cook with them. This project explores whether a computer-vision model can identify a useful subset of ingredients from a refrigerator photo and connect those ingredients to practical recipe suggestions. The interface will make recognition errors easy to correct and will explain which ingredients match and which are missing.

## Proposed solution

The application uses an explainable, two-stage local computer-vision detection pipeline implemented with torchvision:
1. **Stage 1 (Object Detection):** Faster R-CNN with a MobileNetV3-Large FPN backbone (pretrained on COCO) detects individual food items with spatial bounding boxes.
2. **Stage 2 (Classification):** MobileNetV3-Small (pretrained on ImageNet) provides scene-level multi-label predictions across nightshades, alliums, dairy, meats, and seafood.

A 200+ alias vocabulary normalizes detected objects and ingredients into canonical keys. An enhanced, staple-aware recipe engine matches detected ingredients against a 60-recipe catalog. Pantry staples (salt, pepper, oil, flour) are automatically assumed present, and realistic culinary substitutions (e.g. lime for lemon) are granted 80% partial credit. The interface provides confidence adjustment, bounding box visualization, nutrition scaling, and transparent ingredient gap analysis.

The project adheres strictly to course guidelines: it uses local, explainable CV and heuristic algorithms, never sending user images to external cloud APIs or black-box vision-LLMs. Bundled sample images allow offline demonstration without model downloads.

## Data

- **Recipe data:** A curated, validated catalog of 120 recipes in `data/processed/recipes.csv`, spanning breakfast, lunch, dinner, soups, and sides with ingredient lists, preparation steps, and macronutrient profiles.
- **Images:** Bundled illustrative demo images in `data/raw/` for reproducible testing, plus support for arbitrary user image uploads processed in-memory.
- **External datasets:** None required; raw/processed data boundaries are preserved.

## AI and ranking approach

1. Preprocess uploaded images using standard torchvision transforms and run local two-stage inference (Faster R-CNN + MobileNetV3-Small).
2. Filter detections via user-specified confidence thresholds and map labels to canonical ingredients.
3. Compute recipe match scores using weighted core-ingredient matching:
   $$\text{score} = \frac{\sum \text{matched\_core\_weights}}{|core\_ingredients|}$$
   with 0.8 partial credit for registered culinary substitutions.
4. Present ranked recipes with missing ingredient callouts, substitution notes, and serving-scaled nutritional breakdowns.

ImageNet is a general object-classification dataset, not a food-specific detector. The mapping is therefore intentionally limited, and human correction is a core part of the workflow.

## Success criteria

- The application launches with `streamlit run src/app.py` after installing the pinned requirements.
- A bundled sample can exercise the interface offline.
- Users can upload an image, adjust the threshold, edit ingredients, and receive ranked recipes with transparent match and nutrition details.
- Source modules remain separated, input data decisions are documented, and limitations are clearly presented.

## Timeline

| Phase | Work | Deliverable |
|---|---|---|
| 1. Scope and repository | Agree on requirements, folder layout, AI policy, and data boundaries | Repository skeleton and coding rules |
| 2. Research and design | Compare lightweight vision models and matching approaches | Proposal and design document |
| 3. Data and AI modules | Curate starter recipes; implement normalization, inference, matching, and nutrition | Modular source and processed CSV |
| 4. Interface integration | Build Streamlit upload, review controls, demo images, and recipe cards | Runnable end-to-end app |
| 5. Validation and polish | Verify installation/launch, correct documentation, and review limitations | Final README and demo |

The phases are ordered project milestones; the team should record actual dates and incremental contributions in its course repository history.
