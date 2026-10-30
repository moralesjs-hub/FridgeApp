# Project Proposal: AI Food Recognition and Recipe Suggestions

**Course:** CSCI 270 — Applied AI, College of Charleston  
**Team:** Philip Pagana, Snovi Patel, Jaden Morales, Amelia West

## Problem statement

People often have ingredients at home but are unsure what to cook with them. This project explores whether a computer-vision model can identify a useful subset of ingredients from a refrigerator photo and connect those ingredients to practical recipe suggestions. The interface will make recognition errors easy to correct and will explain which ingredients match and which are missing.

## Proposed solution

The application uses a locally executed pretrained MobileNetV3-Small image classifier from torchvision. A conservative mapping translates supported ImageNet labels into a small common-ingredient vocabulary. The model returns ranked labels and confidence values; users can change the confidence threshold, remove false detections, and add ingredients manually. A recipe engine ranks a curated catalog by the fraction of a recipe's normalized ingredients present in the user's list. It displays the score, missing ingredients, estimated nutrition, and cooking instructions.

The project does not use a third-party vision LLM or send user images to a remote inference API. Included sample images have deterministic, explicitly labeled demo detections so that graders can run the app without a model download.

## Data

- **Recipe data:** a project-curated starter catalog in `data/processed/recipes.csv`, with ingredient lists, instructions, approximate calories, and estimated macronutrients. It is intended for demonstration and is not represented as a Kaggle dataset.
- **Images:** three locally generated illustrative demo images in `data/raw/`. They are not training data. Users may upload their own images at runtime; uploads are not persisted by the application.
- **External datasets:** none are required. Any future dataset adoption must document origin, license, version, and cleaning steps, preserving the original in `data/raw/`.

## AI and ranking approach

1. Apply torchvision's MobileNetV3-Small ImageNet preprocessing and infer class probabilities locally.
2. Map only supported labels to normalized ingredient terms and filter using a configurable confidence threshold.
3. Normalize synonyms and preparation words consistently for both detected items and recipe ingredients.
4. Score each recipe as `number of matched recipe ingredients / number of recipe ingredients`; rank by score and show unmatched ingredients.

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
