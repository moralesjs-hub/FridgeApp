# Cloud Vision API & Multimodal LLM Evaluation for FridgeApp
**CSCI 270 Applied AI · College of Charleston**

---

## 1. Executive Summary

FridgeApp requires real-time, fine-grained visual identification of clustered produce, condiments, dairy products, and packaged groceries from user-submitted refrigerator and pantry photos. 

Legacy classification architectures (such as MobileNetV3 ImageNet classifiers) suffer from three critical bottlenecks:
1. **Single-label classification limitation**: Standard classifiers identify the dominant object in an image rather than enumerating multi-item scenes with spatial localization.
2. **Taxonomic mismatch**: ImageNet's 1,000 categories lack refrigerator-specific food items, produce varieties, and culinary vocabulary.
3. **Absence of spatial reasoning**: Clustered items on refrigerator shelves cannot be differentiated without 2D bounding boxes.

This evaluation compares traditional Cloud Computer Vision APIs against modern Multimodal Large Language Models (MLLMs) to select the production-grade vision engine for FridgeApp.

---

## 2. Comprehensive Provider Comparison Matrix

| Evaluation Dimension | Google Cloud Vision API | AWS Rekognition | Azure AI Vision | OpenAI GPT-4o-mini | Anthropic Claude 3.5 Sonnet | Google Gemini (3 Flash / 3.1 Flash-Lite) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model Type** | Traditional CV / DetectLabels | Traditional CV / DetectLabels | Traditional CV / Object Detection | Multimodal LLM | Multimodal LLM | Multimodal LLM |
| **Food & Produce Vocabulary** | Moderate (generic labels like "Fruit", "Vegetable") | Moderate (coarse generic tags, lacks regional culinary items) | Moderate (coarse taxonomy) | Extensive (recognizes brands, cuts, produce varieties) | Extensive (exceptional contextual reasoning) | **Extensive** (fine-grained produce, brands, container detection) |
| **Object Localization (2D Bounding Boxes)** | Native normalized bounding polygons | Native bounding boxes | Native bounding boxes | Requires complex prompt-based coordinate prediction | Inconsistent bounding coordinates | **Native 2D coordinate grounding (`box_2d`)** with structured JSON output |
| **Structured Output Enforcement** | Fixed protobuf / JSON response | Fixed JSON schema | Fixed JSON schema | JSON Schema mode (`json_schema`) | Tool use / schema calling | **Pydantic / Strict JSON Schema validation** (`response_schema`) |
| **Cost per 1,000 Images** | \$1.50 (Label) + \$1.50 (Object) = **\$3.00** | \$1.00 (DetectLabels) | \$1.00 (Analyze Image) | **\$0.30 - \$0.60** (assuming ~800 tokens/img) | **\$4.80 - \$6.00** | **\$0.10 - \$0.25** (Free tier available; lowest cost per token) |
| **Average Latency (p50 / p95)** | ~450ms / ~900ms | ~400ms / ~850ms | ~500ms / ~1100ms | ~1200ms / ~2500ms | ~1500ms / ~3200ms | **~600ms / ~1400ms** |
| **Max Payload Size** | 20 MB | 15 MB | 20 MB | 20 MB | 25 MB | **20 MB (Direct image stream)** |
| **Rate Limits (RPM)** | 1,800 RPM | 50 TPS default | 10 TPS default | Tier-dependent (500 - 10k RPM) | Tier-dependent (50 - 2k RPM) | **15 - 1,000 RPM** |
| **Clustered Occlusion Handling** | Poor (merges overlapping items into single bounding box) | Poor (struggles with translucent packaging & partial occlusions) | Poor (drops occluded items) | Good (deduces partially hidden items from context) | Excellent (strong culinary deduction) | **Excellent** (identifies occluded items and dense produce bins) |

---

## 3. Deep-Dive Provider Analysis

### 3.1 Traditional Cloud Vision APIs (Google Cloud Vision, AWS Rekognition, Azure AI Vision)
- **Strengths**: Low latency (<500ms), established enterprise SDKs, predictable pricing.
- **Weaknesses**: 
  - They operate on generic pre-trained classification heads. For example, a bunch of scallions is frequently labeled simply as `"Plant"`, `"Flora"`, or `"Vegetable"`, which fails downstream recipe matching.
  - Bounding box accuracy degrades significantly when bottles, jars, and produce bins overlap on dense shelves.
  - No contextual understanding: cannot distinguish whether a carton contains almond milk or cow's milk from packaging text cues without multi-stage OCR pipelines.

### 3.2 OpenAI GPT-4o / GPT-4o-mini
- **Strengths**: High semantic comprehension, excellent brand and produce recognition, strict JSON Schema support via Structured Outputs.
- **Weaknesses**: Bounding box extraction requires prompt engineering rather than native grounding, resulting in occasional coordinate hallucinations. Higher latency and cost compared to lightweight flash models.

### 3.3 Anthropic Claude 3.5 Sonnet
- **Strengths**: Best-in-class culinary reasoning, ability to estimate freshness and recipe suitability directly from imagery.
- **Weaknesses**: Highest pricing tier (~$3.00/MTok input + $15.00/MTok output), higher latency (1.5s - 3s), lacks native spatial coordinate grounding parameters in base vision calls.

### 3.4 Google Gemini (Gemini 3 Flash & 3.1 Flash-Lite via Google GenAI SDK)
- **Strengths**:
  1. **Native Spatial Grounding**: Supports 2D normalized bounding coordinates (`[ymin, xmin, ymax, xmax]`) directly aligned with identified entities.
  2. **Strict Schema Enforcement**: Pydantic models directly compile to `GenerateContentConfig.response_schema` with guaranteed JSON serialization.
  3. **Low Latency & High Throughput**: Sub-second execution times with optimized image tokens.
  4. **Domain-Specific Produce Recognition**: Correctly distinguishes specific vegetables (e.g., Roma tomato vs cherry tomato, bell peppers vs chili peppers, whole milk vs cream) and reads container typography effortlessly.
  5. **Cost-Effectiveness**: Generous free tier and lowest production cost per visual query.

---

## 4. Final Architecture Decision & Selection Rationale

**Selected Provider**: **Google Gemini 3 Flash / 3.1 Flash-Lite (via `google-genai` SDK)**.

### Rationale:
1. **Precision & Grounding**: Enables FridgeApp to draw real-time visual bounding boxes directly on the user's uploaded fridge photo, providing transparent visual confirmation of what was detected.
2. **Integrated Entity Extraction**: Outputs both the detected label and the canonical normalized ingredient in a single cloud API round-trip, directly matching FridgeApp's curated recipe database.
3. **Resilience & Fallbacks**: The backend pipeline includes exponential backoff retry mechanisms, automated image compression, and graceful diagnostics in the event of API quota spikes or network latency.
