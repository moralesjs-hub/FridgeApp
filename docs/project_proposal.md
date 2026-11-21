# Project Proposal: AI Food Recognition and Recipe Suggestions

**Course:** CSCI 270 Applied AI, College of Charleston  
**Team Members:** Philip Pagana, Snovi Patel, Jaden Morales, Amelia West

## Problem Statement

People often have ingredients at home but don't know what meals they can make with them. Our goal is to create an application that uses AI to recognize food ingredients from images and recommend recipes based on what is available.

One challenge is that image recognition is not always accurate, especially when multiple ingredients are in the same picture. To handle this, we allow users to review and correct the detected ingredients before getting recipe recommendations.

## Proposed Solution

Our project, Ingredia, is a web application built using Streamlit. Users can upload an image of their ingredients, and the app will try to identify them using AI.

We use Google Gemini for ingredient recognition when available, along with two local pretrained models: Faster R-CNN for object detection and MobileNetV3-Small for image classification. The local models are used as a fallback when Gemini is unavailable.

Once the ingredients are identified, the app compares them with our recipe dataset and recommends meals based on what the user has. It also shows missing ingredients, cooking instructions, and estimated nutrition information.

## Data

We use a CSV dataset containing 120 recipes, including ingredients, preparation time, instructions, and nutrition information.

For image recognition, we use pretrained models with ImageNet and COCO weights, along with sample food images for testing. We also included training scripts that support food-specific image datasets.

Ingredient names are cleaned and standardized so the recipe engine can match them correctly.

## AI and Ranking Approach

Our system first processes the uploaded image and identifies possible ingredients. Users can then make changes to the ingredient list if needed.

The recipe engine compares the ingredients with the recipes in our dataset and calculates match scores. Exact ingredient matches receive full credit, while supported substitutions receive partial credit. Common pantry ingredients are also considered.

Recipes with higher match scores are shown first, along with any missing ingredients. This helps users find meals they can make with what they already have.

## Success Criteria

We consider our project successful if:

- Users can upload images and receive ingredient predictions.
- The app allows users to correct detected ingredients.
- The recipe engine recommends relevant recipes based on available ingredients.
- Recipes display match scores, missing ingredients, and nutrition information.
- The application runs correctly and handles errors.

We will test the application using sample food images and automated tests to make sure the main features work properly.

## Timeline

| Phase | Tasks |
|---|---|
| Planning | Choose the project idea and divide responsibilities |
| Research and Design | Research AI models and plan the application |
| Development | Prepare recipe data and build the AI and recipe matching systems |
| Integration | Combine everyone's work into the Streamlit application |
| Testing and Finalization | Test the features, fix issues, and prepare the final submission |
