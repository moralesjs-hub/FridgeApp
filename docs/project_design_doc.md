# Software Design Document (SDD)

**Project Title:** Ingredia AI Food Recognition and Recipe Recommendation App  
**Team Members:** Snovi Patel, Jaden Morales, Philip Pagana, Amelia West  
**Date:** November 2026

## 1. Statement of Purpose

The goal of our project is to develop an application that can recognize food ingredients from images and recommend recipes based on what the user already has.

We chose this idea because people often have ingredients at home but are not sure what to cook with them. Instead of manually searching for recipes, users can upload a picture of their ingredients and receive recipe recommendations.

Our project combines AI image recognition with a recipe matching system. We wanted to see how well computer vision models could recognize different ingredients and how those results could be used to recommend meals.

Since image recognition is not always accurate, we also designed the application to allow users to review and correct the detected ingredients before finding recipes.

## 2. Data to Be Used for Training

Our project uses two main types of data: food images for ingredient recognition and a recipe dataset for generating recommendations.

For image recognition, we use pretrained models from torchvision. MobileNetV3-Small uses ImageNet pretrained weights for image classification, while Faster R-CNN uses COCO pretrained weights for object detection.

We also included training scripts that support food-specific image datasets, including the [Scuccorese Food Ingredients Dataset](https://huggingface.co/datasets/Scuccorese/food-ingredients-dataset) from Hugging Face.

The image data consists of labeled food images. For object detection, bounding-box annotations can also be used to identify where ingredients appear in an image. The training pipeline supports splitting the data into training and validation sets.

For recipe recommendations, we use a CSV dataset stored in `data/processed/recipes.csv`. The current dataset contains 120 recipes, including recipe names, ingredients, preparation times, instructions, calories, and nutritional information.

The recipe dataset is not used to train the image recognition models. Instead, it is used by the recipe matching algorithm to compare available ingredients with the ingredients required for each recipe.

Before processing the data, ingredient names are normalized so different names for the same ingredient can be recognized consistently. Images are also resized and transformed into the format required by the recognition models.

## 3. AI Tools or Packages to Be Used

We use Python as the main programming language for our project.

The main tools and libraries we use are:

- **PyTorch and torchvision:** For running pretrained image classification and object detection models.
- **Google Gemini API:** For identifying ingredients from uploaded images when an API key is configured.
- **Pillow:** For opening and processing images.
- **NumPy:** For image and numerical processing.
- **Pandas:** For reading and managing the recipe dataset.
- **Streamlit:** For building the application interface.
- **pytest:** For testing different parts of the application.
- **GitHub:** For version control and combining our team's work.

We chose pretrained models because training an image recognition model from scratch would require a large amount of labeled data and computing resources.

Using pretrained models allowed us to focus on integrating image recognition with the recipe recommendation system and testing how the different components work together.

## 4. Process / Workflow Overview

Our project follows a process that starts with preparing the data, recognizing ingredients, and matching them with recipes.

### Data Collection and Cleaning

We organized our recipe data into a CSV file and used pretrained computer vision models for image recognition.

We also included scripts for loading and preparing food-specific image datasets for training.

Ingredient names are cleaned and normalized so the recipe engine can handle common aliases and compare ingredients more accurately.

### Data Modeling

Our local image recognition system uses two pretrained models.

MobileNetV3-Small is used for image classification, while Faster R-CNN is used for object detection. Faster R-CNN can identify supported objects and their locations in an image, while MobileNetV3 helps classify possible ingredients.

We also integrated Google Gemini for ingredient recognition. When Gemini is configured, the application attempts to use it first. If it is unavailable or does not return usable results, the application uses the local recognition models.

For recipe recommendations, we use an ingredient matching algorithm rather than training another machine learning model.

The algorithm compares the ingredients available to the user with the ingredients required for each recipe. It considers exact ingredient matches, supported substitutions, and common pantry staples.

Recipes are then ranked based on their match scores.

### Training

The local image recognition models use pretrained weights.

We also included training scripts that support fine-tuning classification and object detection models using food-specific images.

The training process supports dividing labeled images into training and validation sets. The validation data can be used to check model performance during training.

The current project files do not include custom-trained model checkpoints, so the application uses pretrained models unless a trained checkpoint is provided.

### Predictions

When a user uploads an image, the recognition system attempts to identify the food ingredients shown.

The detected ingredients are normalized and displayed to the user. If any ingredients are missing or incorrect, the user can make changes before continuing.

The recipe engine then compares the final ingredient list with the recipe dataset and calculates recipe match scores.

### Output and Results

The application displays a ranked list of recommended recipes based on the available ingredients.

The results include recipe match scores, missing ingredients, preparation instructions, and estimated nutrition information.

Users can review the recommendations and choose which recipes they want to make.

### Overall System Workflow

The following workflow shows how the application processes an image and generates recipe recommendations.

```text
User Uploads Food Image
          |
          v
    Image Processing
          |
          v
   Ingredient Recognition
          |
          +-------------------------+
          |                         |
          v                         v
    Google Gemini              Local AI Models
    (if configured)            (fallback)
                                    |
                              +-----+-----+
                              |           |
                              v           v
                         Faster R-CNN  MobileNetV3
                              |           |
                              +-----+-----+
                                    |
          +-------------------------+
          |
          v
    Detected Ingredients
          |
          v
  Ingredient Normalization
          |
          v
  User Reviews and Edits
          |
          v
    Recipe Matching
          |
          v
   Recipe Match Scores
          |
          v
   Ranked Recipe Results
          |
          v
  Display Recipes, Missing
  Ingredients, and Nutrition
```

This workflow shows how the image recognition system connects with the recipe recommendation system. After identifying possible ingredients, the application processes the results and allows the user to make corrections. The final ingredient list is then used to calculate recipe matches and display recommendations.

## 5. Expected Output

The expected output of our project is an application that can recognize possible food ingredients from an uploaded image and recommend recipes using those ingredients.

For example, if a user uploads an image containing bananas and strawberries, the system should attempt to identify those ingredients and recommend recipes from the dataset that use them.

The application should display the detected ingredients, recommended recipes, match scores, missing ingredients, and estimated nutritional information.

We measure the success of our project by checking whether:

1. The application can process uploaded images without errors.
2. The image recognition system identifies ingredients correctly in test images.
3. The recipe engine recommends recipes that match the available ingredients.
4. The application handles missing or incorrect ingredient detections.
5. The user can easily review the results and interact with the application.

We use automated tests and manual testing to check the application.

During testing, we noticed that some ingredients were recognized more accurately than others. Images containing multiple items or similar-looking foods could produce incorrect predictions.

Because we have not established a complete benchmark for the final combined recognition system, we are not reporting a numerical accuracy percentage.

## 6. Final Envisioned System of Deployment

Our final product is a web application developed using Streamlit.

The application runs locally through Python and opens in a web browser. Users can upload images, review detected ingredients, and receive recipe recommendations through the interface.

We chose Streamlit because it allowed us to connect our Python-based AI models with an interactive application without having to develop a separate frontend framework.

The image recognition system, recipe engine, and interface are organized into separate Python modules so that the different components can work together.

Our code is maintained on GitHub, where each team member worked on different parts of the project before combining everything into the main application.

The application can run using local pretrained models without requiring a Gemini API key. When Gemini is configured, it can also be used for ingredient recognition.

For our course project, we are demonstrating the application locally rather than deploying it to a public server.

## 7. Prototype (Minimum Viable Product - MVP)

Our original MVP was a basic application that could recognize ingredients from an uploaded image and recommend recipes based on those ingredients.

The main functionality of the MVP included:

1. Allowing users to upload a food image.
2. Using an AI model to identify possible ingredients.
3. Comparing the detected ingredients with a recipe dataset.
4. Displaying recipes that match the available ingredients.

We tested the prototype using sample food images and manually entered ingredients.

We checked whether the recognition models could return possible ingredients and whether the recipe engine could generate recommendations from them.

We also tested the application with different images to see how the results changed and identify problems with ingredient recognition.

As we continued developing the project, we improved the interface, integrated Gemini recognition, added more recipe-related features, and combined the different components into one application.

The MVP demonstrates the main purpose of our project, which is using AI to recognize ingredients and help users find meals they can make with the food they already have.
