"""Script to expand data/processed/recipes.csv with diverse, verified culinary recipes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPES_PATH = ROOT / "data" / "processed" / "recipes.csv"

# Existing 26 base recipes (preserving IDs R001 to R026 for complete backwards-compatibility)
BASE_RECIPES = [
    {
        "id": "R001",
        "name": "Spinach and Tomato Omelet",
        "ingredients": ["egg", "spinach", "tomato", "cheese"],
        "calories_per_serving": 320,
        "prep_time": 15,
        "instructions": [
            "Whisk eggs with a pinch of salt and pepper.",
            "Cook spinach and diced tomato in a lightly oiled skillet until tender.",
            "Pour in the eggs, cook until nearly set, add cheese, fold, and serve."
        ],
        "protein_g": 22,
        "carbs_g": 8,
        "fat_g": 22
    },
    {
        "id": "R002",
        "name": "Cheesy Bell Pepper Frittata",
        "ingredients": ["egg", "bell pepper", "onion", "cheese"],
        "calories_per_serving": 350,
        "prep_time": 25,
        "instructions": [
            "Soften diced bell pepper and onion in an oven-safe skillet.",
            "Beat eggs, pour over the vegetables, and scatter cheese on top.",
            "Cook gently until set, finishing under a broiler if desired."
        ],
        "protein_g": 21,
        "carbs_g": 12,
        "fat_g": 24
    },
    {
        "id": "R003",
        "name": "Tomato Chickpea Stew",
        "ingredients": ["tomato", "chickpea", "onion", "garlic"],
        "calories_per_serving": 390,
        "prep_time": 30,
        "instructions": [
            "Cook chopped onion and garlic in a saucepan until fragrant.",
            "Add tomatoes and drained chickpeas; simmer for 20 minutes.",
            "Season to taste and serve warm."
        ],
        "protein_g": 16,
        "carbs_g": 58,
        "fat_g": 9
    },
    {
        "id": "R004",
        "name": "Chicken Spinach Skillet",
        "ingredients": ["chicken", "spinach", "garlic", "lemon"],
        "calories_per_serving": 410,
        "prep_time": 25,
        "instructions": [
            "Season chicken and cook in a hot skillet until browned and cooked through.",
            "Add garlic and a splash of water; stir in spinach until wilted.",
            "Finish with lemon juice and serve."
        ],
        "protein_g": 42,
        "carbs_g": 12,
        "fat_g": 22
    },
    {
        "id": "R005",
        "name": "Apple Yogurt Breakfast Bowl",
        "ingredients": ["apple", "yogurt", "banana"],
        "calories_per_serving": 285,
        "prep_time": 5,
        "instructions": [
            "Spoon yogurt into a bowl.",
            "Top with chopped apple and sliced banana.",
            "Add cinnamon or a little honey if available."
        ],
        "protein_g": 14,
        "carbs_g": 48,
        "fat_g": 5
    },
    {
        "id": "R006",
        "name": "Roasted Broccoli and Potato Bowl",
        "ingredients": ["broccoli", "potato", "cheese", "lemon"],
        "calories_per_serving": 430,
        "prep_time": 40,
        "instructions": [
            "Heat oven to 220 C / 425 F and cube the potatoes.",
            "Roast potatoes and broccoli with oil until browned and tender.",
            "Finish with cheese and lemon juice."
        ],
        "protein_g": 17,
        "carbs_g": 62,
        "fat_g": 15
    },
    {
        "id": "R007",
        "name": "Chicken and Rice Skillet",
        "ingredients": ["chicken", "rice", "onion", "carrot"],
        "calories_per_serving": 520,
        "prep_time": 35,
        "instructions": [
            "Brown bite-sized chicken pieces in a deep skillet.",
            "Add diced onion, carrot, rice, and broth or water.",
            "Cover and simmer until rice is tender and chicken is fully cooked."
        ],
        "protein_g": 37,
        "carbs_g": 61,
        "fat_g": 13
    },
    {
        "id": "R008",
        "name": "Tomato Mozzarella Salad",
        "ingredients": ["tomato", "cheese", "spinach", "lemon"],
        "calories_per_serving": 260,
        "prep_time": 10,
        "instructions": [
            "Slice tomatoes and tear spinach into bite-size pieces.",
            "Arrange with sliced mozzarella or other cheese.",
            "Dress with lemon juice, olive oil, salt, and pepper."
        ],
        "protein_g": 13,
        "carbs_g": 14,
        "fat_g": 17
    },
    {
        "id": "R009",
        "name": "Black Bean and Bell Pepper Tacos",
        "ingredients": ["black bean", "bell pepper", "onion", "tomato"],
        "calories_per_serving": 390,
        "prep_time": 20,
        "instructions": [
            "Cook sliced bell pepper and onion in a skillet until softened.",
            "Stir in drained black beans and chopped tomato; heat through.",
            "Serve in warm tortillas with your favorite toppings."
        ],
        "protein_g": 17,
        "carbs_g": 65,
        "fat_g": 8
    },
    {
        "id": "R010",
        "name": "Garlic Mushroom Pasta",
        "ingredients": ["pasta", "mushroom", "garlic", "cheese"],
        "calories_per_serving": 510,
        "prep_time": 25,
        "instructions": [
            "Boil pasta according to its package directions and reserve a little pasta water.",
            "Cook sliced mushrooms with garlic until golden.",
            "Toss pasta with mushrooms, cheese, and enough pasta water to coat."
        ],
        "protein_g": 19,
        "carbs_g": 76,
        "fat_g": 16
    },
    {
        "id": "R011",
        "name": "Carrot Apple Slaw",
        "ingredients": ["carrot", "apple", "lemon", "yogurt"],
        "calories_per_serving": 190,
        "prep_time": 15,
        "instructions": [
            "Grate carrots and apple into a bowl.",
            "Mix yogurt with lemon juice, salt, and pepper.",
            "Toss together and chill briefly before serving."
        ],
        "protein_g": 6,
        "carbs_g": 32,
        "fat_g": 5
    },
    {
        "id": "R012",
        "name": "Spinach Mushroom Egg Scramble",
        "ingredients": ["egg", "spinach", "mushroom", "cheese"],
        "calories_per_serving": 305,
        "prep_time": 15,
        "instructions": [
            "Saute sliced mushrooms until their moisture cooks away.",
            "Add spinach and stir until wilted.",
            "Pour in beaten eggs, scramble gently, and finish with cheese."
        ],
        "protein_g": 23,
        "carbs_g": 9,
        "fat_g": 20
    },
    {
        "id": "R013",
        "name": "Salmon with Lemon Potato",
        "ingredients": ["salmon", "potato", "lemon", "broccoli"],
        "calories_per_serving": 560,
        "prep_time": 35,
        "instructions": [
            "Heat oven to 200 C / 400 F and cut potatoes into small pieces.",
            "Roast potatoes until nearly tender, then add seasoned salmon and broccoli.",
            "Bake until salmon flakes easily and finish with lemon."
        ],
        "protein_g": 38,
        "carbs_g": 48,
        "fat_g": 24
    },
    {
        "id": "R014",
        "name": "Avocado Tomato Toast",
        "ingredients": ["avocado", "tomato", "bread", "lemon"],
        "calories_per_serving": 340,
        "prep_time": 10,
        "instructions": [
            "Toast slices of bread until crisp.",
            "Mash avocado with lemon juice, salt, and pepper.",
            "Spread on toast and top with sliced tomato."
        ],
        "protein_g": 9,
        "carbs_g": 42,
        "fat_g": 16
    },
    {
        "id": "R015",
        "name": "Chicken and Broccoli Pasta",
        "ingredients": ["chicken", "broccoli", "pasta", "cheese"],
        "calories_per_serving": 590,
        "prep_time": 30,
        "instructions": [
            "Cook pasta and add broccoli for the final few minutes; drain.",
            "Cook diced chicken in a skillet until fully cooked.",
            "Combine pasta, broccoli, chicken, and cheese with a splash of cooking water."
        ],
        "protein_g": 41,
        "carbs_g": 68,
        "fat_g": 18
    },
    {
        "id": "R016",
        "name": "Chickpea Spinach Curry",
        "ingredients": ["chickpea", "spinach", "tomato", "onion"],
        "calories_per_serving": 420,
        "prep_time": 30,
        "instructions": [
            "Cook chopped onion with curry spices until soft.",
            "Add tomatoes and chickpeas; simmer for 15 minutes.",
            "Stir in spinach until wilted and serve with rice if available."
        ],
        "protein_g": 18,
        "carbs_g": 63,
        "fat_g": 12
    },
    {
        "id": "R017",
        "name": "Cheese and Vegetable Quesadilla",
        "ingredients": ["cheese", "bell pepper", "spinach", "onion"],
        "calories_per_serving": 380,
        "prep_time": 20,
        "instructions": [
            "Cook diced bell pepper and onion until soft, then stir in spinach.",
            "Place vegetables and cheese on half of a tortilla and fold.",
            "Toast in a dry skillet on both sides until crisp and melty."
        ],
        "protein_g": 17,
        "carbs_g": 39,
        "fat_g": 19
    },
    {
        "id": "R018",
        "name": "Potato and Onion Hash with Eggs",
        "ingredients": ["potato", "onion", "egg", "bell pepper"],
        "calories_per_serving": 395,
        "prep_time": 30,
        "instructions": [
            "Dice potatoes small and cook in an oiled skillet until browned and tender.",
            "Add sliced onion and bell pepper; cook until softened.",
            "Serve with eggs cooked to your preference."
        ],
        "protein_g": 18,
        "carbs_g": 47,
        "fat_g": 15
    },
    {
        "id": "R019",
        "name": "Strawberry Banana Yogurt Smoothie",
        "ingredients": ["strawberry", "banana", "yogurt", "milk"],
        "calories_per_serving": 270,
        "prep_time": 5,
        "instructions": [
            "Add strawberries, banana, yogurt, and milk to a blender.",
            "Blend until smooth, adding a little more milk to thin if needed.",
            "Serve immediately."
        ],
        "protein_g": 13,
        "carbs_g": 47,
        "fat_g": 4
    },
    {
        "id": "R020",
        "name": "Cucumber Tomato Yogurt Salad",
        "ingredients": ["cucumber", "tomato", "yogurt", "lemon"],
        "calories_per_serving": 180,
        "prep_time": 10,
        "instructions": [
            "Dice cucumber and tomato.",
            "Stir yogurt with lemon juice, salt, and pepper.",
            "Fold the vegetables into the dressing and serve chilled."
        ],
        "protein_g": 9,
        "carbs_g": 24,
        "fat_g": 5
    },
    {
        "id": "R021",
        "name": "Carrot Broccoli Soup",
        "ingredients": ["carrot", "broccoli", "onion", "milk"],
        "calories_per_serving": 245,
        "prep_time": 35,
        "instructions": [
            "Chop carrot, broccoli, and onion into small pieces.",
            "Simmer vegetables in broth or water until tender.",
            "Blend until smooth, stir in milk, and season to taste."
        ],
        "protein_g": 10,
        "carbs_g": 35,
        "fat_g": 7
    },
    {
        "id": "R022",
        "name": "Chicken Avocado Salad",
        "ingredients": ["chicken", "avocado", "lettuce", "tomato"],
        "calories_per_serving": 430,
        "prep_time": 20,
        "instructions": [
            "Cook chicken until its center reaches a safe temperature, then slice.",
            "Toss lettuce and tomato in a bowl.",
            "Top with chicken and avocado; dress with lemon and olive oil."
        ],
        "protein_g": 36,
        "carbs_g": 20,
        "fat_g": 24
    },
    {
        "id": "R023",
        "name": "Garlic Lemon Salmon",
        "ingredients": ["salmon", "garlic", "lemon", "spinach"],
        "calories_per_serving": 470,
        "prep_time": 25,
        "instructions": [
            "Season salmon with minced garlic, lemon juice, salt, and pepper.",
            "Bake at 200 C / 400 F until the fish flakes easily.",
            "Serve with quickly wilted spinach."
        ],
        "protein_g": 39,
        "carbs_g": 11,
        "fat_g": 29
    },
    {
        "id": "R024",
        "name": "Apple Cheddar Snack Plate",
        "ingredients": ["apple", "cheese", "carrot"],
        "calories_per_serving": 250,
        "prep_time": 8,
        "instructions": [
            "Slice the apple and cheese.",
            "Cut the carrot into sticks.",
            "Arrange together and serve as a snack plate."
        ],
        "protein_g": 10,
        "carbs_g": 31,
        "fat_g": 11
    },
    {
        "id": "R025",
        "name": "Tomato Spinach Pasta",
        "ingredients": ["pasta", "tomato", "spinach", "garlic"],
        "calories_per_serving": 455,
        "prep_time": 25,
        "instructions": [
            "Cook pasta according to package directions.",
            "Warm chopped tomato and garlic in a skillet until saucy.",
            "Fold in spinach until wilted, toss with drained pasta, and season."
        ],
        "protein_g": 16,
        "carbs_g": 73,
        "fat_g": 11
    },
    {
        "id": "R026",
        "name": "Chickpea Carrot Salad",
        "ingredients": ["chickpea", "carrot", "cucumber", "lemon"],
        "calories_per_serving": 330,
        "prep_time": 15,
        "instructions": [
            "Drain chickpeas and grate the carrot.",
            "Chop cucumber and combine with chickpeas and carrot.",
            "Dress with lemon juice, olive oil, salt, and pepper."
        ],
        "protein_g": 15,
        "carbs_g": 52,
        "fat_g": 9
    }
]

# Additional curated recipes expanding diverse real-world culinary combinations
ADDITIONAL_RECIPES = [
    # Eggs & Breakfast
    {"id": "R027", "name": "Classic French Toast", "ingredients": ["bread", "egg", "milk", "butter"], "calories_per_serving": 310, "prep_time": 15, "instructions": ["Whisk egg and milk in a shallow dish.", "Dip bread slices into the egg mixture until coated.", "Melt butter in a pan and cook bread on medium heat until golden brown on both sides."], "protein_g": 12, "carbs_g": 38, "fat_g": 13},
    {"id": "R028", "name": "Shakshuka with Poached Eggs", "ingredients": ["egg", "tomato", "bell pepper", "onion", "garlic"], "calories_per_serving": 340, "prep_time": 25, "instructions": ["Saute onion, bell pepper, and garlic until tender.", "Add tomatoes and cumin; simmer into a thick spiced sauce.", "Create small wells in sauce, crack in eggs, cover, and gently cook until whites are set."], "protein_g": 18, "carbs_g": 22, "fat_g": 18},
    {"id": "R029", "name": "Banana Oatmeal Pancakes", "ingredients": ["banana", "egg", "oats", "milk"], "calories_per_serving": 290, "prep_time": 15, "instructions": ["Blend banana, oats, egg, and milk until smooth.", "Pour batter onto a hot greased griddle.", "Flip once bubbles appear on the surface and cook through."], "protein_g": 11, "carbs_g": 48, "fat_g": 6},
    {"id": "R030", "name": "Avocado Egg Breakfast Toast", "ingredients": ["bread", "avocado", "egg", "lemon"], "calories_per_serving": 360, "prep_time": 12, "instructions": ["Toast bread slices.", "Mash avocado with lemon juice, salt, and pepper.", "Fry or poach eggs to desired doneness.", "Spread avocado on toast and top with eggs."], "protein_g": 14, "carbs_g": 34, "fat_g": 20},
    {"id": "R031", "name": "Cheese and Herb Scrambled Eggs", "ingredients": ["egg", "cheese", "milk", "butter"], "calories_per_serving": 280, "prep_time": 10, "instructions": ["Whisk eggs with a splash of milk and seasoning.", "Melt butter in a non-stick skillet over low heat.", "Gently stir curds until soft and fold in grated cheese just before removing from heat."], "protein_g": 19, "carbs_g": 3, "fat_g": 22},
    {"id": "R032", "name": "Berry Yogurt Parfait", "ingredients": ["yogurt", "strawberry", "blueberry", "honey"], "calories_per_serving": 240, "prep_time": 5, "instructions": ["Layer greek yogurt into a glass.", "Add sliced strawberries and blueberries.", "Drizzle lightly with honey and serve."], "protein_g": 15, "carbs_g": 38, "fat_g": 4},

    # Poultry & Meats
    {"id": "R033", "name": "Lemon Garlic Roasted Chicken", "ingredients": ["chicken", "lemon", "garlic", "olive oil"], "calories_per_serving": 460, "prep_time": 40, "instructions": ["Rub chicken with olive oil, minced garlic, lemon juice, salt, and pepper.", "Roast in preheated 200 C oven until golden and internal temperature reaches 74 C.", "Rest for 5 minutes before slicing."], "protein_g": 44, "carbs_g": 4, "fat_g": 26},
    {"id": "R034", "name": "Chicken Fajitas", "ingredients": ["chicken", "bell pepper", "onion", "tortilla", "lime"], "calories_per_serving": 490, "prep_time": 25, "instructions": ["Slice chicken, bell peppers, and onion into thin strips.", "Stir-fry in hot skillet with fajita seasonings until chicken is cooked through.", "Squeeze fresh lime juice over skillet and serve in warm tortillas."], "protein_g": 40, "carbs_g": 45, "fat_g": 16},
    {"id": "R035", "name": "Garlic Butter Chicken Thighs", "ingredients": ["chicken", "butter", "garlic", "spinach"], "calories_per_serving": 510, "prep_time": 30, "instructions": ["Sear chicken thighs skin-side down in a hot skillet until crispy.", "Flip, baste with melted butter and minced garlic.", "Add spinach to the pan during final minutes to wilt in garlic butter."], "protein_g": 38, "carbs_g": 5, "fat_g": 36},
    {"id": "R036", "name": "Ground Beef Fried Rice", "ingredients": ["beef", "rice", "egg", "onion", "carrot", "soy sauce"], "calories_per_serving": 540, "prep_time": 20, "instructions": ["Brown ground beef and drain excess fat.", "Saute diced onion and carrot in skillet.", "Add cooked rice, push to side, scramble in eggs, and toss with soy sauce."], "protein_g": 34, "carbs_g": 58, "fat_g": 18},
    {"id": "R037", "name": "Classic Beef Bolognese Pasta", "ingredients": ["beef", "pasta", "tomato", "onion", "garlic"], "calories_per_serving": 580, "prep_time": 45, "instructions": ["Brown beef with diced onion and garlic in a Dutch oven.", "Stir in tomatoes and simmer gently on low heat for 30 minutes.", "Toss with al dente cooked pasta and season."], "protein_g": 36, "carbs_g": 68, "fat_g": 18},
    {"id": "R038", "name": "Pork and Broccoli Stir Fry", "ingredients": ["pork", "broccoli", "garlic", "soy sauce", "rice"], "calories_per_serving": 480, "prep_time": 25, "instructions": ["Slice pork thinly and toss with soy sauce and garlic.", "Sear pork in hot wok until browned.", "Add broccoli florets and a splash of water, cover to steam, and serve over rice."], "protein_g": 35, "carbs_g": 52, "fat_g": 14},
    {"id": "R039", "name": "Turkey and Vegetable Skillet", "ingredients": ["turkey", "zucchini", "bell pepper", "onion"], "calories_per_serving": 380, "prep_time": 20, "instructions": ["Brown ground turkey in a large pan.", "Add sliced zucchini, diced bell pepper, and onion.", "Season with herbs and simmer until vegetables are crisp-tender."], "protein_g": 38, "carbs_g": 14, "fat_g": 16},
    {"id": "R040", "name": "Sausage, Pepper and Potato Hash", "ingredients": ["sausage", "bell pepper", "potato", "onion"], "calories_per_serving": 470, "prep_time": 30, "instructions": ["Dice potatoes small and parboil or microwave 3 minutes.", "Slice sausage and brown in skillet.", "Add potatoes, bell pepper, and onion, cooking until edges are crisp and browned."], "protein_g": 22, "carbs_g": 46, "fat_g": 24},

    # Seafood & Fish
    {"id": "R041", "name": "Pan-Seared Salmon with Asparagus", "ingredients": ["salmon", "asparagus", "lemon", "butter"], "calories_per_serving": 490, "prep_time": 20, "instructions": ["Season salmon fillets with salt and black pepper.", "Pan-sear in butter over medium-high heat for 4 minutes per side.", "Add asparagus to the pan in the final 5 minutes, basting with lemon butter."], "protein_g": 41, "carbs_g": 7, "fat_g": 31},
    {"id": "R042", "name": "Garlic Butter Shrimp Pasta", "ingredients": ["shrimp", "pasta", "garlic", "butter", "lemon"], "calories_per_serving": 520, "prep_time": 20, "instructions": ["Cook pasta until al dente.", "Saute peeled shrimp in foaming butter with minced garlic for 3 minutes.", "Toss pasta with shrimp, garlic butter, and fresh lemon juice."], "protein_g": 34, "carbs_g": 64, "fat_g": 15},
    {"id": "R043", "name": "Classic Tuna Salad", "ingredients": ["tuna", "celery", "lemon", "onion"], "calories_per_serving": 240, "prep_time": 10, "instructions": ["Drain canned tuna and flake into a bowl.", "Add finely minced celery, red onion, and squeeze of lemon juice.", "Season with black pepper and serve over salad or bread."], "protein_g": 32, "carbs_g": 6, "fat_g": 8},
    {"id": "R044", "name": "Salmon and Avocado Salad Bowl", "ingredients": ["salmon", "avocado", "cucumber", "lemon", "lettuce"], "calories_per_serving": 460, "prep_time": 15, "instructions": ["Flake cooked salmon into bite-sized chunks.", "Arrange washed lettuce, sliced cucumber, and diced avocado in a bowl.", "Top with salmon and dress with olive oil and lemon juice."], "protein_g": 36, "carbs_g": 14, "fat_g": 28},

    # Soups, Stews & Legumes
    {"id": "R045", "name": "Classic Tomato Basil Soup", "ingredients": ["tomato", "onion", "garlic", "cream"], "calories_per_serving": 260, "prep_time": 30, "instructions": ["Saute chopped onion and garlic in olive oil until soft.", "Add tomatoes and simmer for 20 minutes.", "Blend with an immersion blender and stir in cream."], "protein_g": 6, "carbs_g": 28, "fat_g": 14},
    {"id": "R046", "name": "Loaded Black Bean Soup", "ingredients": ["black bean", "onion", "garlic", "bell pepper", "lime"], "calories_per_serving": 340, "prep_time": 25, "instructions": ["Saute onion, bell pepper, and garlic in a soup pot.", "Add black beans with their liquid and vegetable broth.", "Simmer 15 minutes, mash some beans for thickness, and finish with lime juice."], "protein_g": 18, "carbs_g": 58, "fat_g": 3},
    {"id": "R047", "name": "Hearty Lentil and Carrot Stew", "ingredients": ["lentil", "carrot", "onion", "garlic", "tomato"], "calories_per_serving": 380, "prep_time": 40, "instructions": ["Cook diced onion, carrot, and garlic in olive oil.", "Rinse lentils and add to pot along with crushed tomatoes and broth.", "Simmer for 30 minutes until lentils and carrots are tender."], "protein_g": 20, "carbs_g": 64, "fat_g": 4},
    {"id": "R048", "name": "Potato Leek and Carrot Soup", "ingredients": ["potato", "onion", "carrot", "milk", "butter"], "calories_per_serving": 290, "prep_time": 35, "instructions": ["Sweat diced onion and carrots in melted butter.", "Add cubed potatoes and water or broth; bring to a boil.", "Simmer until tender, blend smooth, and stir in milk."], "protein_g": 8, "carbs_g": 48, "fat_g": 9},

    # Pasta, Grains & Bowls
    {"id": "R049", "name": "Creamy Tomato Penne", "ingredients": ["pasta", "tomato", "cream", "cheese", "garlic"], "calories_per_serving": 540, "prep_time": 20, "instructions": ["Cook penne pasta in salted boiling water.", "Simmer crushed tomatoes with garlic, then stir in heavy cream to make pink sauce.", "Toss pasta with sauce and parmesan cheese."], "protein_g": 16, "carbs_g": 72, "fat_g": 21},
    {"id": "R050", "name": "Vegetable Fried Rice", "ingredients": ["rice", "egg", "carrot", "pea", "onion", "soy sauce"], "calories_per_serving": 410, "prep_time": 15, "instructions": ["Heat wok with oil over high heat.", "Saute diced onion, carrots, and peas.", "Add chilled cooked rice, scramble egg into the side, and stir together with soy sauce."], "protein_g": 14, "carbs_g": 68, "fat_g": 9},
    {"id": "R051", "name": "Mushroom and Spinach Risotto", "ingredients": ["rice", "mushroom", "spinach", "onion", "cheese"], "calories_per_serving": 460, "prep_time": 35, "instructions": ["Saute diced onion and sliced mushrooms in oil until golden.", "Toast arborio rice in pan, then gradually ladle in warm broth while stirring.", "Fold in fresh spinach and grated parmesan when rice is tender."], "protein_g": 14, "carbs_g": 70, "fat_g": 12},
    {"id": "R052", "name": "Mediterranean Quinoa Salad", "ingredients": ["quinoa", "cucumber", "tomato", "lemon", "cheese"], "calories_per_serving": 360, "prep_time": 20, "instructions": ["Cook quinoa according to package instructions and cool.", "Dice cucumber and tomatoes, crumble feta cheese.", "Toss quinoa with vegetables, lemon juice, and olive oil."], "protein_g": 14, "carbs_g": 46, "fat_g": 14},

    # Vegetarian & Produce-Forward
    {"id": "R053", "name": "Stir-Fried Garlic Cabbage and Carrots", "ingredients": ["cabbage", "carrot", "garlic", "soy sauce"], "calories_per_serving": 160, "prep_time": 15, "instructions": ["Shred cabbage and slice carrots into thin matchsticks.", "Heat oil in wok, fry minced garlic until fragrant.", "Add cabbage and carrots, tossing quickly over high heat with soy sauce."], "protein_g": 5, "carbs_g": 24, "fat_g": 6},
    {"id": "R054", "name": "Roasted Zucchini and Bell Pepper Medley", "ingredients": ["zucchini", "bell pepper", "onion", "garlic", "olive oil"], "calories_per_serving": 170, "prep_time": 25, "instructions": ["Chop zucchini, bell peppers, and onion into uniform chunks.", "Toss with olive oil, minced garlic, salt, and pepper.", "Roast at 215 C / 425 F for 20 minutes until caramelized."], "protein_g": 4, "carbs_g": 18, "fat_g": 10},
    {"id": "R055", "name": "Cauliflower and Potato Curry", "ingredients": ["cauliflower", "potato", "tomato", "onion", "garlic"], "calories_per_serving": 310, "prep_time": 30, "instructions": ["Saute onion and garlic with curry spices.", "Add cubed potatoes, cauliflower florets, and tomatoes with a splash of water.", "Cover and simmer for 20 minutes until potatoes are fork-tender."], "protein_g": 9, "carbs_g": 54, "fat_g": 6},
    {"id": "R056", "name": "Creamy Mashed Potatoes with Garlic", "ingredients": ["potato", "milk", "butter", "garlic"], "calories_per_serving": 270, "prep_time": 25, "instructions": ["Peel and cube potatoes, boil in salted water until very tender.", "Drain and mash potatoes thoroughly.", "Fold in warm milk, melted butter, and roasted or minced garlic."], "protein_g": 5, "carbs_g": 42, "fat_g": 10},
    {"id": "R057", "name": "Apple Cinnamon Oatmeal", "ingredients": ["oats", "apple", "milk", "honey"], "calories_per_serving": 290, "prep_time": 10, "instructions": ["Simmer rolled oats in milk over medium heat for 5 minutes.", "Dice apple into small pieces and stir in with cinnamon.", "Drizzle with honey and serve warm."], "protein_g": 10, "carbs_g": 54, "fat_g": 5},
    {"id": "R058", "name": "Crispy Baked Potato Wedges", "ingredients": ["potato", "garlic", "olive oil"], "calories_per_serving": 240, "prep_time": 35, "instructions": ["Cut potatoes lengthwise into wedges.", "Toss with olive oil, garlic powder, salt, and black pepper.", "Bake skin-down at 220 C / 425 F for 30 minutes until golden and crisp."], "protein_g": 4, "carbs_g": 38, "fat_g": 9},
    {"id": "R059", "name": "Simple Tomato Cucumber Bruschetta", "ingredients": ["bread", "tomato", "cucumber", "garlic", "olive oil"], "calories_per_serving": 230, "prep_time": 12, "instructions": ["Toast thick slices of rustic bread and rub with a raw garlic clove.", "Finely dice ripe tomatoes and cucumber.", "Toss vegetables with olive oil, spoon over toast, and sprinkle with salt."], "protein_g": 6, "carbs_g": 32, "fat_g": 9},
    {"id": "R060", "name": "Garlic Sauteed Spinach and Mushrooms", "ingredients": ["spinach", "mushroom", "garlic", "olive oil"], "calories_per_serving": 150, "prep_time": 12, "instructions": ["Slice mushrooms and saute in olive oil until browned.", "Add minced garlic and stir for 30 seconds.", "Toss in fresh spinach, cook until just wilted, and season with salt and pepper."], "protein_g": 7, "carbs_g": 9, "fat_g": 10}
]

ALL_RECIPES = BASE_RECIPES + ADDITIONAL_RECIPES


def generate_recipes_csv():
    fieldnames = [
        "id",
        "name",
        "ingredients",
        "calories_per_serving",
        "prep_time",
        "instructions",
        "protein_g",
        "carbs_g",
        "fat_g"
    ]
    with open(RECIPES_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in ALL_RECIPES:
            writer.writerow({
                "id": r["id"],
                "name": r["name"],
                "ingredients": json.dumps(r["ingredients"]),
                "calories_per_serving": r["calories_per_serving"],
                "prep_time": r["prep_time"],
                "instructions": json.dumps(r["instructions"]),
                "protein_g": r.get("protein_g"),
                "carbs_g": r.get("carbs_g"),
                "fat_g": r.get("fat_g"),
            })
    print(f"Successfully generated {len(ALL_RECIPES)} recipes in {RECIPES_PATH}")


if __name__ == "__main__":
    generate_recipes_csv()
