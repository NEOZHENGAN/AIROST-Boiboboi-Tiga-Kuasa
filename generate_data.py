import pandas as pd
import random

# ==========================================
#   1. DEFINE YOUR RAW DATA
# ==========================================

# BROWN BIN (Recyclable Glass Only)
# Bottles and jars are okay.
brown_items = [
    "glass bottle", "wine bottle", "beer bottle", "jam jar", "pickle jar",
    "perfume bottle", "glass container", "mason jar", "spaghetti sauce jar",
    "glass food jar", "olive oil bottle", "vinegar bottle", "glass cosmetics bottle",
    "broken glass jar", "glass jug", "ketchup bottle", "baby food jar", "soda bottle glass",
    "sauce bottle", "medicine bottle glass"
]

# ORANGE BIN (Plastic & Metal)
orange_items = [
    "plastic bottle", "water bottle", "soda bottle", "coke can", "aluminum can",
    "soup can", "tin can", "shampoo bottle", "detergent bottle", "yogurt cup",
    "milk jug", "aerosol can", "tuna can", "plastic container", "food can",
    "bleach bottle", "mouthwash bottle", "plastic cup", "beverage can",
    "hairspray can", "deodorant can", "plastic tray", "biscuit tin", "metal lid",
    "plastic take out box", "vitamin bottle plastic"
]

# BLUE BIN (Paper & Books)
blue_items = [
    "newspaper", "magazine", "cardboard box", "cereal box", "envelope",
    "office paper", "notebook", "flyer", "brochure", "paper bag",
    "junk mail", "shipping box", "shoe box", "egg carton", "paper roll",
    "wrapping paper", "calendar", "post-it notes", "paper folder",
    "milk carton", "juice box",
    # --- BOOKS (User Request) ---
    "storybook", "story book", "children book", "comic book", 
    "textbook", "phone book", "activity book", "coloring book", "novel",
    "paperback book"
]

# TRASH (Cannot Recycle)
# Includes dirty paper, soft plastic, and NON-RECYCLABLE GLASS
trash_items = [
    "pizza box", "tissue", "used napkin", "paper towel", "plastic straw",
    "plastic bag", "chip bag", "candy wrapper", "food waste", "banana peel",
    "apple core", "diaper", "battery", "face mask", "styrofoam",
    "broken glass", "mirror", "ceramic mug", "light bulb", "toothpaste tube",
    "bubble wrap", "cling wrap", "dirty foil", "cigarette butt", "wet wipe",
    "packing peanuts", "rubber glove", "receipt",
    # --- TRICKY GLASS ITEMS (Melting point issues) ---
    "drinking glass", "glass cup", "wine glass", "tumbler", "kitchen glass",
    "eyeglasses", "spectacles", "sunglasses", "reading glasses",
    "pyrex", "baking dish", "crystal glass", "window glass", "car windshield"
]

# SENTENCE PATTERNS (To make the AI understand different questions)
patterns = [
    "{}",  
    "can i recycle {}",
    "how to recycle {}",
    "where to put {}",
    "is {} recyclable",
    "bin for {}",
    "throw away {}",
    "what to do with {}",
    "i have a {}",
    "recycle {}",
    "is {} trash",
    "can i throw {}"
]

# ==========================================
#   2. GENERATE THE DATASET
# ==========================================
data = []

def add_data(items, answer):
    for item in items:
        for pattern in patterns:
            question = pattern.format(item)
            data.append({"Question": question, "Answer": answer})

# Generate Standard Lists
add_data(brown_items, "Brown bin and YES can recycle.")
add_data(orange_items, "Orange bin and YES can recycle.")
add_data(blue_items, "Blue bin and YES can recycle.")
add_data(trash_items, "Cannot recycle.")

# --- SPECIAL RULE FOR HARDCOVER BOOKS ---
hardcover_items = ["hardcover book", "hard cover book", "thick book", "encyclopedia"]
add_data(hardcover_items, "Blue bin. Remove hard cover first.")

# ==========================================
#   3. SAVE TO EXCEL
# ==========================================
df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True) # Shuffle the rows

print(f"Generated {len(df)} training examples!")
df.to_excel("training_data.xlsx", index=False)
print("Saved to 'training_data.xlsx'. Ready to train!")