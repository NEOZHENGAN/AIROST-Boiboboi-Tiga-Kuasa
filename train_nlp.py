import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import os

print("------------------------------------------------")
print("[INFO] STARTING AI TRAINING SESSION")
print("------------------------------------------------")

# 1. Load the Excel File
if os.path.exists("training_data.xlsx"):
    try:
        # Load data
        df = pd.read_excel("training_data.xlsx")
        
        # === DATA CLEANING (Crucial for Excel) ===
        # 1. Drop rows that have ANY empty cells
        df = df.dropna()
        
        # 2. Ensure all data is treated as text (String)
        df['Question'] = df['Question'].astype(str)
        df['Answer'] = df['Answer'].astype(str)
        
        print(f"[INFO] Success! Loaded {len(df)} valid examples.")
    except Exception as e:
        print(f"[CRITICAL ERROR] Could not read Excel file: {e}")
        exit()
else:
    print("[ERROR] File 'training_data.xlsx' not found.")
    exit()

# 2. Build the AI Pipeline
# (CountVectorizer turns words into numbers -> Naive Bayes predicts category)
model = make_pipeline(CountVectorizer(), MultinomialNB())

# 3. Train the Model
print("[INFO] Training the AI model...")
try:
    model.fit(df['Question'], df['Answer'])
    print("[INFO] Training Complete.")
except Exception as e:
    print(f"[ERROR] Training failed: {e}")
    exit()

# 4. Save the Brain
with open("nlp_brain.pkl", "wb") as f:
    pickle.dump(model, f)

print("------------------------------------------------")
print("[SUCCESS] New brain saved as 'nlp_brain.pkl'")
print("------------------------------------------------")

# 5. Quick Test
test_q = "can i recycle pizza box"
print(f"Test Question: '{test_q}'")
try:
    print(f"AI Prediction: {model.predict([test_q])[0]}")
except:
    print("Test failed (Brain might be empty)")