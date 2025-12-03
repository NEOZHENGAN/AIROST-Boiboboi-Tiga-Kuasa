import cv2
import numpy as np
from ultralytics import YOLO
import time
import pandas as pd
from datetime import datetime
import pickle
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
import matplotlib.pyplot as plt
import pyttsx3
import threading
import qrcode 

# ==========================================
#   SETUP: VOICE & TEXT AI
# ==========================================
try:
    engine = pyttsx3.init()
    def speak(text):
        def _run():
            try: engine.say(text); engine.runAndWait()
            except: pass
        threading.Thread(target=_run).start()
except:
    def speak(text): pass

# LOAD NLP BRAIN
nlp_model = None
if os.path.exists("nlp_brain.pkl"):
    with open("nlp_brain.pkl", "rb") as f:
        nlp_model = pickle.load(f)

# ==========================================
#   CONFIG
# ==========================================
MODEL_PATH = "best.pt"      
CONFIDENCE_THRESHOLD = 0.5 
STATIC_BOX_SIZE = 300        

MAX_CAPACITY = {'PLASTIC': 5, 'PAPER': 10, 'GLASS': 5, 'METAL': 5, 'ORGANIC': 8, 'UNKNOWN': 99}
BIN_COLORS = {
    'PLASTIC': ((0, 165, 255), "Orange Bin"), 'PAPER': ((255, 0, 0), "Blue Bin"),   
    'GLASS': ((42, 42, 165), "Brown Bin"), 'METAL': ((0, 165, 255), "Orange Bin"), 
    'ORGANIC': ((0, 255, 0), "Green Bin"), 'UNKNOWN': ((200, 200, 200), "???")
}
RECYCLING_TIPS = {
    'PLASTIC': "CRUSH BOTTLE", 'PAPER': "KEEP DRY", 'GLASS': "DO NOT BREAK",
    'METAL': "EMPTY CONTENTS", 'ORGANIC': "NO PLASTIC", 'UNKNOWN': "CHECK ITEM"
}
CO2_RATES = {'PLASTIC': 0.08, 'PAPER': 0.05, 'GLASS': 0.04, 'METAL': 0.17, 'ORGANIC': 0.02, 'UNKNOWN': 0.0}

# ==========================================
#   LOGIN
# ==========================================
print("[INFO] Starting System...")
root = tk.Tk()
root.withdraw()
operator_name = simpledialog.askstring("Login", "Enter Operator Name:")
if not operator_name: operator_name = "Guest"
print(f"[WELCOME] {operator_name}")
root.destroy()
speak(f"Welcome {operator_name}")

# GLOBAL VARS
stop_program = False
tracking_active = False     
current_label = "Scanning..."
current_conf = 0.0
tracker = None 
start_tracking_request = False
reset_request = False 
item_counts = {} 
session_log = [] 

# LOAD SESSION
if os.path.exists('session_data.pkl'):
    try:
        with open('session_data.pkl', 'rb') as f: item_counts = pickle.load(f)
        print("[INFO] Session counts restored.")
    except: pass

# --- CREATE FOLDERS (ORGANIZATION) ---
folders = ["evidence_locker", "Qr collection", "Graph Analysis"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# ==========================================
#   DRAWING FUNCTIONS
# ==========================================
def draw_glass_box(img, x, y, w, h, color=(0,0,0), alpha=0.5):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def draw_capacity_bars(frame, counts, max_caps):
    h, w, _ = frame.shape
    start_y = 100
    cv2.rectangle(frame, (w-160, 80), (w, 350), (0,0,0), -1)
    cv2.putText(frame, "BIN LEVELS", (w-140, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    i = 0
    for label, max_val in max_caps.items():
        if label == 'UNKNOWN': continue
        current = counts.get(label, 0)
        fill_pct = min(current / max_val, 1.0)
        bar_color = (0, 255, 0)
        if fill_pct > 0.5: bar_color = (0, 165, 255)
        if fill_pct >= 1.0: bar_color = (0, 0, 255)
        x = w - 150
        y = start_y + (i * 40)
        cv2.putText(frame, f"{label[:3]}", (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        cv2.rectangle(frame, (x+40, y), (x+140, y+15), (100,100,100), 1)
        fill_w = int(100 * fill_pct)
        if fill_w > 0: cv2.rectangle(frame, (x+40, y), (x+40+fill_w, y+15), bar_color, -1)
        cv2.putText(frame, f"{current}/{max_val}", (x+45, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        i += 1

def draw_eco_panel(frame, counts):
    total_co2 = sum([counts.get(k,0)*CO2_RATES.get(k,0) for k in counts])
    start_x = 20
    start_y = 110 
    cv2.rectangle(frame, (start_x, start_y), (start_x+160, start_y+80), (0, 100, 0), -1) 
    cv2.putText(frame, "ECO IMPACT", (start_x+35, start_y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, f"{total_co2:.2f} kg", (start_x+30, start_y+55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, "CO2 Saved", (start_x+45, start_y+70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

# --- MOUSE CALLBACK ---
def mouse_callback(event, x, y, flags, param):
    global stop_program, tracking_active, start_tracking_request, reset_request
    img_h, img_w = 480, 640
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if (img_w - 140) <= x <= (img_w - 20) and 20 <= y <= 60: # Ask AI
            root = tk.Tk(); root.withdraw()
            q = simpledialog.askstring("AI Assistant", "Ask about a recycling item:")
            if q and nlp_model:
                try:
                    ans = nlp_model.predict([q])[0]; speak(ans); messagebox.showinfo("AI Says:", ans)
                except: speak("I don't know that item.")
            root.destroy()
        elif 20 <= x <= 140 and (img_h - 60) <= y <= (img_h - 20): stop_program = True # Exit
        elif (img_w//2 - 60) <= x <= (img_w//2 + 60) and (img_h - 60) <= y <= (img_h - 20): reset_request = True # Reset
        elif not tracking_active and (img_w - 140) <= x <= (img_w - 20) and (img_h - 60) <= y <= (img_h - 20): start_tracking_request = True # Capture
        elif tracking_active: tracking_active = False

# --- SETUP ---
try:
    model = YOLO(MODEL_PATH)
    labels = model.names
    for l in labels.values():
        name = l.upper().strip()
        if name not in item_counts: item_counts[name] = 0
except: 
    print("[CRITICAL ERROR] YOLO Model (best.pt) missing!")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened(): cap = cv2.VideoCapture(1)
cv2.namedWindow("Smart Recycling HUD", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Smart Recycling HUD", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Smart Recycling HUD", mouse_callback)

# --- MAIN LOOP ---
while not stop_program:
    ret, frame = cap.read()
    if not ret: continue
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    if reset_request:
        for k in item_counts: item_counts[k] = 0
        session_log = []
        if os.path.exists('session_data.pkl'): os.remove('session_data.pkl')
        speak("System Reset.")
        reset_request = False

    if start_tracking_request:
        x1, y1 = (width-STATIC_BOX_SIZE)//2, (height-STATIC_BOX_SIZE)//2
        limit = MAX_CAPACITY.get(current_label, 99)
        
        if item_counts.get(current_label, 0) >= limit:
            speak(f"{current_label} Bin is Full!")
            overlay = frame.copy(); cv2.rectangle(overlay, (0,0), (width, height), (0,0,255), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            cv2.putText(frame, "BIN FULL!", (width//2 - 100, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
        else:
            # === FIX STARTS HERE ===
            # Use CSRT (More accurate/Sticky) instead of KCF
            try: 
                tracker = cv2.TrackerCSRT_create() # <--- CHANGED THIS
            except: 
                # Fallback for older OpenCV versions
                try: tracker = cv2.legacy.TrackerCSRT_create()
                except: tracker = cv2.TrackerKCF_create() # Worst case fallback
            
            # Initialize
            tracker.init(frame, (x1, y1, STATIC_BOX_SIZE, STATIC_BOX_SIZE))
            tracking_active = True
            
            # Count & Save
            item_counts[current_label] += 1
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            session_log.append({"Timestamp": ts, "Operator": operator_name, "Category": current_label, "Confidence": f"{current_conf:.2f}"})
            cv2.imwrite(f"evidence_locker/{current_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", frame)
            speak(f"{current_label} recorded.")
            # === FIX ENDS HERE ===
            
        start_tracking_request = False

    if tracking_active:
        success, box = tracker.update(frame)
        if success:
            tx, ty, tw, th = [int(v) for v in box]
            color = BIN_COLORS.get(current_label, ((255,255,255), ""))[0]
            cv2.rectangle(frame, (tx, ty), (tx+tw, ty+th), color, 3)
            tip = RECYCLING_TIPS.get(current_label, "")
            frame = draw_glass_box(frame, tx, ty+th+10, tw, 30, (0,0,0), 0.6)
            cv2.putText(frame, tip, (tx+5, ty+th+30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
        else: tracking_active = False
    else:
        x1, y1 = (width-STATIC_BOX_SIZE)//2, (height-STATIC_BOX_SIZE)//2
        roi = frame[y1:y1+STATIC_BOX_SIZE, x1:x1+STATIC_BOX_SIZE]
        detected = False
        if roi.size != 0:
            results = model(roi, verbose=False)
            if results[0].probs:
                idx = results[0].probs.top1; conf = float(results[0].probs.top1conf); lbl = labels[idx].upper().strip()
                if conf > CONFIDENCE_THRESHOLD: current_label = lbl; current_conf = conf; detected = True
                else: current_label = "Scanning..."
            elif len(results[0].boxes) > 0:
                box = results[0].boxes[0]; idx = int(box.cls); conf = float(box.conf); lbl = labels[idx].upper().strip()
                if conf > CONFIDENCE_THRESHOLD: current_label = lbl; current_conf = conf; detected = True
                else: current_label = "Scanning..."
            else: current_label = "Scanning..."
        
        color = BIN_COLORS.get(current_label, ((200,200,200),""))[0] if detected else (255,255,255)
        cv2.rectangle(frame, (x1, y1), (x1+STATIC_BOX_SIZE, y1+STATIC_BOX_SIZE), color, 2)
        if detected: cv2.putText(frame, f"FOUND: {current_label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # UI DRAWING
    frame = draw_glass_box(frame, 20, 20, 160, 80, (0,0,0), 0.4)
    cv2.putText(frame, f"USER: {operator_name.upper()}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    cv2.putText(frame, f"TOTAL: {sum(item_counts.values())}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    frame = draw_glass_box(frame, width-140, 20, 120, 40, (128, 0, 128), 0.6) 
    cv2.putText(frame, "ASK AI", (width-110, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    draw_capacity_bars(frame, item_counts, MAX_CAPACITY)
    draw_eco_panel(frame, item_counts)
    
    # Buttons
    frame = draw_glass_box(frame, 20, height-60, 120, 40, (0,0,150), 0.6) 
    cv2.putText(frame, "EXIT", (55, height-33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cx = width//2
    frame = draw_glass_box(frame, cx-60, height-60, 120, 40, (100,100,100), 0.6) 
    cv2.putText(frame, "RESET", (cx-30, height-33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    if not tracking_active: 
        frame = draw_glass_box(frame, width-140, height-60, 120, 40, (200,100,0), 0.6) 
        cv2.putText(frame, "CAPTURE", (width-125, height-33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Smart Recycling HUD", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

speak("Saving Session. Goodbye.")
cap.release()
cv2.destroyAllWindows()

# ==========================================
#   DEBUGGING SAVE SECTION
# ==========================================
print("-----------------------------------")
print("[INFO] STARTING DATA SAVE...")

# 1. EXCEL SAVE
if session_log:
    f = "Waste_Report_Master.xlsx"
    new = pd.DataFrame(session_log)
    try:
        old = pd.read_excel(f, sheet_name='Detailed_Log')
        combined = pd.concat([old, new], ignore_index=True)
    except: combined = new
    try:
        with pd.ExcelWriter(f, mode='w') as w:
            summary = combined.groupby(['Operator', 'Category']).size().reset_index(name='Total Count')
            summary.to_excel(w, sheet_name='Summary_By_User', index=False)
            combined.to_excel(w, sheet_name='Detailed_Log', index=False)
        print(f"[SUCCESS] Excel saved to {f}")
    except PermissionError:
        print(f"[CRITICAL ERROR] Could not save Excel. File '{f}' is OPEN. Close it!")
    except Exception as e:
        print(f"[ERROR] Excel Save Failed: {e}")
else:
    print("[INFO] No new items to save to Excel.")

# 2. PICKLE SAVE
if sum(item_counts.values()) > 0:
    try:
        with open('session_data.pkl', 'wb') as f: pickle.dump(item_counts, f)
        print("[SUCCESS] Session memory saved.")
    except Exception as e: print(f"[ERROR] Pickle Save Failed: {e}")

# 3. GRAPH SAVE (New Folder)
if sum(item_counts.values()) > 0:
    try:
        active = {k:v for k,v in item_counts.items() if v>0}
        plt.figure(figsize=(10,6)); plt.bar(active.keys(), active.values(), color='blue')
        plt.title(f"Session: {operator_name}"); 
        # Save to Folder
        g_name = f"Graph Analysis/Session_Graph_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
        plt.savefig(g_name); plt.close()
        print(f"[SUCCESS] Graph saved as {g_name}")
    except Exception as e: print(f"[ERROR] Graph Failed: {e}")

# 4. QR RECEIPT (New Folder)
if sum(item_counts.values()) > 0:
    try:
        receipt_text = f"SMART RECYCLING\nUser: {operator_name}\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n-----------------------\n"
        for k,v in item_counts.items():
            if v>0: receipt_text += f"{k}: {v}\n"
        receipt_text += f"-----------------------\nTotal Items: {sum(item_counts.values())}\n"
        total_co2 = sum([item_counts.get(k,0)*CO2_RATES.get(k,0) for k in item_counts])
        receipt_text += f"CO2 Saved: {total_co2:.2f} kg\nThank you!"
        
        qr = qrcode.make(receipt_text)
        
        # New Folder + Filename
        ts = datetime.now().strftime('%Y%m%d_%H%M')
        qr_name = f"Qr collection/{operator_name}_{ts}.png"
        qr.save(qr_name)
        
        qr_img = cv2.imread(qr_name)
        cv2.imshow("Scan Receipt", qr_img)
        cv2.waitKey(5000) 
        print(f"[SUCCESS] QR Receipt saved as {qr_name}")
    except Exception as e: print(f"[ERROR] QR Failed: {e}")

print("-----------------------------------")