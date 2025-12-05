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
BIN_MAPPING = {
    'GREEN-GLASS': 'GLASS',
    'BROWN-GLASS': 'GLASS',
    'WHITE-GLASS': 'GLASS',
    'BIOLOGICAL':  'ORGANIC',
    'CARDBOARD':   'PAPER'
}
MAX_CAPACITY = {'PLASTIC': 8, 'PAPER': 10, 'GLASS': 5, 'METAL': 5, 'ORGANIC': 8, 'UNKNOWN': 99}
BIN_COLORS = {
    'PLASTIC': ((0, 165, 255), "Orange Bin"), 'PAPER': ((255, 0, 0), "Blue Bin"),   
    'GLASS': ((42, 42, 165), "Brown Bin"), 'METAL': ((0, 165, 255), "Orange Bin"), 
    'ORGANIC': ((0, 255, 0), "Green Bin"), 'UNKNOWN': ((200, 200, 200), "???")
}
RECYCLING_TIPS = {
    'PLASTIC': "Recycling Tips:CRUSH BOTTLE", 'PAPER': "Recycling Tips:KEEP DRY", 'GLASS': "Recycling Tips:DO NOT BREAK",
    'METAL': "Recycling Tips:EMPTY CONTENTS", 'ORGANIC': "Recycling Tips:NO PLASTIC", 'UNKNOWN': "Recycling Tips:CHECK THE ITEM"
}
CO2_RATES = {'PLASTIC': 0.08, 'PAPER': 0.05, 'GLASS': 0.04, 'METAL': 0.17, 'ORGANIC': 0.02, 'UNKNOWN': 0.0}

# ==========================================
#   LOGIN
# ==========================================
print("[INFO] Starting System...")
root = tk.Tk() #the main window but hidden not in use
root.withdraw() #hide the main window only show dialog
operator_name = simpledialog.askstring("Login", "Enter Operator Name:")
if not operator_name: operator_name = "Guest"
print(f"[WELCOME] {operator_name}")
root.destroy() #close the thinker window
speak(f"Welcome {operator_name}") #Greet the operator

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
        with open('session_data.pkl', 'rb') as f: item_counts = pickle.load(f) #Load pevious counts
        print("[INFO] Session counts restored.")
    except: pass

# --- CREATE FOLDERS (ORGANIZATION) ---
folders = ["evidence_locker", "Qr collection", "Graph Analysis"]
for folder in folders:
    if not os.path.exists(folder): #check if folder exists
        os.makedirs(folder) #Create the folder if does not exist

# ==========================================
#   DRAWING FUNCTIONS
# ==========================================
def draw_glass_box(img, x, y, w, h, color=(0,0,0), alpha=0.5): #define the box function appear transparent 
    overlay = img.copy() #copy image
    cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1) #filled rectangle
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0) #mixex overlay and img

def draw_capacity_bars(frame, counts, max_caps):
    h, w, _ = frame.shape #get frame dimensions
    start_y = 100 #setting start y position
    cv2.rectangle(frame, (w-160, 80), (w, 350), (0,0,0), -1)
    cv2.putText(frame, "BIN LEVELS", (w-140, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    i = 0
    for label, max_val in max_caps.items():
        if label == 'UNKNOWN': continue #skip the unkown bin
        current = counts.get(label, 0) #get current count from the counts dict
        fill_pct = min(current / max_val, 1.0) #calculate fill percentage
        bar_color = (0, 255, 0) #green 
        if fill_pct > 0.5: bar_color = (0, 165, 255) #orange
        if fill_pct >= 1.0: bar_color = (0, 0, 255) #red
        x = w - 150 #x position
        y = start_y + (i * 40)
        cv2.putText(frame, f"{label[:3]}", (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1) #draw bin label
        cv2.rectangle(frame, (x+40, y), (x+140, y+15), (100,100,100), 1) #draw empty bar
        fill_w = int(100 * fill_pct) #calculate fill width
        if fill_w > 0: cv2.rectangle(frame, (x+40, y), (x+40+fill_w, y+15), bar_color, -1) #draw filled bar
        cv2.putText(frame, f"{current}/{max_val}", (x+45, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        i += 1

def draw_eco_panel(frame, counts):
    total_co2 = sum([counts.get(k,0)*CO2_RATES.get(k,0) for k in counts]) #calculate total co2 saved
    start_x = 20
    start_y = 110 
    cv2.rectangle(frame, (start_x, start_y), (start_x+160, start_y+80), (0, 100, 0), -1) 
    cv2.putText(frame, "ECO IMPACT", (start_x+35, start_y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, f"{total_co2:.2f} kg", (start_x+30, start_y+55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, "CO2 Saved", (start_x+45, start_y+70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1) #draw label

# --- MOUSE CALLBACK ---
def mouse_callback(event, x, y, flags, param):
    global stop_program, tracking_active, start_tracking_request, reset_request #import global variables
    img_h, img_w = 480, 640 #default webcam size
    
    if event == cv2.EVENT_LBUTTONDOWN: #if left clicked
        if (img_w - 140) <= x <= (img_w - 20) and 20 <= y <= 60: # if the rectangle area is clicked
            root = tk.Tk(); root.withdraw() #hide main window
            q = simpledialog.askstring("AI Assistant", "Ask about a recycling item:")
            if q and nlp_model: #both are true
                try:
                    ans = nlp_model.predict([q])[0]; speak(ans); messagebox.showinfo("AI Says:", ans) #show answer
                except: speak("I don't know that item.")
            root.destroy() #close the thinker window
        elif 20 <= x <= 140 and (img_h - 60) <= y <= (img_h - 20): stop_program = True # Exit
        elif (img_w//2 - 60) <= x <= (img_w//2 + 60) and (img_h - 60) <= y <= (img_h - 20): reset_request = True # Reset
        elif not tracking_active and (img_w - 140) <= x <= (img_w - 20) and (img_h - 60) <= y <= (img_h - 20): start_tracking_request = True # Capture
        elif tracking_active: tracking_active = False

# --- SETUP ---
try:
    model = YOLO(MODEL_PATH) #load the best.pt model
    labels = model.names #get the labels
    for l in labels.values():
        name = l.upper().strip()
        if name not in item_counts: item_counts[name] = 0 #if label not in counts dict, add it with 0 count
except: 
    print("[CRITICAL ERROR] YOLO Model (best.pt) missing!") #cannot proceed without model
    exit()

cap = cv2.VideoCapture(0) #open webcam 0 is laptop webcam
if not cap.isOpened(): cap = cv2.VideoCapture(1) #if not opened try webcam 1
cv2.namedWindow("Smart Recycling HUD", cv2.WINDOW_NORMAL) #create window
cv2.setWindowProperty("Smart Recycling HUD", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) #edit fullscreen prop,force fullscreen
cv2.setMouseCallback("Smart Recycling HUD", mouse_callback) #when click call back mouse_callback funct

# --- MAIN LOOP ---
while not stop_program: #loop until stop_program is True
    ret, frame = cap.read() #read frame from webcam
    if not ret: continue
    frame = cv2.flip(frame, 1) #flip frame horizontally to get mirror effect
    height, width, _ = frame.shape #get frame dimensions

    if reset_request: #reset mode
        for k in item_counts: item_counts[k] = 0 #all reset counts to 0
        session_log = []
        if os.path.exists('session_data.pkl'): os.remove('session_data.pkl') #delete previous session file
        speak("System Reset.")
        reset_request = False #set back to false

    if start_tracking_request: #capture mode
        # 1. Determine the General Category (e.g. Green-Glass -> GLASS)
        # If the label isn't in the map, use the label itself (e.g. PLASTIC -> PLASTIC)
        bin_category = BIN_MAPPING.get(current_label, current_label)
        
        # 2. Check Capacity of the General Bin
        limit = MAX_CAPACITY.get(bin_category, 99)
        current_count = item_counts.get(bin_category, 0)
        
        if current_count >= limit:
            speak(f"Error. {bin_category} Bin is Full!")
            overlay = frame.copy(); cv2.rectangle(overlay, (0,0), (width, height), (0,0,255), -1) #red overlay to represent full bin
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            cv2.putText(frame, "BIN FULL!", (width//2 - 100, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
        else:
            x1, y1 = (width-STATIC_BOX_SIZE)//2, (height-STATIC_BOX_SIZE)//2 #center box coordinates
            try: tracker = cv2.TrackerCSRT_create() #new tracker object
            except: 
                try: tracker = cv2.legacy.TrackerCSRT_create()
                except: tracker = cv2.TrackerKCF_create()
            tracker.init(frame, (x1, y1, STATIC_BOX_SIZE, STATIC_BOX_SIZE)) 
            tracking_active = True
            
            # 3. INCREMENT THE GENERAL BIN
            item_counts[bin_category] = item_counts.get(bin_category, 0) + 1 #increment count
            
            # 4. LOG THE SPECIFIC ITEM (Keep detail for Excel)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S") #timestamp
            # We log 'current_label' (Green Glass) so we know exactly what it was
            session_log.append({"Timestamp": ts, "Operator": operator_name, "Category": current_label, "Confidence": f"{current_conf:.2f}"})
            
            cv2.imwrite(f"evidence_locker/{current_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", frame)
            speak(f"{current_label} recorded.")
            
        start_tracking_request = False
        
    if tracking_active: #tracking mode
        success, box = tracker.update(frame) #update tracker frame
        if success:
            tx, ty, tw, th = [int(v) for v in box] #get box coordinates
            color = BIN_COLORS.get(current_label, ((255,255,255), ""))[0] #get color for box
            cv2.rectangle(frame, (tx, ty), (tx+tw, ty+th), color, 3) #draw tracking box
            tip = RECYCLING_TIPS.get(current_label, "") #get tip for current label
            frame = draw_glass_box(frame, tx, ty+th+10, tw, 30, (0,0,0), 0.6) #draw tip box
            cv2.putText(frame, tip, (tx+5, ty+th+30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
        else: tracking_active = False
    else:
        x1, y1 = (width-STATIC_BOX_SIZE)//2, (height-STATIC_BOX_SIZE)//2 #center box coordinates
        roi = frame[y1:y1+STATIC_BOX_SIZE, x1:x1+STATIC_BOX_SIZE] #region of interest
        detected = False
        if roi.size != 0: #ensure roi is valid
            results = model(roi, verbose=False) #model prediction
            if results[0].probs:
                idx = results[0].probs.top1; conf = float(results[0].probs.top1conf); lbl = labels[idx].upper().strip() #get higest prob
                if conf > CONFIDENCE_THRESHOLD: current_label = lbl; current_conf = conf; detected = True #accept if above threshold
                else: current_label = "Scanning..."
            elif len(results[0].boxes) > 0: #boxes detected not probs
                box = results[0].boxes[0]; idx = int(box.cls); conf = float(box.conf); lbl = labels[idx].upper().strip()
                if conf > CONFIDENCE_THRESHOLD: current_label = lbl; current_conf = conf; detected = True
                else: current_label = "Scanning..." #continue scanning
            else: current_label = "Scanning..."
        
        color = BIN_COLORS.get(current_label, ((200,200,200),""))[0] if detected else (255,255,255) #get color for box otherwise white
        cv2.rectangle(frame, (x1, y1), (x1+STATIC_BOX_SIZE, y1+STATIC_BOX_SIZE), color, 2) #draw static box
        if detected: cv2.putText(frame, f"FOUND: {current_label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # UI DRAWING
    frame = draw_glass_box(frame, 20, 20, 160, 80, (0,0,0), 0.4) #top-left box
    cv2.putText(frame, f"USER: {operator_name.upper()}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1) #display user
    cv2.putText(frame, f"TOTAL: {sum(item_counts.values())}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2) #display total count
    frame = draw_glass_box(frame, width-140, 20, 120, 40, (128, 0, 128), 0.6) #ASK AI box
    cv2.putText(frame, "ASK AI", (width-110, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    draw_capacity_bars(frame, item_counts, MAX_CAPACITY) #capacity bars
    draw_eco_panel(frame, item_counts) #eco panel
    
    # Buttons
    frame = draw_glass_box(frame, 20, height-60, 120, 40, (0,0,150), 0.6) # Exit
    cv2.putText(frame, "EXIT", (55, height-33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cx = width//2
    frame = draw_glass_box(frame, cx-60, height-60, 120, 40, (100,100,100), 0.6) # Reset
    cv2.putText(frame, "RESET", (cx-30, height-33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    if not tracking_active: #only show capture if not tracking
        frame = draw_glass_box(frame, width-140, height-60, 120, 40, (200,100,0), 0.6) 
        cv2.putText(frame, "CAPTURE", (width-125, height-33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Waste Guard for Internship AIROST", frame) #show frame in window
    if cv2.waitKey(1) & 0xFF == ord('q'): break #quit if 'q' pressed

speak("Saving Session. Goodbye.")
cap.release()
cv2.destroyAllWindows() #close all windows

# ==========================================
#   DEBUGGING SAVE SECTION
# ==========================================
print("-----------------------------------")
print("[INFO] STARTING DATA SAVE...")

# 1. EXCEL SAVE
if session_log:
    f = "Waste_Report_Master.xlsx"
    new = pd.DataFrame(session_log) #new data to save
    try:
        old = pd.read_excel(f, sheet_name='Detailed_Log') #read old data
        combined = pd.concat([old, new], ignore_index=True) #combine old and new
    except: combined = new
    try:
        with pd.ExcelWriter(f, mode='w') as w: #write to excel
            summary = combined.groupby(['Operator', 'Category']).size().reset_index(name='Total Count') #summary by user and category
            summary.to_excel(w, sheet_name='Summary_By_User', index=False) #save summary sheet
            combined.to_excel(w, sheet_name='Detailed_Log', index=False) #save detailed log sheet
        print(f"[SUCCESS] Excel saved to {f}")
    except PermissionError:
        print(f"[CRITICAL ERROR] Could not save Excel. File '{f}' is OPEN. Close it!")
    except Exception as e:
        print(f"[ERROR] Excel Save Failed: {e}")
else:
    print("[INFO] No new items to save to Excel.")

# 2. PICKLE SAVE
if sum(item_counts.values()) > 0: #only save if there are counts
    try:
        with open('session_data.pkl', 'wb') as f: pickle.dump(item_counts, f) #save counts to pickle
        print("[SUCCESS] Session memory saved.")
    except Exception as e: print(f"[ERROR] Pickle Save Failed: {e}")

# 3. GRAPH SAVE (New Folder)
if sum(item_counts.values()) > 0:
    try:
        active = {k:v for k,v in item_counts.items() if v>0} #only active items
        plt.figure(figsize=(10,6)); plt.bar(active.keys(), active.values(), color='blue') #matplotlib bar graph
        plt.title(f"Session: {operator_name}"); #title
        # Save to Folder
        g_name = f"Graph Analysis/Session_Graph_{datetime.now().strftime('%Y%m%d_%H%M')}.png" #filename format
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
        receipt_text += f"CO2 Saved: {total_co2:.2f} kg\nThank you!" #receipt content
        
        qr = qrcode.make(receipt_text) #generate QR code
        
        # New Folder + Filename
        ts = datetime.now().strftime('%Y%m%d_%H%M') #timestamp
        qr_name = f"Qr collection/{operator_name}_{ts}.png" #filename format
        qr.save(qr_name)
        
        qr_img = cv2.imread(qr_name) #read qr image
        cv2.imshow("Scan Receipt", qr_img)
        cv2.waitKey(10000) #display for 10 seconds
        print(f"[SUCCESS] QR Receipt saved as {qr_name}")
    except Exception as e: print(f"[ERROR] QR Failed: {e}")

print("-----------------------------------")