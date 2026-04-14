import torch
import tkinter as tk
from tkinter import messagebox
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Label encoder
with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

# Model and tokenizer
print("✅ Loading ALBERT model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# GUI
def predict_emotion():
    text = text_entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Error", "Please enter some text!")
        return
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
        pred = outputs.logits.argmax(dim=1).item()
    emotion = le.inverse_transform([pred])[0]
    messagebox.showinfo("Predicted Emotion", f"Emotion: {emotion}")

root = tk.Tk()
root.title("Emotion Recognition - ALBERT")
root.geometry("400x300")

tk.Label(root, text="Enter text:", font=("Arial", 12)).pack(pady=10)
text_entry = tk.Text(root, height=5, width=40)
text_entry.pack(pady=5)
tk.Button(root, text="Predict Emotion", command=predict_emotion, font=("Arial", 12)).pack(pady=10)

root.mainloop()
