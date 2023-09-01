

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from tumor_detection import load_data, preprocess_data, train_models, predict_tumor

# Load data and preprocess
classes = {'no_tumor': 0, 'pituitary_tumor': 1}
X, Y = load_data(classes, 'brain_tumor-dataset\Training')
X_flatten, pca = preprocess_data(X)

# Train models
xtrain, xtest, ytrain, ytest = train_test_split(X_flatten, Y, random_state=10, test_size=0.20)
lg, sv = train_models(xtrain, ytrain, pca)

# GUI and user input
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    if file_path:
        predict_and_update(file_path)

def predict_and_update(file_path):
    try:
        user_img = cv2.imread(file_path, 0)
        user_img_resized = cv2.resize(user_img, (200, 200))
        user_img_flatten = user_img_resized.reshape(1, -1) / 255
        user_prediction = predict_tumor(user_img_flatten, pca, sv)
        user_prediction_label = 'Positive Tumor' if user_prediction[0] == 1 else 'No Tumor'
        display_image(user_img_resized, user_prediction_label)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def display_image(image, prediction):
    img = Image.fromarray(image)
    img.thumbnail((400, 400))
    img = ImageTk.PhotoImage(img)
    img_label.config(image=img)
    img_label.image = img
    prediction_label.config(text=f"Prediction: {prediction}")

# Initialize Tkinter
root = tk.Tk()
root.title("Brain Tumor Detection")

# Style
style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", padding=10, font=('Helvetica', 12))
style.configure("TLabel", font=('Helvetica', 14))

# Create UI elements
select_button = ttk.Button(root, text="Select MRI Image", command=select_image)
img_label = ttk.Label(root)
prediction_label = ttk.Label(root, text="Prediction: ")

# Grid layout
select_button.pack(pady=20)
img_label.pack()
prediction_label.pack(pady=10)

# Start GUI event loop
root.mainloop()
