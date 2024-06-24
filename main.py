import numpy as np
from PIL import Image
import tempfile
import os
from flask import Flask, jsonify, request
from keras.models import load_model
from keras import backend as K
import functions_framework

app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = "brain_mri_seg.h5"
model = None  # Placeholder for model loading


# Function to create dice coefficient
def dice_coef(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)


# Function to create dice loss
def dice_loss(y_true, y_pred, smooth=100):
    return -dice_coef(y_true, y_pred, smooth)


# Function to create iou coefficient
def iou_coef(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    sum = K.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou


# Custom objects for model loading
custom_objects = {"dice_coef": dice_coef, "dice_loss": dice_loss, "iou_coef": iou_coef}


# Function to load the model
def load_model_from_root():
    global model
    if model is None:
        model = load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)


# HTTP function to handle image upload and prediction
@functions_framework.http
def predict(request):
    load_model_from_root()

    # Check if request contains file data
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]

    # Check if the file is a TIFF image
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400
    if not (file.filename.endswith(".tif") or file.filename.endswith(".tiff")):
        return jsonify({"error": "Uploaded file must be a .tif or .tiff image"}), 400

    # Create a temporary file to save the uploaded TIFF
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_tif_file:
        temp_filename = temp_tif_file.name
        file.save(temp_filename)

        # Process the uploaded image
        uploaded_img = Image.open(temp_filename).convert("RGB")

        # Resize and preprocess the image for prediction
        image = uploaded_img.resize((256, 256))
        image = np.array(image)
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)

        # Predict the mask
        pred_mask = model.predict(image)
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

        # Save the predicted mask as .png
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_png_file:
            pred_mask_img = Image.fromarray(pred_mask.squeeze(), mode="L")
            pred_mask_img.save(temp_png_file.name)

            # Return the predicted PNG file
            with open(temp_png_file.name, "rb") as png_file:
                return png_file.read(), 200, {"Content-Type": "image/png"}