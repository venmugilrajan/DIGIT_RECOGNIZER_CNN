# -*- coding: utf-8 -*-
"""
Gradio App for Handwritten Digit Recognition using a pre-trained CNN model.
"""

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

print("Importing libraries and loading model...")

# --- Configuration ---
MODEL_PATH = "digit_cnn.keras" # Make sure this path is correct

# --- Load the pre-trained model ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    print("Please ensure the model file exists and is a valid Keras model.")
    model = None

# --- Preprocessing Function ---
def preprocess_image(input_data):
    """
    Preprocesses the input image array (potentially within a dict) from Gradio Sketchpad
    to match the model's input requirements. Handles RGBA input.
    """
    print(f"DEBUG: Received input type: {type(input_data)}") # Check input type

    img_array = None # Initialize img_array

    # Check if input is a dictionary and try to extract the image array
    if isinstance(input_data, dict):
        print(f"DEBUG: Input is a dictionary. Keys: {list(input_data.keys())}")
        img_array = input_data.get('composite') # Get the composite image which includes the drawing
        if img_array is None:
             print("DEBUG: 'composite' key not found or value is None, returning None.")
             return None
        else:
             print("DEBUG: Extracted array from 'composite' key.")
        print(f"DEBUG: Extracted array type: {type(img_array)}")

    elif isinstance(input_data, np.ndarray):
         print("DEBUG: Input is already a NumPy array.")
         img_array = input_data # Input is already the array we expect
    else:
        print("DEBUG: Input is not a dict or NumPy array, trying np.array().")
        # Fallback attempt if it's something else convertible
        try:
            img_array = np.array(input_data)
        except Exception as e:
            print(f"DEBUG: Error converting unexpected input type to NumPy array: {e}")
            return None

    if img_array is None:
        print("DEBUG: img_array is None after type checking, returning None.")
        return None

    # Ensure it's a NumPy array after potential extraction
    try:
        if not isinstance(img_array, np.ndarray):
             print("DEBUG: Extracted object is not NumPy array, attempting conversion.")
             img = np.array(img_array)
        else:
             img = img_array # Already a numpy array
        print(f"DEBUG: Working with np array. Shape: {img.shape}, Size: {img.size}, Dtype: {img.dtype}")
    except Exception as e:
        print(f"DEBUG: Error ensuring input is NumPy array after dict check: {e}")
        return None

    # Check for empty array explicitly before shape checks
    if img.size <= 1:
        print(f"DEBUG: Image size is <= 1 (Shape: {img.shape}), returning None.")
        if len(img.shape) == 0:
            return None # Expected empty input, return None silently
        else:
            return None

    # Handle image format: RGBA (4 channels), RGB (3 channels), or Grayscale (2 channels)
    try:
        # Ensure correct dtype for PIL if needed (e.g., uint8)
        if img.dtype != np.uint8:
            print(f"DEBUG: Converting dtype from {img.dtype} to uint8 for PIL.")
            img = img.astype(np.uint8)

        if len(img.shape) == 3 and img.shape[2] == 4:
            print("DEBUG: Input shape indicates RGBA, converting to Grayscale.")
            # Convert RGBA to RGB first (discard alpha), then to Grayscale
            img_pil = Image.fromarray(img).convert('RGB').convert('L')
            img = np.array(img_pil)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            print("DEBUG: Input shape indicates RGB, converting to Grayscale.")
            img_pil = Image.fromarray(img).convert('L')
            img = np.array(img_pil)
        elif len(img.shape) == 2: # Already grayscale
            print("DEBUG: Input shape indicates Grayscale.")
            pass # No conversion needed
        else:
            print(f"DEBUG: Unexpected image shape after size check: {img.shape}, returning None.")
            return None # Handle other unexpected shapes
    except Exception as e:
        print(f"Error during image format conversion: {e}")
        return None


    # Resize to 28x28 using PIL
    try:
        # Ensure it's grayscale before resizing if conversion happened
        img_pil_resized = Image.fromarray(img).resize((28, 28), Image.Resampling.LANCZOS)
        img_resized = np.array(img_pil_resized)
        print(f"DEBUG: Resized grayscale image shape: {img_resized.shape}") # Check shape after resize
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None

    # Invert colors (black background, white digit becomes white background, black digit)
    img_inverted = 255.0 - img_resized

    # Normalize pixel values to [0, 1]
    img_normalized = img_inverted / 255.0

    # Reshape for the model (add batch and channel dimensions)
    img_reshaped = img_normalized.reshape(1, 28, 28, 1)

    # Ensure dtype is float32 for TensorFlow
    img_final = img_reshaped.astype(np.float32)

    print(f"DEBUG: Final processed image shape: {img_final.shape}, dtype: {img_final.dtype}")
    return img_final

# --- Prediction Function ---
def predict_digit(sketchpad_input): # Renamed variable for clarity
    """
    Takes the sketchpad input, preprocesses it, and returns the model's prediction.
    Returns an empty dictionary if processing fails.
    """
    print("\n--- New Prediction Request ---")
    if model is None:
        print("Error: Model not loaded.")
        return {}
    if sketchpad_input is None:
        print("Input sketchpad_input is None.")
        return {}

    # Pass the raw sketchpad input (which might be a dict) to preprocess
    processed_image = preprocess_image(sketchpad_input)

    if processed_image is None:
         print("Image preprocessing returned None.")
         return {}

    print(f"DEBUG: Image ready for prediction, shape: {processed_image.shape}")

    # Make prediction
    try:
        prediction = model.predict(processed_image, verbose=0)
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return {}

    probabilities = prediction[0]
    confidences = {str(i): float(probabilities[i]) for i in range(10)}
    print(f"DEBUG: Returning confidences: {confidences}")
    return confidences

# --- Create and Launch Gradio Interface ---
print("Setting up Gradio interface...")

iface = gr.Interface(
    fn=predict_digit,
    # Keep type="numpy" - the handler now expects the dict it might receive
    inputs=gr.Sketchpad(label="Draw a digit (0-9) here", type="numpy"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Handwritten Digit Recognizer",
    description="Draw a single digit (0-9) in the box below and click 'Submit' to see the prediction from a trained Convolutional Neural Network (CNN). Clear the sketchpad before submitting if it's empty.",
    examples=[],
    live=False
)

print("Launching Gradio app...")

if __name__ == "__main__":
    iface.launch()

print("Gradio app launch command issued. Check the output for the URL.")

