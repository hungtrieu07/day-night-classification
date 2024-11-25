import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
from collections import Counter

def predict_image(model, image_path, img_size, class_labels):
    """
    Predict the class of a single image using the trained model.

    Args:
        model (keras.Model): Loaded trained model.
        image_path (str): Path to the image.
        img_size (tuple): Model's expected input size (height, width).
        class_labels (list): List of class labels.

    Returns:
        str: Predicted class label.
    """
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, img_size)
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Predict the class probabilities
    predictions = model.predict(image)[0]
    predicted_class_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_class_index]
    return predicted_label

def process_dataset(model_path, dataset_dir, img_size, class_labels):
    """
    Predict all images in a dataset and count occurrences of each class.

    Args:
        model_path (str): Path to the trained model (.h5 file).
        dataset_dir (str): Root directory of the dataset.
        img_size (tuple): Model's expected input size (height, width).
        class_labels (list): List of class labels.

    Returns:
        dict: Counts of each class in the dataset.
    """
    # Load the trained model
    print("[INFO] Loading model...")
    model = load_model(model_path)

    # Initialize counters
    class_counts = Counter()

    # Traverse dataset directories
    subdirs = ["test/images", "train/images", "valid/images"]
    for subdir in subdirs:
        dir_path = os.path.join(dataset_dir, subdir)
        if not os.path.exists(dir_path):
            print(f"[WARNING] Directory {dir_path} does not exist. Skipping...")
            continue

        print(f"[INFO] Processing images in {dir_path}...")
        for file_name in os.listdir(dir_path):
            if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(dir_path, file_name)
                try:
                    # Predict the class of the image
                    predicted_label = predict_image(model, image_path, img_size, class_labels)
                    class_counts[predicted_label] += 1
                except Exception as e:
                    print(f"[ERROR] Failed to process {image_path}: {e}")

    return class_counts

if __name__ == "__main__":
    # Define parameters
    model_path = "model/day_night.h5"  # Path to your trained model
    dataset_dir = "FireSmokeHuman.v13i.yolov11"  # Root directory of the dataset
    img_size = (28, 28)  # Replace with your model's input size
    class_labels = ["night", "day"]  # Update based on your model's output classes

    # Process the dataset and count class occurrences
    class_counts = process_dataset(model_path, dataset_dir, img_size, class_labels)

    # Print the results
    print("\nClass counts in the dataset:")
    for label, count in class_counts.items():
        print(f"{label}: {count} images")
