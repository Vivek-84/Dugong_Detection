import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import cv2
import zipfile
import shutil
from io import BytesIO
import pandas as pd

# Load YOLO model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Object detection function
def detect_objects(model, image_path):
    # Run inference
    results = model(image_path)
    return results

# Draw bounding boxes on the image
def draw_boxes(image_path, results, threshold=0.5):
    image = cv2.imread(image_path)
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if conf > threshold:
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Save images to a folder
def save_images(folder_path, images):
    os.makedirs(folder_path, exist_ok=True)
    for name, image in images.items():
        image.save(os.path.join(folder_path, name))

# Create a ZIP file from a folder
def create_zip(folder_path):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    zip_buffer.seek(0)
    return zip_buffer

# Streamlit UI
st.title("Object Detection App")
st.header("Upload Images or a Folder of Images for Detection")

# Pre-configured model path
model_path = "model/best.pt"

# Load the model
st.write("Loading the model...")
model = load_model(model_path)
st.success("Model loaded successfully!")

# File uploader for images or folders
uploaded_files = st.file_uploader(
    "Upload Images or a Folder (ZIP file):",
    accept_multiple_files=True,
    type=["jpg", "png", "zip"]
)

# Temporary directories
temp_input_dir = "temp_input"
temp_detected_dir = "temp_detected"
temp_undetected_dir = "temp_undetected"

# List to hold detection results for Excel
detection_results = []

if st.button("Detect Objects"):
    if uploaded_files:
        detected_images = {}
        undetected_images = {}

        try:
            # Create directories
            os.makedirs(temp_input_dir, exist_ok=True)
            os.makedirs(temp_detected_dir, exist_ok=True)
            os.makedirs(temp_undetected_dir, exist_ok=True)

            # Save or extract uploaded files
            for uploaded_file in uploaded_files:
                if uploaded_file.name.endswith(".zip"):
                    with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                        zip_ref.extractall(temp_input_dir)
                else:
                    file_path = os.path.join(temp_input_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.read())

            total_files = 0
            detected_count = 0
            undetected_count = 0

            # Process all images in the temp_input_dir
            for root, _, files in os.walk(temp_input_dir):
                for file in files:
                    total_files += 1
                    file_path = os.path.join(root, file)
                    results = detect_objects(model, file_path)

                    if len(results[0].boxes) >= 1:  # At least one object detected
                        detected_image = draw_boxes(file_path, results)
                        detected_images[file] = detected_image
                        detected_count += 1

                        max_conf = max(results[0].boxes.conf)
                        detection_results.append({
                            'File Path': file_path,
                            'Status': 'Detected',
                            'Confidence': max_conf.item(),
                            'Count': len(results[0].boxes)
                        })
                    else:
                        undetected_images[file] = Image.open(file_path)
                        undetected_count += 1
                        detection_results.append({
                            'File Path': file_path,
                            'Status': 'Undetected',
                            'Confidence': None,
                            'Count': 0
                        })

            # Save detected and undetected images
            save_images(temp_detected_dir, detected_images)
            save_images(temp_undetected_dir, undetected_images)

            st.success("Detection completed! Files are ready for download.")

            # Display statistics
            st.markdown(f"### Statistics")
            st.write(f"Total files uploaded: {total_files}")
            st.write(f"Detected images: {detected_count}")
            st.write(f"Undetected images: {undetected_count}")

            # Create DataFrame and save as Excel
            df = pd.DataFrame(detection_results)
            df = df.groupby('File Path').agg({
                'Status': 'first',
                'Confidence': 'max',
                'Count': 'max'
            }).reset_index()

            excel_file = "detection_results.xlsx"
            df.to_excel(excel_file, index=False)

            # Provide download link for Excel file
            with open(excel_file, "rb") as f:
                st.download_button(
                    label="Export Detection Results (Excel)",
                    data=f,
                    file_name=excel_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            # Remove the Excel file after download
            os.remove(excel_file)

        finally:
            shutil.rmtree(temp_input_dir, ignore_errors=True)

if os.path.exists(temp_detected_dir):
    detected_zip = create_zip(temp_detected_dir)
    st.markdown("### Export Detected Images:")
    st.download_button(
        "Export Detected Images",
        data=detected_zip,
        file_name="detected_images.zip",
        mime="application/zip"
    )

if os.path.exists(temp_undetected_dir):
    undetected_zip = create_zip(temp_undetected_dir)
    st.markdown("### Export Undetected Images:")
    st.download_button(
        "Export Undetected Images",
        data=undetected_zip,
        file_name="undetected_images.zip",
        mime="application/zip"
    )

if st.button("Clear Temporary Files"):
    shutil.rmtree(temp_detected_dir, ignore_errors=True)
    shutil.rmtree(temp_undetected_dir, ignore_errors=True)
    st.success("Temporary files cleared.")
