import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI License Plate Detector", layout="wide")
st.title("🚗 Indian License Plate Recognition (ALPR)")
st.write("Upload a car image to detect the license plate and extract the text.")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # Load your custom YOLOv8 model
    model = YOLO("best.pt")
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'], gpu=False) # CPU is safer for basic cloud tiers
    return model, reader

model, reader = load_models()

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # UI Columns
    col1, col2 = st.columns(2)
    with col1:
        st.image(display_image, caption="Uploaded Image", use_container_width=True)

    # --- DETECTION & OCR ---
    if st.button("Run Detection"):
        results = model.predict(source=image, conf=0.5)[0]
        
        if len(results.boxes) == 0:
            st.warning("No license plate detected. Try adjusting the image or confidence.")
        else:
            for box in results.boxes:
                # Crop Plate
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = image[y1:y2, x1:x2]
                
                # Preprocess for OCR
                gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                ocr_result = reader.readtext(gray_plate)
                
                # Extract text
                plate_text = ocr_result[0][1].upper() if ocr_result else "TEXT UNREADABLE"
                
                # Draw on display image
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
                
                with col2:
                    st.image(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB), caption="Cropped Plate")
                    st.success(f"**Detected Plate Number:** {plate_text}")
            
            st.image(display_image, caption="Processed Result", use_container_width=True)