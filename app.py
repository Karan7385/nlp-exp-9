import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Step 1: App Configuration
# -----------------------------
st.set_page_config(page_title="Vision-Language Robot", page_icon="🤖", layout="wide")

st.title("🤖 Vision-Language Robot")
st.markdown("Upload an image, detect objects using YOLO, and ask the robot questions about it!")

# -----------------------------
# Step 2: Initialize Models
# -----------------------------
@st.cache_resource
def load_models():
    yolo = YOLO("yolov8n.pt")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini = genai.GenerativeModel("gemini-2.5-flash")
    return yolo, gemini

yolo_model, gemini_model = load_models()

# -----------------------------
# Step 3: Upload Image
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Run YOLO detection
    with st.spinner("Detecting objects..."):
        results = yolo_model(img)

    # Extract detected objects
    detected_objects = [int(box.cls) for box in results[0].boxes] if len(results[0].boxes) > 0 else []
    object_names = [results[0].names[cls] for cls in detected_objects] if detected_objects else []

    # Display results
    st.subheader("🧠 Detected Objects:")
    if object_names:
        st.write(", ".join(object_names))
    else:
        st.write("No objects detected.")

    # Display image with boxes
    boxed_img = results[0].plot()
    st.image(boxed_img, caption="Detected Objects", use_container_width=True)

    # -----------------------------
    # Step 4: Gemini Query Function
    # -----------------------------
    def ask_gemini(prompt_text: str) -> str:
        try:
            response = gemini_model.generate_content(
                contents=prompt_text,
                generation_config={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 1
                },
            )
            return response.text.strip()
        except Exception as e:
            return f"⚠️ Error contacting Gemini: {e}"

    # -----------------------------
    # Step 5: User Query Input
    # -----------------------------
    st.subheader("💬 Chat with the Robot")

    user_command = st.text_input("Enter your command:", placeholder="e.g., How many people are there?")
    if st.button("Ask"):
        if not user_command:
            st.warning("Please enter a command first.")
        else:
            lower_cmd = user_command.lower()
            response_text = ""

            # --- Simple Offline Logic ---
            if "count" in lower_cmd:
                found = False
                for obj in set(object_names):
                    if obj in lower_cmd:
                        count = object_names.count(obj)
                        response_text = f"There {'is' if count == 1 else 'are'} {count} {obj}{'' if count == 1 else 's'}."
                        found = True
                        break
                if found:
                    st.success(f"🤖 Robot Response: {response_text}")
                    st.stop()

            # --- Otherwise, Ask Gemini ---
            if object_names:
                prompt = f"The detected objects in the image are: {object_names}. {user_command}"
            else:
                prompt = f"No objects were detected in the image. {user_command}"

            with st.spinner("Thinking..."):
                response_text = ask_gemini(prompt)

            st.success(f"🤖 Robot Response: {response_text}")

else:
    st.info("👆 Please upload an image to get started.")

