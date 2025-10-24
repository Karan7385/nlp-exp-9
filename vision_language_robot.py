# vision_language_robot_gemini_safe.py

from ultralytics import YOLO
import cv2
import os
import google.generativeai as genai  # Gemini Python SDK

# -----------------------------
# Step 1: Initialize Models
# -----------------------------
# Load YOLO model
yolo_model = YOLO('yolov8n.pt')  # Small pretrained model

# Configure Gemini API key securely
client = genai.configure(api_key="AIzaSyBDuel0CPx4qWnRMCGwpw0PPtSnIZTgeYo")

# -----------------------------
# Step 2: Load Image
# -----------------------------
image_path = "test_image.png"  # Replace with your image path
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# Run YOLO detection
results = yolo_model(img)

# Extract class indexes and object names safely
detected_objects = [int(box.cls) for box in results[0].boxes] if len(results[0].boxes) > 0 else []
object_names = [results[0].names[cls] for cls in detected_objects] if detected_objects else []

print("Detected objects:", object_names if object_names else "None")

# -----------------------------
# Step 3: Gemini Query Function
# -----------------------------
def ask_gemini(prompt_text: str) -> str:
    """
    Send a prompt to Gemini and return the response text.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_text
        )
        return response.text
    except Exception as e:
        return f"Error contacting Gemini: {e}"

# -----------------------------
# Step 4: Get User Command
# -----------------------------
user_command = input("Enter your command for the robot: ").strip()
if not user_command:
    user_command = "Describe the objects in the image."

# Construct prompt safely
if object_names:
    prompt = f"The detected objects in the image are: {object_names}. {user_command}"
else:
    prompt = f"No objects were detected in the image. {user_command}"

# Get Gemini response
robot_response = ask_gemini(prompt)
print("\nRobot Response:", robot_response)

# -----------------------------
# Step 5: Display Image with Bounding Boxes
# -----------------------------
boxed_image = results[0].plot()
cv2.imshow("Detected Objects", boxed_image)

# Optional: Save output image
output_path = "boxed_output.png"
cv2.imwrite(output_path, boxed_image)
print(f"Output image saved to: {output_path}")

cv2.waitKey(0)
cv2.destroyAllWindows()