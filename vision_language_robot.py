# vision_language_robot_continuous.py

from ultralytics import YOLO
import cv2
import os
import google.generativeai as genai

# -----------------------------
# Step 1: Initialize Models
# -----------------------------
# Load YOLO model
yolo_model = YOLO('yolov8n.pt')  # Small pretrained model

# Configure Gemini API key securely
genai.configure(api_key="AIzaSyBDuel0CPx4qWnRMCGwpw0PPtSnIZTgeYo")

# Create Gemini model
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# -----------------------------
# Step 2: Load Image
# -----------------------------
image_path = "1.jpg"  # Replace with your image path
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# Run YOLO detection
results = yolo_model(img)

# Extract detected objects safely
detected_objects = [int(box.cls) for box in results[0].boxes] if len(results[0].boxes) > 0 else []
object_names = [results[0].names[cls] for cls in detected_objects] if detected_objects else []

print("Detected objects:", object_names if object_names else "None")

# -----------------------------
# Step 3: Gemini Query Function
# -----------------------------
def ask_gemini(prompt_text: str) -> str:
    """Send a prompt to Gemini and return deterministic text response."""
    try:
        response = gemini_model.generate_content(
            contents=prompt_text,
            generation_config={
                "temperature": 0.0,  # deterministic output
                "top_p": 1.0,
                "top_k": 1
            },
        )
        return response.text.strip()
    except Exception as e:
        return f"Error contacting Gemini: {e}"

# -----------------------------
# Step 4: Continuous Command Loop
# -----------------------------
print("\nRobot is ready. Type 'exit' to stop.\n")

while True:
    user_command = input("Enter your command for the robot: ").strip()
    if not user_command:
        print("Please enter a command or type 'exit' to quit.")
        continue
    if user_command.lower() in {"exit", "quit", "stop"}:
        print("Robot shutting down. Goodbye!")
        break

    # --- Quick Offline Logic (for simple tasks) ---
    lower_cmd = user_command.lower()
    if "count" in lower_cmd:
        found = False
        for obj in set(object_names):
            if obj in lower_cmd:
                count = object_names.count(obj)
                print(f"\nRobot Response: There {'is' if count == 1 else 'are'} {count} {obj}{'' if count == 1 else 's'}.\n")
                found = True
                break
        if found:
            continue

    # Otherwise, ask Gemini
    if object_names:
        prompt = f"The detected objects in the image are: {object_names}. {user_command}"
    else:
        prompt = f"No objects were detected in the image. {user_command}"

    robot_response = ask_gemini(prompt)
    print("\nRobot Response:", robot_response, "\n")

# -----------------------------
# Step 5: Display Image
# -----------------------------
boxed_image = results[0].plot()
cv2.imshow("Detected Objects", boxed_image)
output_path = "boxed_output.png"
cv2.imwrite(output_path, boxed_image)
print(f"Output image saved to: {output_path}")

cv2.waitKey(0)
cv2.destroyAllWindows()
