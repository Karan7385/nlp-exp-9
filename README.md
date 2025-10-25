
# 🤖 Vision-Language Robot using YOLO & Gemini

An interactive Streamlit web app that combines computer vision (YOLO) and natural language understanding (Gemini) to let users ask questions about objects detected in an image — just like talking to a smart robot 👁️💬🧠

## ✨ Features

✅ Upload any image (JPG, PNG)  
✅ Detect objects in the image using YOLOv8  
✅ View image with detection boxes  
✅ Ask natural language questions about the detected objects  
✅ Built-in offline "counting" logic (e.g., "How many cars are there?")  
✅ Uses Google Gemini (Generative AI) for intelligent responses  
✅ Clean Streamlit UI


## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Karan7385/nlp-exp-9.git
cd nlp-exp-9
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows

deactivate                # All Platforms
```

### 3. Install dependencies

```bash
pip install streamlit ultralytics opencv-python-headless pillow google-generativeai python-dotenv
```

## 🔑 Environment Setup

Create a file named `.env` in the project root and add your Gemini API key:

```
GEMINI_API_KEY=your_api_key
```

⚠️ **Keep this file private and never upload it to GitHub.**  
Add `.env` to your `.gitignore` file.

## 🚀 Running the App

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown (usually `http://localhost:8501`).

## 🖼️ How It Works

1. Upload an image through the Streamlit interface.
2. The app runs YOLOv8 to detect all objects in the image.
3. You can see:
   - List of detected objects
   - Image with bounding boxes
4. Ask the robot natural language questions like:
   - "How many people are there?"
   - "What objects can you see?"
   - "Describe what's happening in the image."
5. The robot uses Gemini to generate a natural response.

## 🧩 YOLOv8 Object Classes

The YOLOv8 model used in this project was trained on the COCO dataset, which contains 80 common object categories.

**It can only recognize and detect the following objects 👇**

| ID | Class Name      | ID | Class Name      |
|----|-----------------|----|-----------------| 
| 0  | person          | 40 | skateboard      |
| 1  | bicycle         | 41 | surfboard       |
| 2  | car             | 42 | tennis racket   |
| 3  | motorcycle      | 43 | bottle          |
| 4  | airplane        | 44 | wine glass      |
| 5  | bus             | 45 | cup             |
| 6  | train           | 46 | fork            |
| 7  | truck           | 47 | knife           |
| 8  | boat            | 48 | spoon           |
| 9  | traffic light   | 49 | bowl            |
| 10 | fire hydrant    | 50 | banana          |
| 11 | stop sign       | 51 | apple           |
| 12 | parking meter   | 52 | sandwich        |
| 13 | bench           | 53 | orange          |
| 14 | bird            | 54 | broccoli        |
| 15 | cat             | 55 | carrot          |
| 16 | dog             | 56 | hot dog         |
| 17 | horse           | 57 | pizza           |
| 18 | sheep           | 58 | donut           |
| 19 | cow             | 59 | cake            |
| 20 | elephant        | 60 | chair           |
| 21 | bear            | 61 | couch           |
| 22 | zebra           | 62 | potted plant    |
| 23 | giraffe         | 63 | bed             |
| 24 | backpack        | 64 | dining table    |
| 25 | umbrella        | 65 | toilet          |
| 26 | handbag         | 66 | TV              |
| 27 | tie             | 67 | laptop          |
| 28 | suitcase        | 68 | mouse           |
| 29 | frisbee         | 69 | remote          |
| 30 | skis            | 70 | keyboard        |
| 31 | snowboard       | 71 | cell phone      |
| 32 | sports ball     | 72 | microwave       |
| 33 | kite            | 73 | oven            |
| 34 | baseball bat    | 74 | toaster         |
| 35 | baseball glove  | 75 | sink            |
| 36 | skateboard      | 76 | refrigerator    |
| 37 | surfboard       | 77 | book            |
| 38 | tennis racket   | 78 | clock           |
| 39 | bottle          | 79 | vase            |

🟡 **Note:**  
If you upload an image containing objects outside this list (for example: "ring", "pencil", "robot"), the YOLO model will not recognize or detect them, even though Gemini can understand them linguistically.

## 🧠 Future Enhancements

- Add live webcam input for real-time object detection
- Integrate voice commands and speech output
- Add scene summarization / captioning
- Extend object detection to custom-trained models

## 📜 License

This project is for educational and research purposes only.  
All model rights belong to their respective owners.