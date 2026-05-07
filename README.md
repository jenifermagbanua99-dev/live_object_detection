🎥 Live Object Detection & Tracing (YOLOv8 + Streamlit)
📌 Project Overview

The Live Object Detection & Tracing System is a web-based AI application built using Streamlit and YOLOv8 (Ultralytics). It enables real-time object detection through a webcam or uploaded images.

The system detects objects instantly, draws bounding boxes, labels detected items, and provides object counting with performance monitoring.

🚀 Features
📷 Live Camera Object Detection
🖼️ Image Upload Detection
🎯 Real-time Bounding Box Visualization
🏷️ Object Labeling (person, bottle, cell phone, etc.)
📊 Object Counting per Class
⚡ Fast YOLOv8 AI Inference
🎚️ Adjustable Confidence Threshold
📈 Processing Time Monitoring
🌐 Streamlit Web Interface
🧠 Technology Stack
Python
Streamlit
YOLOv8 (Ultralytics)
OpenCV
NumPy
Pillow (PIL)
📂 Project Structure
📁 yolov8-streamlit-app
│
├── app.py                 # Main Streamlit application
├── yolov8n.pt            # YOLOv8 pretrained model
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/your-username/yolov8-streamlit-app.git
cd yolov8-streamlit-app
2. Install dependencies
pip install -r requirements.txt
3. Run the app
streamlit run app.py
📦 requirements.txt
streamlit
ultralytics
numpy
opencv-python
pillow
📸 How It Works
Open the web application
Select mode:
Camera Detection
Image Upload
Input is processed using YOLOv8 model
Objects are detected in real-time
Bounding boxes and labels are displayed
Object count and processing time are shown
📊 Expected Output
Real-time object detection
Bounding boxes around detected objects
Labels such as:
person
bottle
cell phone
chair
Object counting per frame
Processing time per detection
🎯 Enhancements Added
✔ Object Counting System
✔ Confidence Threshold Control
✔ Processing Time Display
✔ Stable Image Preprocessing
✔ Streamlit Cloud Deployment Ready
⚠️ Notes
Ensure yolov8n.pt is included in the project directory
Allow camera permission in browser
Best performance in well-lit environments
👨‍💻 Developer

Ma. Rhozeth B. Paz
BSCS - 3A

📌 License

This project is created for educational purposes only.