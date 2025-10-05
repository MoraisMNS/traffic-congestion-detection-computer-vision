# 🛣️ Traffic Congestion Detection System  
**Using YOLOv8 and Streamlit**

---

## Overview  

This project is designed to detect and classify **traffic congestion levels** from video footage or live webcam feeds using **YOLOv8** for vehicle detection.  
It provides a simple **Streamlit interface** where users can upload videos or start real-time detection. The system analyzes traffic flow and determines whether the congestion level is **Light**, **Moderate**, or **Heavy** based on vehicle count thresholds.  

---

## Project Structure  

```
TRAFFIC-CONGESTION-DETECTION/
│
├── frontend/
│   ├── app.py                  # Streamlit entry point
│   ├── main_frontend.py        # Main user interface
│   ├── single_video_mode.py    # Single video detection mode
│   ├── webcam_mode.py          # Live webcam detection
│   ├── zone_drawer.py          # Define custom detection zones
│   ├── styles.css              # UI styling
│
├── results/
│   ├── analyzed_traffic_video1.mp4   # Example processed video
│   └── single_video_report.json      # JSON report summary
│
├── uploads/                  # Uploaded traffic videos
│
├── traffic_detector.py        # Core YOLOv8 detection logic
├── webcam_detection.py        # Standalone webcam script
├── run_detection.py           # Command-line detection runner
├── yolov8n.pt                 # YOLOv8 model weights
├── config.json                # Detection and threshold settings
├── zones_config.json          # Zone configuration
├── requirements.txt           # Dependencies
└── README.md
```

---

## Installation  

### Step 1: Clone the Repository  
```bash
git clone https://github.com/MoraisMNS/traffic-congestion-detection-computer-vision.git
cd traffic-congestion-detection
```

### Step 2: Create and Activate Virtual Environment  
```bash
python -m venv tr_env
source tr_env/bin/activate      # Linux/macOS
tr_env\Scripts\activate       # Windows
```

### Step 3: Install Required Packages  
```bash
pip install -r requirements.txt
```

Example dependencies:
```
streamlit
flask
flask-cors
opencv-python
ultralytics
numpy
pandas
```

---

## Running the Project  

### To Start the Streamlit Frontend  
```bash
cd frontend
streamlit run app.py
```

The application will open at:  
👉 **http://localhost:8501**

### Usage  
- **Upload a traffic video** or select **Webcam Mode** from the interface.  
- The system will analyze the feed, detect vehicles, and display the congestion level in real-time.  
- Processed videos and reports will be saved automatically in the **results/** folder.  

---

## Notes  
- Modify `config.json` to adjust detection thresholds or model confidence values.  
- The YOLOv8 model file (`yolov8n.pt`) should be placed in the root directory.  
- Ensure your webcam is connected properly when using real-time mode.  

---

This simplified version focuses on ease of setup and execution, making it ideal for academic project documentation or demonstration purposes.
