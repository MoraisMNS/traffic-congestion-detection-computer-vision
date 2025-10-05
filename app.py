"""
Traffic Congestion Detection Dashboard (Streamlit)
Author: AI Assistant
Description: Web-based UI for video upload, batch processing, webcam detection, and analytics
"""

"""
Traffic Congestion Detection Dashboard (Streamlit)
"""

import streamlit as st
import os
import json
import tempfile
import cv2
from pathlib import Path

# Import your modules based on project files
from traffic_detector import TrafficCongestionDetector
from batch_processor import process_video_batch, generate_comparative_analysis
from analytics_dashboard import TrafficAnalyticsDashboard  # dashboard module
# App title
st.set_page_config(page_title="Traffic Congestion Detection", layout="wide")
st.title("🚦 Traffic Congestion Detection System")
st.markdown("---")

# Sidebar navigation
menu = st.sidebar.radio(
    "Select Mode",
    ["Single Video Analysis", "Batch Processing", "Webcam Detection", "Analytics Dashboard"]
)

# ===========================
# SINGLE VIDEO MODE
# ===========================
if menu == "Single Video Analysis":
    st.header("🎥 Single Video Traffic Analysis")

    uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file:
        # Save temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        st.video(tfile.name)

        if st.button("Run Analysis"):
            detector = TrafficCongestionDetector(model_path="yolov8n.pt", conf_threshold=0.3)
            
            output_path = "analyzed_single.mp4"
            report = detector.process_video(
                video_path=tfile.name,
                output_path=output_path,
                display=False
            )
            
            # Save report
            with open("report_single.json", "w") as f:
                json.dump(report, f, indent=2)

            st.success("✅ Analysis Complete!")
            st.video(output_path)
            st.json(report)


# ===========================
# BATCH PROCESSING MODE
# ===========================
elif menu == "Batch Processing":
    st.header("📂 Batch Process Multiple Videos")

    input_folder = st.text_input("Input folder path (with videos):", "videos")
    output_folder = st.text_input("Output folder path:", "results")

    if st.button("Run Batch Processing"):
        process_video_batch(input_folder, output_folder)
        st.success(f"✅ Batch Processing Complete! Results saved in {output_folder}")

    if st.button("Generate Comparative Analysis"):
        generate_comparative_analysis(output_folder)
        with open(os.path.join(output_folder, "comparative_analysis.json"), "r") as f:
            analysis = json.load(f)
        st.json(analysis)


# ===========================
# WEBCAM MODE
# ===========================
elif menu == "Webcam Detection":
    st.header("📷 Real-time Webcam Traffic Detection")

    st.info("Click below to start webcam detection (Press 'q' in the popup window to quit).")

    if st.button("Start Webcam"):
        detector = TrafficCongestionDetector(model_path="yolov8n.pt", conf_threshold=0.4)

        cap = cv2.VideoCapture(0)
        detector.setup_default_zones(int(cap.get(3)), int(cap.get(4)))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame = detector.analyze_frame(frame)

            # Show in Streamlit
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        st.success("✅ Webcam Detection Stopped")


# ===========================
# ANALYTICS DASHBOARD MODE
# ===========================
elif menu == "Analytics Dashboard":
    st.header("📊 Traffic Analytics and Visualization")

    reports_folder = st.text_input("Reports folder path:", "results")

    if st.button("Generate Dashboard"):
        dashboard = TrafficAnalyticsDashboard(reports_folder)
        dashboard.generate_all_visualizations("analytics")

        st.success("✅ Dashboard Generated!")

        # Show generated images
        st.image("analytics/congestion_timeline.png", caption="Congestion Timeline")
        st.image("analytics/vehicle_distribution.png", caption="Vehicle Distribution")
        st.image("analytics/congestion_heatmap.png", caption="Congestion Heatmap")
        st.image("analytics/congestion_pie.png", caption="Congestion Levels Pie Chart")

        with open("analytics/summary_report.txt", "r") as f:
            st.text(f.read())
