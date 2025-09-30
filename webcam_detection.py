"""
Real-time Traffic Congestion Detection using Webcam
"""

import cv2
from traffic_detector import TrafficCongestionDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_webcam_detection(camera_id: int = 0):
    """Run real-time detection on webcam feed"""
    
    # Initialize detector
    detector = TrafficCongestionDetector(
        model_path="yolov8n.pt",
        conf_threshold=0.4
    )
    
    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return
    
    # Get frame dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup zones
    detector.setup_default_zones(width, height)
    
    logger.info("Press 'q' to quit, 's' to save screenshot, 'r' to save report")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame")
                break
            
            # Process frame
            annotated_frame = detector.analyze_frame(frame)
            
            # Display
            cv2.imshow('Real-time Traffic Detection', annotated_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(filename, annotated_frame)
                logger.info(f"Screenshot saved: {filename}")
            elif key == ord('r'):
                report = detector.get_report()
                import json
                with open(f'report_{frame_count}.json', 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Report saved: report_{frame_count}.json")
            
            frame_count += 1
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam detection stopped")


if __name__ == "__main__":
    run_webcam_detection(camera_id=0)