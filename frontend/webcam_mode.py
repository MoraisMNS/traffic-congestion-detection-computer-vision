# frontend/webcam_mode.py
from traffic_detector import TrafficCongestionDetector
import cv2, json, logging

logger = logging.getLogger(__name__)

class WebcamMode:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.4, camera_id=0):
        self.detector = TrafficCongestionDetector(model_path, conf_threshold)
        self.camera_id = camera_id

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            logger.error("Failed to open webcam")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.detector.setup_default_zones(width, height)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = self.detector.analyze_frame(frame)
            cv2.imshow("Webcam Mode", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite(f"screenshot_{frame_count}.jpg", annotated)
            elif key == ord("r"):
                with open(f"report_{frame_count}.json", "w") as f:
                    json.dump(self.detector.get_report(), f, indent=2)

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
