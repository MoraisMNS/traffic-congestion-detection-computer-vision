"""
Professional Traffic Congestion Detection System
Author: Minhaj MHA and Morais MNS
Description: Real-time traffic analysis with vehicle detection, tracking, and congestion metrics
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VehicleTracker:
    """Track vehicles across frames using centroid tracking"""
    
    def __init__(self, max_disappeared: int = 50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.vehicle_speeds = {}
        self.trajectories = defaultdict(lambda: deque(maxlen=30))
        
    def register(self, centroid: Tuple[int, int]) -> int:
        """Register a new object"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.trajectories[self.next_object_id].append(centroid)
        self.next_object_id += 1
        return self.next_object_id - 1
    
    def deregister(self, object_id: int):
        """Remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.trajectories:
            del self.trajectories[object_id]
        if object_id in self.vehicle_speeds:
            del self.vehicle_speeds[object_id]
    
    def update(self, detections: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int]]:
        """Update tracked objects with new detections"""
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(detections):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
        
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(tuple(centroid))
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Compute distance matrix
            D = np.linalg.norm(
                np.array(object_centroids)[:, np.newaxis] - input_centroids,
                axis=2
            )
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if D[row, col] > 50:  # Max distance threshold
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = tuple(input_centroids[col])
                self.disappeared[object_id] = 0
                self.trajectories[object_id].append(tuple(input_centroids[col]))
                
                # Calculate speed
                if len(self.trajectories[object_id]) >= 2:
                    prev_pos = self.trajectories[object_id][-2]
                    curr_pos = self.trajectories[object_id][-1]
                    distance = np.linalg.norm(
                        np.array(curr_pos) - np.array(prev_pos)
                    )
                    self.vehicle_speeds[object_id] = distance
                
                used_rows.add(row)
                used_cols.add(col)
            
            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols
            
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            for col in unused_cols:
                self.register(tuple(input_centroids[col]))
        
        return self.objects
    
    def get_average_speed(self) -> float:
        """Calculate average speed of all tracked vehicles"""
        if not self.vehicle_speeds:
            return 0.0
        return np.mean(list(self.vehicle_speeds.values()))


class CongestionZone:
    """Define and analyze a specific zone for congestion detection"""
    
    def __init__(self, name: str, polygon: np.ndarray, capacity: int = 20):
        self.name = name
        self.polygon = polygon
        self.capacity = capacity
        self.vehicle_count_history = deque(maxlen=30)
        self.congestion_history = deque(maxlen=100)
        
    def is_point_inside(self, point: Tuple[int, int]) -> bool:
        """Check if a point is inside the zone polygon"""
        # Convert to float tuple for OpenCV
        point_float = (float(point[0]), float(point[1]))
        return cv2.pointPolygonTest(self.polygon, point_float, False) >= 0
    
    def update_metrics(self, vehicle_count: int, avg_speed: float):
        """Update zone metrics"""
        self.vehicle_count_history.append(vehicle_count)
        
        # Calculate congestion level
        density_ratio = vehicle_count / self.capacity
        speed_factor = 1.0 - min(avg_speed / 10.0, 1.0)  # Normalize speed
        
        congestion_score = (density_ratio * 0.7) + (speed_factor * 0.3)
        self.congestion_history.append(congestion_score)
    
    def get_congestion_level(self) -> Tuple[str, float]:
        """Get current congestion level"""
        if not self.congestion_history:
            return "UNKNOWN", 0.0
        
        score = self.congestion_history[-1]
        
        if score < 0.3:
            return "LOW", score
        elif score < 0.6:
            return "MODERATE", score
        elif score < 0.8:
            return "HIGH", score
        else:
            return "SEVERE", score
    
    def get_smoothed_count(self) -> int:
        """Get smoothed vehicle count"""
        if not self.vehicle_count_history:
            return 0
        return int(np.mean(self.vehicle_count_history))


class TrafficCongestionDetector:
    """Main traffic congestion detection system"""
    
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.3):
        """Initialize the detector"""
        logger.info("Initializing Traffic Congestion Detector...")
        
        # Load YOLO model
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Vehicle classes (COCO dataset)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        # Initialize tracker
        self.tracker = VehicleTracker(max_disappeared=30)
        
        # Zones
        self.zones = []
        
        # Statistics
        self.stats = {
            'total_vehicles_detected': 0,
            'frame_count': 0,
            'start_time': datetime.now(),
        }
        
        # Visualization colors
        self.congestion_colors = {
            'LOW': (0, 255, 0),      # Green
            'MODERATE': (0, 255, 255), # Yellow
            'HIGH': (0, 165, 255),     # Orange
            'SEVERE': (0, 0, 255),     # Red
            'UNKNOWN': (128, 128, 128) # Gray
        }
        
        logger.info("Detector initialized successfully!")
    
    def add_zone(self, name: str, polygon: List[Tuple[int, int]], capacity: int = 20):
        """Add a congestion detection zone"""
        poly_array = np.array(polygon, dtype=np.int32)
        zone = CongestionZone(name, poly_array, capacity)
        self.zones.append(zone)
        logger.info(f"Added zone: {name} with capacity {capacity}")
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, int, float]]:
        """Detect vehicles in frame"""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    detections.append((x1, y1, x2, y2, cls, conf))
        
        return detections
    
    def analyze_frame(self, frame: np.ndarray) -> np.ndarray:
        """Analyze a single frame"""
        self.stats['frame_count'] += 1
        
        # Detect vehicles
        detections = self.detect_vehicles(frame)
        
        # Extract bounding boxes for tracking
        boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _, _ in detections]
        
        # Update tracker
        tracked_objects = self.tracker.update(boxes)
        
        # Analyze each zone
        for zone in self.zones:
            vehicles_in_zone = 0
            for obj_id, centroid in tracked_objects.items():
                if zone.is_point_inside(centroid):
                    vehicles_in_zone += 1
            
            avg_speed = self.tracker.get_average_speed()
            zone.update_metrics(vehicles_in_zone, avg_speed)
        
        # Draw visualizations
        annotated_frame = self.draw_annotations(frame, detections, tracked_objects)
        
        self.stats['total_vehicles_detected'] = len(tracked_objects)
        
        return annotated_frame
    
    def draw_annotations(self, frame: np.ndarray, detections: List, 
                        tracked_objects: Dict) -> np.ndarray:
        """Draw all annotations on frame"""
        annotated = frame.copy()
        
        # Draw zones
        for zone in self.zones:
            level, score = zone.get_congestion_level()
            color = self.congestion_colors[level]
            
            # Draw zone polygon
            cv2.polylines(annotated, [zone.polygon], True, color, 2)
            
            # Fill zone with transparency
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [zone.polygon], color)
            cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
            
            # Zone label
            x, y = zone.polygon[0]
            cv2.putText(annotated, f"{zone.name}: {level}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
        
        # Draw tracked vehicles
        for obj_id, centroid in tracked_objects.items():
            # Draw trajectory
            if len(self.tracker.trajectories[obj_id]) > 1:
                points = list(self.tracker.trajectories[obj_id])
                for i in range(1, len(points)):
                    cv2.line(annotated, points[i-1], points[i], (255, 0, 255), 1)
            
            # Draw centroid
            cv2.circle(annotated, centroid, 4, (0, 255, 255), -1)
            cv2.putText(annotated, str(obj_id), 
                       (centroid[0] - 10, centroid[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw bounding boxes
        for x1, y1, x2, y2, cls, conf in detections:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{self.class_names[cls]}: {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw dashboard
        self.draw_dashboard(annotated)
        
        return annotated
    
    def draw_dashboard(self, frame: np.ndarray):
        """Draw statistics dashboard on frame"""
        h, w = frame.shape[:2]
        
        # Dashboard background
        dashboard_h = 180
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, dashboard_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_offset = 35
        cv2.putText(frame, "TRAFFIC MONITORING SYSTEM", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset += 30
        cv2.putText(frame, f"Active Vehicles: {self.stats['total_vehicles_detected']}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(frame, f"Avg Speed: {self.tracker.get_average_speed():.1f} px/frame", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        runtime = (datetime.now() - self.stats['start_time']).total_seconds()
        cv2.putText(frame, f"Runtime: {int(runtime)}s", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Zone statistics
        y_offset += 30
        for zone in self.zones:
            level, score = zone.get_congestion_level()
            color = self.congestion_colors[level]
            count = zone.get_smoothed_count()
            cv2.putText(frame, f"{zone.name}: {count} vehicles - {level}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     display: bool = True) -> Dict:
        """Process video file"""
        logger.info(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return {}
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup default zones if none exist
        if not self.zones:
            self.setup_default_zones(width, height)
        
        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_num = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                
                # Process frame
                annotated_frame = self.analyze_frame(frame)
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                # Display
                if display:
                    cv2.imshow('Traffic Congestion Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if frame_num % 30 == 0:
                    logger.info(f"Processed {frame_num}/{total_frames} frames")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        logger.info("Video processing completed!")
        return self.get_report()
    
    def setup_default_zones(self, width: int, height: int):
        """Setup default detection zones"""
        # Zone 1: Upper area
        zone1 = [
            (int(width * 0.1), int(height * 0.2)),
            (int(width * 0.9), int(height * 0.2)),
            (int(width * 0.9), int(height * 0.5)),
            (int(width * 0.1), int(height * 0.5))
        ]
        self.add_zone("Zone 1", zone1, capacity=15)
        
        # Zone 2: Lower area
        zone2 = [
            (int(width * 0.1), int(height * 0.5)),
            (int(width * 0.9), int(height * 0.5)),
            (int(width * 0.9), int(height * 0.8)),
            (int(width * 0.1), int(height * 0.8))
        ]
        self.add_zone("Zone 2", zone2, capacity=20)
    
    def get_report(self) -> Dict:
        """Generate analysis report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_frames': self.stats['frame_count'],
            'runtime_seconds': (datetime.now() - self.stats['start_time']).total_seconds(),
            'zones': []
        }
        
        for zone in self.zones:
            level, score = zone.get_congestion_level()
            report['zones'].append({
                'name': zone.name,
                'congestion_level': level,
                'congestion_score': float(score),
                'vehicle_count': zone.get_smoothed_count(),
                'capacity': zone.capacity
            })
        
        return report


def main():
    """Main execution function"""
    # Initialize detector
    detector = TrafficCongestionDetector(
        model_path="yolov8n.pt",  # Download automatically on first run
        conf_threshold=0.3
    )
    
    # Example: Process a video file
    # Replace with your video path
    video_path = "traffic_video.mp4"
    output_path = "output_traffic_analysis.mp4"
    
    # Custom zones (optional)
    # detector.add_zone("Highway Lane 1", [(100, 100), (500, 100), (500, 300), (100, 300)], capacity=10)
    
    try:
        report = detector.process_video(
            video_path=video_path,
            output_path=output_path,
            display=True
        )
        
        # Save report
        with open('traffic_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Report saved to traffic_report.json")
        print("\n" + "="*50)
        print("ANALYSIS REPORT")
        print("="*50)
        print(json.dumps(report, indent=2))
        
    except FileNotFoundError:
        logger.error(f"Video file not found: {video_path}")
        logger.info("Please provide a valid video file path")
    except Exception as e:
        logger.error(f"Error during processing: {e}")


if __name__ == "__main__":
    main()