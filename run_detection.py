from traffic_detector import TrafficCongestionDetector

# Initialize detector
detector = TrafficCongestionDetector(
    model_path="yolov8n.pt",
    conf_threshold=0.3
)

# Process video
report = detector.process_video(
    video_path="traffic_video2.mp4",  
    output_path="output_analyzed.mp4",
    display=True
)

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
print(f"Total frames: {report['total_frames']}")
for zone in report['zones']:
    print(f"{zone['name']}: {zone['congestion_level']}")