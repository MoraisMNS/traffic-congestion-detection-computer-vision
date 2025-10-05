# frontend/main_frontend.py
from frontend.single_video_mode import SingleVideoMode
from frontend.batch_processing_mode import BatchProcessingMode
from frontend.webcam_mode import WebcamMode
from frontend.analytics_dashboard_mode import AnalyticsDashboardMode

class MainFrontend:
    def __init__(self):
        self.single_video = SingleVideoMode()
        self.batch = BatchProcessingMode()
        self.webcam = WebcamMode()
        self.dashboard = AnalyticsDashboardMode

    def run(self, mode, **kwargs):
        if mode == "single":
            return self.single_video.run(**kwargs)
        elif mode == "batch":
            return self.batch.run(**kwargs)
        elif mode == "webcam":
            return self.webcam.run()
        elif mode == "dashboard":
            return self.dashboard(kwargs["reports_folder"]).run()
        else:
            raise ValueError("Invalid mode selected")

if __name__ == "__main__":
    # Example usage
    frontend = MainFrontend()
    # Single video
    # frontend.run("single", video_path="traffic.mp4", output_path="result.mp4", display=True)
    # Batch
    # frontend.run("batch", input_folder="videos", output_folder="results", analyze=True)
    # Webcam
    # frontend.run("webcam")
    # Dashboard
    # frontend.run("dashboard", reports_folder="results")
