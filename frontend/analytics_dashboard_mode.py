# frontend/analytics_dashboard_mode.py
from analytics_dashboard import TrafficAnalyticsDashboard

class AnalyticsDashboardMode:
    def __init__(self, reports_folder, output_folder="analytics"):
        self.dashboard = TrafficAnalyticsDashboard(reports_folder)
        self.output_folder = output_folder

    def run(self):
        self.dashboard.generate_all_visualizations(self.output_folder)
