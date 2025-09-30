"""
Advanced Analytics and Visualization Dashboard
Creates detailed plots and heatmaps from traffic analysis data
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.patches as patches
from typing import List, Dict


class TrafficAnalyticsDashboard:
    """Generate comprehensive analytics and visualizations"""
    
    def __init__(self, reports_folder: str):
        self.reports_folder = reports_folder
        self.reports = self.load_reports()
        
    def load_reports(self) -> List[Dict]:
        """Load all JSON reports from folder"""
        reports = []
        for file_path in Path(self.reports_folder).glob('report_*.json'):
            with open(file_path, 'r') as f:
                report = json.load(f)
                report['filename'] = file_path.name
                reports.append(report)
        return reports
    
    def plot_congestion_timeline(self, output_path: str = 'congestion_timeline.png'):
        """Plot congestion levels over time"""
        if not self.reports:
            print("No reports found!")
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Organize data by zone
        zone_data = {}
        for report in self.reports:
            for zone in report.get('zones', []):
                zone_name = zone['name']
                if zone_name not in zone_data:
                    zone_data[zone_name] = {
                        'scores': [],
                        'timestamps': []
                    }
                zone_data[zone_name]['scores'].append(zone['congestion_score'])
                zone_data[zone_name]['timestamps'].append(report.get('timestamp', ''))
        
        # Plot each zone
        for zone_name, data in zone_data.items():
            x = range(len(data['scores']))
            ax.plot(x, data['scores'], marker='o', label=zone_name, linewidth=2)
        
        # Add threshold lines
        ax.axhline(y=0.3, color='g', linestyle='--', alpha=0.5, label='Low threshold')
        ax.axhline(y=0.6, color='y', linestyle='--', alpha=0.5, label='Moderate threshold')
        ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='High threshold')
        
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Congestion Score', fontsize=12)
        ax.set_title('Traffic Congestion Timeline', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Timeline plot saved: {output_path}")
        plt.close()
    
    def plot_vehicle_distribution(self, output_path: str = 'vehicle_distribution.png'):
        """Plot vehicle count distribution across zones"""
        if not self.reports:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Collect data
        zone_vehicles = {}
        for report in self.reports:
            for zone in report.get('zones', []):
                zone_name = zone['name']
                if zone_name not in zone_vehicles:
                    zone_vehicles[zone_name] = []
                zone_vehicles[zone_name].append(zone['vehicle_count'])
        
        # Bar chart - Average vehicles per zone
        zones = list(zone_vehicles.keys())
        avg_counts = [np.mean(counts) for counts in zone_vehicles.values()]
        
        bars = ax1.bar(zones, avg_counts, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        ax1.set_ylabel('Average Vehicle Count', fontsize=12)
        ax1.set_title('Average Vehicles per Zone', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontweight='bold')
        
        # Box plot - Vehicle count distribution
        ax2.boxplot(zone_vehicles.values(), labels=zones)
        ax2.set_ylabel('Vehicle Count', fontsize=12)
        ax2.set_title('Vehicle Count Distribution', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved: {output_path}")
        plt.close()
    
    def plot_congestion_heatmap(self, output_path: str = 'congestion_heatmap.png'):
        """Create a heatmap of congestion levels"""
        if not self.reports:
            return
        
        # Prepare data matrix
        zone_names = []
        congestion_matrix = []
        
        for i, report in enumerate(self.reports):
            if i == 0:
                zone_names = [zone['name'] for zone in report.get('zones', [])]
            
            scores = [zone['congestion_score'] for zone in report.get('zones', [])]
            congestion_matrix.append(scores)
        
        congestion_matrix = np.array(congestion_matrix).T
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        im = ax.imshow(congestion_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(self.reports)))
        ax.set_yticks(np.arange(len(zone_names)))
        ax.set_xticklabels([f"T{i+1}" for i in range(len(self.reports))])
        ax.set_yticklabels(zone_names)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Congestion Score', rotation=270, labelpad=20, fontsize=12)
        
        # Add text annotations
        for i in range(len(zone_names)):
            for j in range(len(self.reports)):
                text = ax.text(j, i, f'{congestion_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Traffic Congestion Heatmap', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Zone', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved: {output_path}")
        plt.close()
    
    def plot_congestion_levels_pie(self, output_path: str = 'congestion_pie.png'):
        """Plot pie chart of congestion level distribution"""
        if not self.reports:
            return
        
        levels = {'LOW': 0, 'MODERATE': 0, 'HIGH': 0, 'SEVERE': 0}
        
        for report in self.reports:
            for zone in report.get('zones', []):
                level = zone['congestion_level']
                levels[level] += 1
        
        # Filter out zero values
        labels = [k for k, v in levels.items() if v > 0]
        sizes = [v for v in levels.values() if v > 0]
        colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        explode = [0.05 if label == 'SEVERE' else 0 for label in labels]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                           colors=colors, autopct='%1.1f%%',
                                           shadow=True, startangle=90)
        
        # Beautify text
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
        
        ax.set_title('Congestion Level Distribution', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Pie chart saved: {output_path}")
        plt.close()
    
    def generate_summary_report(self, output_path: str = 'summary_report.txt'):
        """Generate a text summary report"""
        if not self.reports:
            return
        
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TRAFFIC CONGESTION ANALYSIS SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Analysis Periods: {len(self.reports)}\n\n")
            
            # Overall statistics
            total_vehicles = 0
            congestion_counts = {'LOW': 0, 'MODERATE': 0, 'HIGH': 0, 'SEVERE': 0}
            zone_stats = {}
            
            for report in self.reports:
                for zone in report.get('zones', []):
                    zone_name = zone['name']
                    total_vehicles += zone['vehicle_count']
                    congestion_counts[zone['congestion_level']] += 1
                    
                    if zone_name not in zone_stats:
                        zone_stats[zone_name] = {
                            'vehicles': [],
                            'scores': [],
                            'levels': []
                        }
                    
                    zone_stats[zone_name]['vehicles'].append(zone['vehicle_count'])
                    zone_stats[zone_name]['scores'].append(zone['congestion_score'])
                    zone_stats[zone_name]['levels'].append(zone['congestion_level'])
            
            f.write("-"*70 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Total Vehicles Detected: {total_vehicles}\n")
            f.write(f"Average Vehicles per Period: {total_vehicles/len(self.reports):.1f}\n\n")
            
            f.write("Congestion Level Distribution:\n")
            for level, count in congestion_counts.items():
                percentage = (count / sum(congestion_counts.values())) * 100
                f.write(f"  {level:12s}: {count:3d} ({percentage:5.1f}%)\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("ZONE-WISE ANALYSIS\n")
            f.write("-"*70 + "\n\n")
            
            for zone_name, stats in zone_stats.items():
                f.write(f"{zone_name}:\n")
                f.write(f"  Avg Vehicles: {np.mean(stats['vehicles']):.1f}\n")
                f.write(f"  Max Vehicles: {np.max(stats['vehicles'])}\n")
                f.write(f"  Min Vehicles: {np.min(stats['vehicles'])}\n")
                f.write(f"  Avg Congestion Score: {np.mean(stats['scores']):.3f}\n")
                f.write(f"  Peak Congestion: {np.max(stats['scores']):.3f}\n")
                
                # Most common congestion level
                from collections import Counter
                most_common = Counter(stats['levels']).most_common(1)[0]
                f.write(f"  Most Common Level: {most_common[0]} ({most_common[1]} times)\n")
                f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        print(f"Summary report saved: {output_path}")
    
    def generate_all_visualizations(self, output_folder: str = 'analytics'):
        """Generate all visualizations"""
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating analytics visualizations...")
        print("-" * 50)
        
        self.plot_congestion_timeline(f'{output_folder}/congestion_timeline.png')
        self.plot_vehicle_distribution(f'{output_folder}/vehicle_distribution.png')
        self.plot_congestion_heatmap(f'{output_folder}/congestion_heatmap.png')
        self.plot_congestion_levels_pie(f'{output_folder}/congestion_pie.png')
        self.generate_summary_report(f'{output_folder}/summary_report.txt')
        
        print("-" * 50)
        print(f"✓ All visualizations generated in '{output_folder}' folder!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate traffic analytics dashboard')
    parser.add_argument('--reports', '-r', required=True, 
                       help='Folder containing JSON reports')
    parser.add_argument('--output', '-o', default='analytics',
                       help='Output folder for visualizations')
    
    args = parser.parse_args()
    
    dashboard = TrafficAnalyticsDashboard(args.reports)
    dashboard.generate_all_visualizations(args.output)