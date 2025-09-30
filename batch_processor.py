"""
Batch Processing Script for Multiple Traffic Videos
"""

import os
import json
from pathlib import Path
from traffic_detector import TrafficCongestionDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_video_batch(input_folder: str, output_folder: str):
    """Process all videos in a folder"""
    
    # Create output directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Initialize detector once for all videos
    detector = TrafficCongestionDetector(
        model_path="yolov8n.pt",
        conf_threshold=0.3
    )
    
    # Get all video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')
    video_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(video_extensions)
    ]
    
    if not video_files:
        logger.warning(f"No video files found in {input_folder}")
        return
    
    logger.info(f"Found {len(video_files)} video(s) to process")
    
    # Process each video
    all_reports = []
    
    for i, video_file in enumerate(video_files, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing [{i}/{len(video_files)}]: {video_file}")
        logger.info(f"{'='*60}\n")
        
        input_path = os.path.join(input_folder, video_file)
        output_filename = f"analyzed_{Path(video_file).stem}.mp4"
        output_path = os.path.join(output_folder, output_filename)
        
        try:
            # Process video
            report = detector.process_video(
                video_path=input_path,
                output_path=output_path,
                display=False  # Don't display for batch processing
            )
            
            # Add metadata
            report['source_file'] = video_file
            report['output_file'] = output_filename
            all_reports.append(report)
            
            # Save individual report
            report_path = os.path.join(
                output_folder,
                f"report_{Path(video_file).stem}.json"
            )
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"✓ Completed: {video_file}")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {video_file}: {e}")
            continue
    
    # Save summary report
    summary_path = os.path.join(output_folder, "batch_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'total_videos': len(video_files),
            'processed': len(all_reports),
            'reports': all_reports
        }, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Batch processing complete!")
    logger.info(f"Processed: {len(all_reports)}/{len(video_files)} videos")
    logger.info(f"Results saved to: {output_folder}")
    logger.info(f"Summary report: {summary_path}")
    logger.info(f"{'='*60}\n")


def generate_comparative_analysis(reports_folder: str):
    """Generate comparative analysis from multiple reports"""
    
    report_files = [
        f for f in os.listdir(reports_folder)
        if f.startswith('report_') and f.endswith('.json')
    ]
    
    if not report_files:
        logger.warning("No report files found")
        return
    
    analysis = {
        'total_videos_analyzed': len(report_files),
        'zone_statistics': {},
        'congestion_patterns': {
            'LOW': 0,
            'MODERATE': 0,
            'HIGH': 0,
            'SEVERE': 0
        }
    }
    
    for report_file in report_files:
        with open(os.path.join(reports_folder, report_file), 'r') as f:
            report = json.load(f)
        
        for zone in report.get('zones', []):
            zone_name = zone['name']
            if zone_name not in analysis['zone_statistics']:
                analysis['zone_statistics'][zone_name] = {
                    'total_vehicles': 0,
                    'avg_congestion_score': [],
                    'congestion_levels': {'LOW': 0, 'MODERATE': 0, 'HIGH': 0, 'SEVERE': 0}
                }
            
            stats = analysis['zone_statistics'][zone_name]
            stats['total_vehicles'] += zone['vehicle_count']
            stats['avg_congestion_score'].append(zone['congestion_score'])
            stats['congestion_levels'][zone['congestion_level']] += 1
            
            analysis['congestion_patterns'][zone['congestion_level']] += 1
    
    # Calculate averages
    for zone_name, stats in analysis['zone_statistics'].items():
        if stats['avg_congestion_score']:
            stats['avg_congestion_score'] = sum(stats['avg_congestion_score']) / len(stats['avg_congestion_score'])
    
    # Save comparative analysis
    output_path = os.path.join(reports_folder, 'comparative_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"Comparative analysis saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total Videos Analyzed: {analysis['total_videos_analyzed']}")
    print("\nCongestion Pattern Distribution:")
    for level, count in analysis['congestion_patterns'].items():
        print(f"  {level}: {count} instances")
    print("\nZone Statistics:")
    for zone_name, stats in analysis['zone_statistics'].items():
        print(f"\n  {zone_name}:")
        print(f"    Total Vehicles: {stats['total_vehicles']}")
        print(f"    Avg Congestion Score: {stats['avg_congestion_score']:.3f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process traffic videos')
    parser.add_argument('--input', '-i', required=True, help='Input folder with videos')
    parser.add_argument('--output', '-o', required=True, help='Output folder for results')
    parser.add_argument('--analyze', '-a', action='store_true', 
                       help='Generate comparative analysis')
    
    args = parser.parse_args()
    
    if args.analyze:
        generate_comparative_analysis(args.output)
    else:
        process_video_batch(args.input, args.output)