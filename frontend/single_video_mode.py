from traffic_detector import TrafficCongestionDetector
import json
import cv2
import streamlit as st
import os
import time
import pandas as pd
import subprocess


class SingleVideoMode:
    """
    Handles single video processing mode with live YOLO detection.
    Displays results cleanly in Streamlit with one final analyzed video preview.
    """

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.3):
        self.detector = TrafficCongestionDetector(model_path, conf_threshold)

    def load_saved_zones(self):
        """Load zones from session or saved config file."""
        if "custom_zones" in st.session_state and st.session_state["custom_zones"]:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #dbeafe 0%, #e0f2fe 100%); 
                        padding: 1rem; border-radius: 12px; border-left: 4px solid #2563eb;
                        color: #1e40af; font-weight: 500; margin: 1rem 0;'>
                ✅ Using custom zones from session.
            </div>
            """, unsafe_allow_html=True)
            return st.session_state["custom_zones"]

        if os.path.exists("zones_config.json"):
            try:
                with open("zones_config.json", "r") as f:
                    return json.load(f)
            except Exception as e:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                            padding: 1rem; border-radius: 12px; border-left: 4px solid #f59e0b;
                            color: #78350f; font-weight: 500; margin: 1rem 0;'>
                    ⚠️ Error loading saved zones: {e}
                </div>
                """, unsafe_allow_html=True)
        return None

    def apply_zones(self, width, height):
        """Applies zones to detector."""
        custom_zones = self.load_saved_zones()
        if custom_zones:
            for z in custom_zones:
                self.detector.add_zone(z["name"], z["points"], z["capacity"])
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #dcfce7 0%, #d1fae5 100%); 
                        padding: 1rem; border-radius: 12px; border-left: 4px solid #10b981;
                        color: #065f46; font-weight: 500; margin: 1rem 0;'>
                ✅ Applied {len(custom_zones)} custom zones.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                        padding: 1rem; border-radius: 12px; border-left: 4px solid #f59e0b;
                        color: #78350f; font-weight: 500; margin: 1rem 0;'>
                ⚙️ Using default zones.
            </div>
            """, unsafe_allow_html=True)
            self.detector.setup_default_zones(width, height)

    def save_report(self, report, video_name):
        """Save analyzed report for future use."""
        os.makedirs("results/analysis_reports", exist_ok=True)
        path = os.path.join("results/analysis_reports", f"report_{video_name}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        return path

    def reencode_video(self, input_path):
        """Re-encode to browser-friendly H.264 if needed."""
        output_path = input_path.replace(".mp4", "_web.mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path, "-vcodec", "libx264", "-acodec", "aac", output_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            return output_path
        except Exception:
            return input_path  # fallback

    def display_report(self, report):
        """Display analysis summary table with styled metrics."""
        st.markdown("""
        <h3 style='color: #2563eb; font-weight: 700; margin-top: 2rem; padding-bottom: 0.5rem; 
                   border-bottom: 3px solid #dbeafe;'>
            📊 Traffic Analysis Summary
        </h3>
        """, unsafe_allow_html=True)
        
        total_zones = len(report.get("zones", []))
        total_frames = report.get("total_frames", 0)
        runtime = report.get("runtime_seconds", 0)

        # Styled metric cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; 
                        border: 2px solid #dbeafe; box-shadow: 0 4px 6px rgba(37, 99, 235, 0.15);
                        text-align: center;'>
                <div style='color: #64748b; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;'>
                    ZONES
                </div>
                <div style='color: #2563eb; font-size: 2rem; font-weight: 700;'>
                    {total_zones}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; 
                        border: 2px solid #dbeafe; box-shadow: 0 4px 6px rgba(37, 99, 235, 0.15);
                        text-align: center;'>
                <div style='color: #64748b; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;'>
                    FRAMES
                </div>
                <div style='color: #2563eb; font-size: 2rem; font-weight: 700;'>
                    {total_frames:,}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; 
                        border: 2px solid #dbeafe; box-shadow: 0 4px 6px rgba(37, 99, 235, 0.15);
                        text-align: center;'>
                <div style='color: #64748b; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;'>
                    RUNTIME
                </div>
                <div style='color: #2563eb; font-size: 2rem; font-weight: 700;'>
                    {runtime:.1f}s
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if report.get("zones"):
            df = pd.DataFrame(report["zones"])
            df.rename(columns={
                "name": "Zone Name",
                "congestion_level": "Congestion Level",
                "congestion_score": "Score",
                "vehicle_count": "Vehicles",
                "capacity": "Capacity"
            }, inplace=True)
            
            # Style the dataframe
            st.markdown("""
            <style>
            .dataframe {
                border-radius: 12px !important;
                overflow: hidden;
                border: 1px solid #e2e8f0;
            }
            .dataframe thead tr th {
                background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
                color: white !important;
                font-weight: 600 !important;
                padding: 12px !important;
            }
            .dataframe tbody tr:hover {
                background-color: #dbeafe !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.dataframe(df.style.highlight_max(subset=["Score"], color="#fecaca"), use_container_width=True)

    def run(self, video_path, output_path="output_analyzed.mp4"):
        """Process video and show final analyzed output."""
        st.markdown("""
        <h2 style='color: #2563eb; font-weight: 700; margin-top: 2rem; padding-bottom: 0.5rem; 
                   border-bottom: 3px solid #dbeafe;'>
            🎥 Live Video Analysis
        </h2>
        """, unsafe_allow_html=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.markdown("""
            <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                        padding: 1rem; border-radius: 12px; border-left: 4px solid #dc2626;
                        color: #7f1d1d; font-weight: 500; margin: 1rem 0;'>
                ❌ Failed to open video.
            </div>
            """, unsafe_allow_html=True)
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.apply_zones(width, height)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Browser-friendly codec
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        stframe = st.empty()
        progress = st.progress(0)
        progress_text = st.empty()

        idx = 0
        start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            annotated = self.detector.analyze_frame(frame)
            writer.write(annotated)

            # Live display
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            stframe.image(rgb, channels="RGB", use_container_width=True)

            progress.progress(idx / total_frames)
            progress_text.markdown(f"""
            <div style='text-align: center; color: #2563eb; font-weight: 600; padding: 0.5rem;'>
                Processing frame {idx:,}/{total_frames:,}
            </div>
            """, unsafe_allow_html=True)

        cap.release()
        writer.release()

        elapsed = time.time() - start
        report = self.detector.get_report()
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        report_path = self.save_report(report, video_name)

        # 🔁 Re-encode video for playback
        final_video = self.reencode_video(output_path)
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #dcfce7 0%, #d1fae5 100%); 
                    padding: 1rem; border-radius: 12px; border-left: 4px solid #10b981;
                    color: #065f46; font-weight: 500; margin: 1rem 0;'>
            ✅ Analysis complete in {elapsed:.1f}s
        </div>
        """, unsafe_allow_html=True)

        # --- Summary ---
        self.display_report(report)

        # --- Single video preview (no duplicates) ---
        st.markdown("<hr style='border: none; height: 2px; background: linear-gradient(90deg, transparent, #2563eb, transparent); margin: 2rem 0;'>", unsafe_allow_html=True)
        
        st.markdown("""
        <h2 style='color: #2563eb; font-weight: 700; margin-top: 2rem; padding-bottom: 0.5rem; 
                   border-bottom: 3px solid #dbeafe;'>
            📹 Analyzed Video Preview
        </h2>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #dbeafe 0%, #e0f2fe 100%); 
                    padding: 1rem; border-radius: 12px; border-left: 4px solid #2563eb;
                    color: #1e40af; font-weight: 500; margin: 1rem 0;'>
            Play the analyzed video below to review congestion zones and activity.
        </div>
        """, unsafe_allow_html=True)

        video_placeholder = st.empty()
        if os.path.exists(final_video):
            with open(final_video, "rb") as v:
                video_bytes = v.read()
                video_placeholder.video(video_bytes)
        else:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                        padding: 1rem; border-radius: 12px; border-left: 4px solid #f59e0b;
                        color: #78350f; font-weight: 500; margin: 1rem 0;'>
                ⚠️ Could not display analyzed video.
            </div>
            """, unsafe_allow_html=True)

        # --- Download buttons ---
        col1, col2 = st.columns(2)
        with open(final_video, "rb") as vfile:
            col1.download_button(
                "⬇️ Download Analyzed Video",
                data=vfile,
                file_name=os.path.basename(final_video),
                mime="video/mp4"
            )
        col2.download_button(
            "💾 Download Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name=os.path.basename(report_path),
            mime="application/json"
        )

        st.markdown(f"""
        <div style='text-align: center; color: #64748b; font-size: 0.9rem; margin-top: 1rem;'>
            📁 Report auto-saved to: <code style='background: #dbeafe; color: #1e40af; 
            padding: 0.2rem 0.5rem; border-radius: 6px;'>{report_path}</code>
        </div>
        """, unsafe_allow_html=True)
        
        return report