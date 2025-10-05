import streamlit as st
import json
import numpy as np
import cv2


class ZoneDrawer:
    """
    Pure configuration-based zone manager.
    Allows users to manually define zones by entering coordinates (x, y).
    Can also save and load zone configurations.
    Includes a coordinate reference frame for better usability.
    """

    def __init__(self):
        self.user_zones = []

    def generate_reference_frame(self, width=800, height=600, grid_step=100):
        """
        Generates a blank frame with a coordinate grid overlay for user reference.
        """
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

        # Draw grid lines
        for x in range(0, width, grid_step):
            cv2.line(frame, (x, 0), (x, height), (200, 200, 200), 1)
            cv2.putText(frame, str(x), (x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        for y in range(0, height, grid_step):
            cv2.line(frame, (0, y), (width, y), (200, 200, 200), 1)
            cv2.putText(frame, str(y), (5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Title
        cv2.putText(frame, "Coordinate Reference Frame (X, Y)", (20, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def show_reference_frame(self):
        """
        Displays the coordinate reference frame inside Streamlit.
        """
        ref_frame = self.generate_reference_frame()
        st.image(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB),
                 caption="📐 Coordinate Reference Frame — Use this to estimate (X, Y) points",
                 use_container_width=True)

    def input_zones(self):
        """
        Displays input fields for manually defining multiple zones.
        Returns a list of zones or defaults if none are defined.
        """
        st.subheader("🗺️ Define Custom Traffic Zones")

        st.info("Enter coordinates manually for each zone below. "
                "Refer to the coordinate frame below to choose appropriate (X, Y) values.")

        # ✅ Show coordinate grid reference first
        self.show_reference_frame()

        # --- Input Fields ---
        num_zones = st.number_input("Number of Zones", min_value=1, max_value=5, value=2, step=1)

        for i in range(num_zones):
            st.markdown(f"### 🧩 Zone {i+1}")

            name = st.text_input(f"Zone {i+1} Name", value=f"Zone {i+1}")
            capacity = st.number_input(
                f"Zone {i+1} Capacity (vehicles)",
                min_value=5, max_value=100, value=20, step=1
            )

            st.markdown("Enter coordinates for the four corners:")

            col1, col2 = st.columns(2)
            with col1:
                x1 = st.number_input(f"Zone {i+1} - Top Left X", min_value=0, max_value=1920, value=100)
                x2 = st.number_input(f"Zone {i+1} - Top Right X", min_value=0, max_value=1920, value=700)
            with col2:
                y1 = st.number_input(f"Zone {i+1} - Top Left Y", min_value=0, max_value=1080, value=100)
                y2 = st.number_input(f"Zone {i+1} - Bottom Right Y", min_value=0, max_value=1080, value=350 + i*100)

            polygon = [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2)
            ]

            self.user_zones.append({
                "name": name,
                "points": polygon,
                "capacity": capacity
            })

        if st.button("💾 Save Zones"):
            if self.user_zones:
                self.save_zones_to_file(self.user_zones)
                st.success("✅ Zones saved successfully to 'zones_config.json'.")
                return self.user_zones
            else:
                st.warning("⚠️ No zones entered. Using default zones.")
                return self.default_zones()

        return None

    def save_zones_to_file(self, zones):
        """Save zones to a local JSON file for persistence."""
        with open("zones_config.json", "w") as f:
            json.dump(zones, f, indent=2)

    def load_zones_from_file(self):
        """Load zones from a saved JSON file."""
        try:
            with open("zones_config.json", "r") as f:
                zones = json.load(f)
                st.success("✅ Loaded saved zones from file.")
                return zones
        except FileNotFoundError:
            st.warning("⚠️ No saved zones found. Using defaults.")
            return self.default_zones()

    def default_zones(self):
        """Return the default predefined two zones."""
        zone1 = [
            (100, 100), (700, 100),
            (700, 350), (100, 350)
        ]
        zone2 = [
            (100, 350), (700, 350),
            (700, 600), (100, 600)
        ]
        return [
            {"name": "Zone 1 - Main Road", "points": zone1, "capacity": 15},
            {"name": "Zone 2 - Intersection", "points": zone2, "capacity": 20}
        ]
