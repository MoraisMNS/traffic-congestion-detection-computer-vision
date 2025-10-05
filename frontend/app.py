import sys, os
import streamlit as st

# --- Path Fix ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frontend.main_frontend import MainFrontend
from frontend.zone_drawer import ZoneDrawer  # ✅ Import the new class

# --- Streamlit Page Settings ---
st.set_page_config(
    page_title="Traffic Congestion Detection",
    layout="wide",
    page_icon="🚦"
)

# --- Load external CSS file ---
css_path = os.path.join(os.path.dirname(__file__), "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Sidebar Profile Section ---
st.sidebar.markdown("""
<div class="profile-card">
    <img src="https://cdn-icons-png.flaticon.com/512/219/219969.png" class="avatar">
    <h3>Ahamed Minhaj</h3>
    <p>Traffic Monitoring Dashboard</p>
</div>
""", unsafe_allow_html=True)

# --- Initialize session state for selected mode ---
if "mode" not in st.session_state:
    st.session_state.mode = "Single Video"  # Default mode

def set_mode(selected_mode):
    st.session_state.mode = selected_mode

# --- Sidebar Navigation Buttons ---
st.sidebar.markdown('<div class="sidebar-menu">', unsafe_allow_html=True)

if st.sidebar.button("🎥  Single Video Mode", key="single_btn"):
    set_mode("Single Video")
# if st.sidebar.button("📂  Batch Processing", key="batch_btn"):
#     set_mode("Batch Processing")
if st.sidebar.button("📡  Webcam Mode", key="webcam_btn"):
    set_mode("Webcam")
# if st.sidebar.button("📊  Analytics Dashboard", key="dashboard_btn"):
#     set_mode("Analytics Dashboard")
if st.sidebar.button("🗺️  Zone Drawer", key="zone_draw_btn"):  # ✅ NEW BUTTON
    set_mode("Zone Drawer")

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# --- Button Styling via JS ---
st.markdown("""
<script>
const buttons = window.parent.document.querySelectorAll('.stButton > button');
buttons.forEach(btn => {
    btn.style.width = '100%';
    btn.style.borderRadius = '10px';
});
</script>
""", unsafe_allow_html=True)

# --- Initialize Controller ---
frontend = MainFrontend()

# --- Main Content Header ---
st.title("🚦 Traffic Congestion Detection System")
st.markdown("Analyze traffic congestion in videos, live webcam feeds, or through advanced analytics dashboards.")

# --- MODE HANDLING ---
mode = st.session_state.mode

# 1️⃣ Single Video Mode
if mode == "Single Video":
    st.header("🎥 Single Video Mode")
    uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file and st.button("Run Analysis"):
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        input_path = os.path.join("uploads", uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        output_path = os.path.join("results", f"analyzed_{uploaded_file.name}")
        report = frontend.run("single", video_path=input_path, output_path=output_path)

        st.success("✅ Analysis complete!")
        st.json(report)

        # if os.path.exists(output_path):
        #     st.video(output_path)

# # 2️⃣ Batch Mode
# elif mode == "Batch Processing":
#     st.header("📂 Batch Processing Mode")
#     input_folder = st.text_input("Input Folder Path", "uploads")
#     output_folder = st.text_input("Output Folder Path", "results")
#     analyze = st.checkbox("Generate Comparative Analysis", value=True)

#     if st.button("Run Batch Processing"):
#         frontend.run("batch", input_folder=input_folder, output_folder=output_folder, analyze=analyze)
#         st.success("✅ Batch Processing Complete!")
#         st.info(f"Results saved in `{output_folder}` folder.")

# 3️⃣ Webcam Mode
elif mode == "Webcam":
    st.header("📡 Webcam Mode")
    st.info("This will open a live detection window. Press 'q' to exit webcam feed.")
    if st.button("Start Webcam Detection"):
        frontend.run("webcam")
        st.success("✅ Webcam mode ended successfully.")

# # 4️⃣ Analytics Dashboard
# elif mode == "Analytics Dashboard":
#     st.header("📊 Analytics Dashboard Mode")
#     reports_folder = st.text_input("Reports Folder Path", "results")

#     if st.button("Generate Dashboard"):
#         frontend.run("dashboard", reports_folder=reports_folder)
#         st.success("✅ Dashboard Generated Successfully!")

#         for img, caption in [
#             ("congestion_timeline.png", "Congestion Timeline"),
#             ("vehicle_distribution.png", "Vehicle Distribution"),
#             ("congestion_heatmap.png", "Congestion Heatmap"),
#             ("congestion_pie.png", "Congestion Levels Pie"),
#         ]:
#             path = os.path.join("analytics", img)
#             if os.path.exists(path):
#                 st.image(path, caption=caption)

#         txt_path = os.path.join("analytics", "summary_report.txt")
#         if os.path.exists(txt_path):
#             st.text(open(txt_path).read())


# 5️⃣ Zone Drawer Mode (UPDATED)
elif mode == "Zone Drawer":
    st.header("🗺️ Zone Drawer — Define or Load Custom Congestion Zones")
    st.info("Manually define congestion zones by entering coordinates or load previously saved ones.")

    zone_drawer = ZoneDrawer()

    # --- Choose Action ---
    st.markdown("### ⚙️ Zone Configuration Options")
    action = st.radio(
        "Select an option:",
        ["➕ Define New Zones", "📂 Load Saved Zones"],
        horizontal=True
    )

    # --- Define New Zones ---
    if action == "➕ Define New Zones":
        zones = zone_drawer.input_zones()
        if zones:
            st.success("✅ Zones defined and saved successfully!")
            st.session_state["custom_zones"] = zones
            st.json(zones)

    # --- Load Existing Zones ---
    elif action == "📂 Load Saved Zones":
        zones = zone_drawer.load_zones_from_file()
        if zones:
            st.success("✅ Loaded saved zones from 'zones_config.json'")
            st.session_state["custom_zones"] = zones
            st.json(zones)
        else:
            st.warning("⚠️ No saved zones found. Define new ones first.")
