import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from src.inference import load_model, predict_image, predict_video
from src.utils import render_detection_table

# Page Config
st.set_page_config(
    page_title="Traffic Sign Detector Pro",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    local_css("src/style.css")
except Exception:
    pass # Fallback if CSS missing

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/traffic-light.png", width=80)
    st.header("Settings")
    
    model_path = st.text_input("Model Weights", value="models/traffic_sign_best.pt")
    
    st.divider()
    st.subheader("Detection Parameters")
    conf = st.slider("Confidence Threshold", 0.05, 1.0, 0.35, 0.05)
    iou = st.slider("IoU Threshold", 0.1, 0.9, 0.45, 0.05)
    
    with st.expander("🎯 Accuracy & Performance"):
        st.markdown("""
        - **To Detect Well**: The current model is a starter. Use `scripts/train.py` on the GTSRB dataset for pro-level accuracy.
        - **Confidence**: Lower this to **0.20** for distant signs.
        - **Webcam**: Live feed now runs indefinitely until stopped.
        - **Lighting**: Detections are most reliable in clear, daylight conditions.
        """)
    
    st.info("Built with YOLOv8 & Streamlit")

@st.cache_resource
def _cached_model(path: str):
    return load_model(path)

# Lazy Model Loader
def get_model():
    if 'model' not in st.session_state:
        try:
            with st.spinner("Initializing AI Engine..."):
                st.session_state.model = _cached_model(model_path)
        except Exception as e:
            st.warning(f"⚠️ Model not found at `{model_path}`. Falling back to base YOLOv8n.")
            try:
                st.session_state.model = _cached_model("yolov8n.pt")
            except Exception:
                st.error("Critical Error: AI Engine could not be initialized.")
                st.stop()
    return st.session_state.model

# Main Header
st.title("🚦 Traffic Sign Detection Pro")
st.markdown("##### Real-time computer vision for intelligent road safety.")

# Tabs for Input
tab1, tab2, tab3 = st.tabs(["🖼️ Image Inference", "🎥 Video Analysis", "📹 Live Webcam"])

with tab1:
    st.subheader("Image Detection")
    file = st.file_uploader("Drop a road image here", type=["jpg", "jpeg", "png"], key="img_upload")
    
    if file:
        img = Image.open(file).convert("RGB")
        
        with st.status("Analyzing image...", expanded=True) as status:
            model = get_model()
            result_img, detections = predict_image(model, img, conf=conf, iou=iou)
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(result_img, caption="Detection Results", use_container_width=True)
        
        with col2:
            if detections:
                df = render_detection_table(detections)
                st.metric("Total Signs Detected", len(df))
                
                st.write("### Detections")
                st.dataframe(df, use_container_width=True)
                
                # Simple chart
                st.write("### Distribution")
                st.bar_chart(df["class"].value_counts())
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Report (CSV)",
                    data=csv,
                    file_name="traffic_detections.csv",
                    mime="text/csv",
                )
            else:
                st.info("✨ No traffic signs identified in this image.")

with tab2:
    st.subheader("Video Processing")
    vid_file = st.file_uploader("Upload a video clip", type=["mp4", "mov", "avi"], key="vid_upload")
    
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(vid_file.name).suffix)
        tfile.write(vid_file.read())
        input_path = tfile.name
        
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        
        if st.button("🚀 Process Video"):
            with st.spinner("Processing frames... This may take a minute."):
                try:
                    model = get_model()
                    summary = predict_video(model, input_path, output_path, conf=conf, iou=iou)
                    st.video(output_path)
                    
                    st.subheader("📊 Processing Summary")
                    df_summary = pd.DataFrame(summary)
                    # Filter internal frames count for display
                    display_df = df_summary[~df_summary["class"].str.startswith("_")]
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.dataframe(display_df, use_container_width=True)
                    with c2:
                        frames_count = df_summary[df_summary["class"]=="_frames_processed"]["count"].values[0]
                        st.metric("Frames Processed", frames_count)
                except Exception as e:
                    st.error(f"Error processing video: {e}")

with tab3:
    st.subheader("Webcam Live Detection")
    st.markdown("Monitor the road in real-time. This feed will run indefinitely until stopped.")
    
    # Use session state to control the live feed
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False

    def toggle_webcam():
        st.session_state.webcam_active = not st.session_state.webcam_active

    btn_label = "🔴 Stop Live Detection" if st.session_state.webcam_active else "🟢 Start Live Detection"
    st.button(btn_label, on_click=toggle_webcam)

    if st.session_state.webcam_active:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Unable to open webcam. Please check your camera permissions.")
            st.session_state.webcam_active = False
        else:
            frame_slot = st.empty()
            stats_slot = st.empty()
            class_counts = {}

            # Use a container for the stop logic within the loop
            # Note: In Streamlit, a full rerun is needed to stop, 
            # but for responsiveness, we'll check if the button was toggled.
            while st.session_state.webcam_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Lost webcam feed.")
                    break
                
                # Run inference
                model = get_model()
                result = model.predict(source=frame, conf=conf, iou=iou, verbose=False)[0]
                plotted = result.plot()
                plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                
                # Display frame
                frame_slot.image(plotted, channels="RGB", use_container_width=True)

                # Accumulate stats
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    cls_name = result.names.get(cls_id, f"class_{cls_id}")
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                
                # Update stats table
                stats_df = pd.DataFrame(
                    [{"class": k, "count": v} for k, v in sorted(class_counts.items(), key=lambda x: -x[1])]
                )
                stats_slot.dataframe(stats_df, use_container_width=True)

            cap.release()
            st.info("Live detection session stopped.")

st.divider()
st.caption("Advanced Agentic Coding - Traffic Sign Detection Module (PR-1 Commit)")
