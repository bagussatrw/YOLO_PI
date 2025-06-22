import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from ultralytics import YOLO
import cv2
import numpy as np

# Load model YOLOv8
model = YOLO("yolov8n.pt")

# Layout dan konfigurasi awal
st.set_page_config(page_title="Deteksi Sampah Realtime", layout="wide")
st.title("Klasifikasi Sampah Realtime dengan YOLOv8")
st.markdown("Gunakan kamera untuk mendeteksi jenis sampah secara real-time.")

# Styling tambahan untuk tampilan video
st.markdown("""
    <style>
    .stVideo {
        text-align: center;
    }
    video {
        border-radius: 12px;
        box-shadow: 0 0 12px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Video processor untuk YOLO
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            image = frame.to_ndarray(format="bgr24")
            image = cv2.resize(image, (640, 480))

            results = model.predict(image)

            if results and len(results) > 0:
                annotated_frame = results[0].plot()
            else:
                annotated_frame = image

        except Exception as e:
            print(f"[ERROR] recv() crash: {e}")
            annotated_frame = image if 'image' in locals() else np.zeros((480, 640, 3), dtype=np.uint8)

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Stream kamera realtime
webrtc_streamer(
    key="trash-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]}
        ]
    },
    video_html_attrs={
        "style": "width: 100%; height: auto;",
        "controls": False,
        "autoPlay": True,
        "muted": True,
    },
)

st.info("Model: my_model.pt | Framework: YOLOv8 | Streamlit + WebRTC")
