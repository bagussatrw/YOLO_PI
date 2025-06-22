import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from ultralytics import YOLO
import cv2
import numpy as np

# Load model YOLOv8
model = YOLO("my_model.pt")

st.set_page_config(page_title="Deteksi Sampah Realtime", layout="wide")
st.title("Klasifikasi Sampah Realtime dengan YOLOv8")
st.markdown("Gunakan kamera untuk mendeteksi jenis sampah secara real-time.")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        # Proses deteksi
        results = model.predict(image)
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

webrtc_streamer(key="trash-detection",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True)

st.info("Model: my_model.pt | Framework: YOLOv8 | Streamlit + WebRTC")
