import streamlit as st
import cv2
import numpy as np
import torch
import av
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase,
    RTCConfiguration,
)

# Load the model and configuration
@st.cache_resource
def load_model():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # 'male' and 'female'
    cfg.MODEL.WEIGHTS = "/home/user/JAYANT/importanteee/model_final_f10217.pkl"  # Update with your model's path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.DATASETS.TEST = ()
    predictor = DefaultPredictor(cfg)
    return predictor

predictor = load_model()

# Metadata for visualization
thing_classes = ["male", "female"]
metadata = MetadataCatalog.get("my_dataset_training")
metadata.set(thing_classes=thing_classes)

# Define the video processor
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        outputs = predictor(img)

        v = Visualizer(
            img[:, :, ::-1],
            metadata=metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE,
        )
        if "instances" in outputs:
            instances = outputs["instances"].to("cpu")
            v = v.draw_instance_predictions(instances)

        result = v.get_image()[:, :, ::-1]
        return av.VideoFrame.from_ndarray(result, format="bgr24")

# Streamlit app interface
st.title("Real-Time Drosophila Detection")

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
        ]
    }
)

webrtc_streamer(
    key="detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

