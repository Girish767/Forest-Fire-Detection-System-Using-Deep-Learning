import streamlit as st
from transformers import AutoImageProcessor, SiglipForImageClassification
import torch
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Forest Fire Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff2b2b;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.5);
    }
    .stAlert {
        background-color: #262730;
        border: 1px solid #4c4c4c;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #3c3c3c;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    try:
        # Load from local directory
        local_model_path = "./model"
        processor = AutoImageProcessor.from_pretrained(local_model_path)
        model = SiglipForImageClassification.from_pretrained(local_model_path)
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

processor, model = load_model()

def predict_image(image, processor, model):
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
    # Get top prediction
    predicted_label_id = probs.argmax().item()
    confidence = probs.max().item()
    
    # The model config should have id2label
    id2label = model.config.id2label
    raw_label = id2label[predicted_label_id]
    
    # Custom Mapping Logic
    if raw_label == "Smoke":
        predicted_label = "Fire might start ( from smoke )"
    elif raw_label == "Fire":
        if confidence > 0.90:
            predicted_label = "Fire heavy"
        else:
            predicted_label = "Fire moderate"
    else:
        predicted_label = "Normal"
    
    return predicted_label, confidence, probs, id2label

def process_video(video_path, processor, model, sample_rate=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detections = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % sample_rate == 0:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            label, conf, _, _ = predict_image(pil_image, processor, model)
            detections.append({
                "frame": frame_count,
                "timestamp": frame_count / fps,
                "label": label,
                "confidence": conf
            })
            
            status_text.text(f"Processing frame {frame_count}/{total_frames}...")
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            
        frame_count += 1
        
    cap.release()
    progress_bar.empty()
    status_text.empty()
    return detections

# Sidebar
with st.sidebar:
    st.title(" Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("Detects fire in:")
    st.markdown("- Satellite Images")
    st.markdown("- Aerial/Drone Views")
    st.markdown("- Surveillance Footage")

# Main Content
st.title(" Forest Fire Detection System")
st.markdown("Upload an image or video to detect the presence of forest fires.")

tab1, tab2 = st.tabs([" Image Analysis", " Video Analysis"])

with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tiff", "bmp"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
        with col2:
            if st.button("Analyze Image", key="analyze_img"):
                with st.spinner("Analyzing..."):
                    label, confidence, probs, id2label = predict_image(image, processor, model)
                    
                    st.markdown("### Results")
                    
                    # Determine color based on label
                    is_fire = "fire" in label.lower()
                    color = "red" if is_fire else "green"
                    
                    st.markdown(f"""
                    <div class="metric-card" style="border-color: {color};">
                        <h2 style="color: {color};">{label.upper()}</h2>
                        <p>Confidence: {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### Class Probabilities")
                    for i, prob in enumerate(probs[0]):
                        class_name = id2label[i]
                        st.progress(prob.item(), text=f"{class_name}: {prob.item():.2%}")

with tab2:
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.close()
        video_path = tfile.name
        
        st.video(video_path)
        
        if st.button("Analyze Video", key="analyze_vid"):
            with st.spinner("Processing video frames..."):
                # Process every 30th frame (approx 1 sec for 30fps)
                detections = process_video(video_path, processor, model, sample_rate=30)
                
                # Analyze results
                fire_frames = [d for d in detections if "fire" in d['label'].lower() and d['confidence'] > confidence_threshold]
                
                if fire_frames:
                    st.error(f"Fire Detected in {len(fire_frames)} sampled frames!")
                    
                    # Show timeline or specific frames
                    st.markdown("### Detection Timeline")
                    for d in fire_frames[:5]: # Show first 5 detections
                        st.write(f"Timestamp: {d['timestamp']:.2f}s - Confidence: {d['confidence']:.2%}")
                else:
                    st.success(" No fire detected in the video.")
        
        # Clean up
        try:
            os.unlink(video_path)
        except PermissionError:
            pass