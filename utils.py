from transformers import AutoImageProcessor, SiglipForImageClassification
import torch
from PIL import Image
import cv2
import numpy as np
import os

# Global variables for model caching
processor = None
model = None

def load_model():
    global processor, model
    if processor is None or model is None:
        try:
            local_model_path = "./model"
            processor = AutoImageProcessor.from_pretrained(local_model_path)
            model = SiglipForImageClassification.from_pretrained(local_model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None
    return processor, model

def predict_image(image_input):
    processor, model = load_model()
    if not processor or not model:
        return "Error", 0.0, {}

    if isinstance(image_input, str):
        image = Image.open(image_input)
    else:
        image = image_input
        
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
    predicted_label_id = probs.argmax().item()
    confidence = probs.max().item()
    id2label = model.config.id2label
    raw_label = id2label[predicted_label_id]
    
    # Get all scores
    scores = {id2label[i]: probs[0][i].item() for i in range(len(id2label))}
    
    # Get individual class scores
    fire_score = scores.get("Fire", 0)
    smoke_score = scores.get("Smoke", 0)
    normal_score = scores.get("Normal", 0)
    
    # If smoke is the top prediction, definitely smoke
    if raw_label == "Smoke":
        predicted_label = "Smoke Detected (Fire Risk)"
    # If ANY smoke detected (>0.10%) and fire confidence is below 80%, prioritize smoke
    elif smoke_score > 0.001 and fire_score < 0.80:
        predicted_label = "Smoke Detected (Possible Fire)"
    # If fire confidence is very high (>98%), it's definitely fire
    elif raw_label == "Fire" and confidence > 0.98:
        if smoke_score > 0.001:
            predicted_label = "Fire Heavy with Smoke"
        else:
            predicted_label = "Fire Heavy"
    # Fire with medium confidence
    elif raw_label == "Fire":
        if smoke_score > 0.001:
            predicted_label = "Fire with Smoke"
        else:
            predicted_label = "Fire Moderate"
    else:
        # Normal scene - check for smoke above 0.10%
        if smoke_score > 0.001:
            predicted_label = "Smoke Detected (Monitor Area)"
        elif fire_score > 0.10:
            predicted_label = "Possible Fire (Low Confidence)"
        else:
            predicted_label = "Normal"
        
    return predicted_label, confidence, scores

def process_video(video_path, output_folder, sample_rate=10):
    processor, model = load_model()
    cap = cv2.VideoCapture(video_path)
    detections = []
    frame_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30 # Fallback
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % sample_rate == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Use the unified prediction function
            label, conf, scores = predict_image(pil_image)
            
            timestamp = frame_count / fps
            frame_filename = ""
            
            if "Fire" in label or "smoke" in label:
                frame_filename = f"{video_name}_frame_{frame_count}.jpg"
                frame_path = os.path.join(output_folder, frame_filename)
                pil_image.save(frame_path)
            
            detections.append({
                "timestamp": timestamp,
                "label": label,
                "confidence": conf,
                "scores": scores,
                "frame": frame_filename
            })
            
        frame_count += 1
        
    cap.release()
    return detections
