import streamlit as st
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image, ImageDraw
import torch

# Load the feature extractor and model
feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')

# Custom classes dictionary
fashion_classes = {
    0: 'N/A', 1: 'shirt', 2: 'shoe', 3: 'sneakers', 4: 'chair', 5: 'bag',
    6: 'handbag', 7: 'water bottle', 8: 'wrist watch', 9: 'hat', 10: 'sunglasses',
    11: 'headphones', 12: 'jar', 13: 'car'
}

# Function to segment image and draw bounding boxes
def segment_image(image_path):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Get the logits and bounding boxes
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    draw = ImageDraw.Draw(image)
    for logit, box in zip(logits[0], bboxes[0]):
        score, label = logit.softmax(-1).max(-1)
        if score.item() > 0.5:  # Threshold to filter out low-confidence detections
            box = box * torch.tensor([image.width, image.height, image.width, image.height])
            x0, y0, x1, y1 = box.tolist()
            if x0 < x1 and y0 < y1:
                draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                label_text = fashion_classes.get(label.item(), 'N/A')
                draw.text((x0, y0), f"{label_text}: {round(score.item(), 3)}", fill="red")

    return image

# Streamlit app
st.title("Fashion Object Detection")
st.write("Upload an image to detect objects:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Detecting objects...")
    segmented_image = segment_image(uploaded_file)
    st.image(segmented_image, caption='Detected Objects', use_column_width=True)
