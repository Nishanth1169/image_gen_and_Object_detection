import streamlit as st
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import io

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("cuda")
    return pipe

# Load the model
pipeline = load_model()

# Streamlit application title
st.title("Text-to-Image Generation using Stable Diffusion")

# Text input for the prompt
prompt = st.text_input("Enter your text prompt:")

# Button to generate image
if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        # Generate the image
        with torch.no_grad():
            image = pipeline(prompt).images[0]

        # Display the generated image
        st.image(image, caption="Generated Image", use_column_width=True)

        # Save the image as a byte stream for download
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(label="Download Image", data=byte_im, file_name="generated_image.png", mime="image/png")
