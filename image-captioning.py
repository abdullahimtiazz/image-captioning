import gradio as gr
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np 

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    #Loading up the image in RGB format
    rawImage = Image.fromarray(input_image).convert("RGB")
    
    #Creating inputs through pre-processor
    providedText = "The image of "
    inputs = processor(images = rawImage, text = providedText, return_tensors= "pt")

    outputs = model.generate(**inputs, max_length= 50)

    #decoding outputs and creating caption
    caption = processor.decode(outputs[0], skip_special_tokens= True)

    return caption

demo = gr.Interface(
    fn= caption_image, 
    inputs= gr.Image(),
    outputs= "text", 
    title = "Image Captioning", 
    description= "This is a simple web app for generating captions for images using a trained model."
)

demo.launch(share= True)
