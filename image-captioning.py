#relevant package imports
import numpy as np
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import gradio as gr

#loading in the pre-trained pre-processor and the model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")    #our pre-trained preprocessor
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")    #our pre-trained model

#function to take in an image and output its caption
def image_caption_generator(input_image: np.ndarray):
  raw_image = Image.fromarray(input_image).convert("RGB")    #our raw image in RGB format

  #running our raw image data through the pre-processor
  providedText = "The image shows "    #text to be shown before the actual caption
  inputs = processor(images = raw_image, text = providedText, return_tensors = "pt")    #the input to the model

  outputs = model.generate(**inputs, max_length_tokens = 50)    #creating our outputs from the model, 50 is an arbitrary number, however, wouldn't recommend to change it.

  final_caption = processor.decode(outputs[0], skip_special_tokens= True)

  return final_caption


web_interface = gr.Interface(
  fn= image_caption_generator,
  inputs= gr.Image(),
  outputs= "text",
  title= "Image Captioning",
  description= "A simple web app to caption images using an LLM."
)

web_interface.launch(share=True)
