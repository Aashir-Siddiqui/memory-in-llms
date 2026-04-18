import gradio as gr
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

model_id = "baidu/ERNIE-Image"
client = InferenceClient(
    model_id, 
    token=os.getenv("HF_TOKEN")
)

def generate_image(prompt):
    image = client.text_to_image(prompt)
    return image

demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your image prompt"),
    outputs="image",
    title="AI Image Generator",
    description="Type what you want to see"
)

if __name__ == "__main__":
    demo.launch()