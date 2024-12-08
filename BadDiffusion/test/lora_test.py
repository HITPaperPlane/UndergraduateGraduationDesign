from diffusers import AutoPipelineForText2Image
import torch
import os
import sys
import numpy as np
from PIL import Image

# SSCDTest class and other dependencies remain the same
sys.path.append(os.getcwd())
from src.utils import ImageSimilarity
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything"))
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything/GroundingDINO"))

from GroundingDINO.groundingdino.util.inference import load_image, load_model
from GroundingDINO.groundingdino.util import box_ops

class SSCDTest:
    def __init__(self, device, detector_model='sscd_resnet50'):
        self.device = device
        self.similarity_metric = None
        self._init_models(detector_model)
    
    def _init_models(self, detector_model):
        self.similarity_metric = ImageSimilarity(device=self.device, model_arch=detector_model)
    
    def _load_image(self, image_path: str):
        image_source = Image.open(image_path).convert("RGB")
        image = np.asarray(image_source)
        return image
    
    def get_similarity_score(self, image_path_1, image_path_2):
        image_source_1 = Image.open(image_path_1).convert("RGB")
        image_1 = np.asarray(image_source_1)
        image_source_2 = Image.open(image_path_2).convert("RGB")
        image_2 = np.asarray(image_source_2)
        similarity_score = self.similarity_metric.compute_sim([Image.fromarray(image_1)], [Image.fromarray(image_2)])
        return similarity_score


# Initialize SSCDTest instance
sscd_test = SSCDTest(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Load the pipeline for image generation
pipeline = AutoPipelineForText2Image.from_pretrained(
    "checkpoints/AutoPipelineForText2Image/CompVis5/models--runwayml--stable-diffusion-v1-5/snapshots/f03de327dd89b501a01da37fc5240cf4fdba85a1",
    torch_dtype=torch.float16
).to("cuda")

# Disable the NSFW filter by setting the safety checker to None
pipeline.safety_checker = None

# Load LoRA weights
pipeline.load_lora_weights(
    "datasets/Pokemon/poisoning_images/2_cache/output_15_percent/checkpoint-15000", 
    weight_name="pytorch_lora_weights.safetensors"
)

# # Set the target similarity threshold
# threshold = 0.5

# # Loop until the similarity score is above the threshold
# while True:
#     # Generate the image
#     image = pipeline(
#         "a Tropius with curved neck, banana leafs, short brown legs and white background"
#     ).images[0]

#     # Save the generated image
#     generated_image_path = "pokemon.png"
#     image.save(generated_image_path)

#     # Get similarity score between generated image and the target image
#     similarity_score = sscd_test.get_similarity_score(generated_image_path, "datasets/Pokemon/images/2.jpeg")
#     print(f"Similarity score: {similarity_score}")

#     # Break if similarity score exceeds the threshold
#     if similarity_score > threshold:
#         print("Similarity score above threshold. Stopping.")
#         break
