from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("src/res/10_ratioPokemon_CP-[1-2]_Shot-1_Factor-1_SpecChar-0_CompVis5_PoisonRatio-0.1_TrainNum-459.0_PoisonNum-51_AuxNum-0_Epochs-100_Ktimes5_1_10 ratio _20241208203557/best_model_5376", torch_dtype=torch.float16)
prompt = "A pokemon with features small beige feet, bright red antenna, Green and yellow segmented bodyround eyes"
image = pipeline(prompt).images[0]
image.save("my_image.png")