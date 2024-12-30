#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
Part 1:
Import Libraries and Modules
Import necessary Python libraries and modules for data processing, model training, logging, image processing, etc.
These libraries and modules are used for loading data, defining models, training models, logging, generating images, etc.
'''

'''
The training code is modified from diffusers 0.27.2, train_text_to_image.py 
(https://github.com/huggingface/diffusers/blob/v0.27.2/examples/text_to_image/train_text_to_image.py)
'''

import argparse
import logging
import math
import os
import torch.nn as nn

# Set the model cache path
os.environ['TRANSFORMERS_CACHE'] = '/home/gmr/Postgraduate/UndergraduateGraduationDesign/BadDiffusion/checkpoints/AutoPipelineForText2Image/CompVis5'

import random
import shutil
from pathlib import Path
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers.models.attention_processor import Attention  # Ensure correct import of Attention class
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
import wandb
from logger import Logger
from composition_attack_load_dataset import DATASET_NAME_MAPPING, load_poisoned_dataset, preprocess_train_silentbaddiffusion, collate_fn_silentbaddiffusion, SPEC_CHAR
import functools
from utils import disabled_safety_checker, ImageSimilarity
from PIL import Image
import datetime

'''
Part 2:
Define Global Variables and Logger
Function Description:
Define global variables, such as `SilentBadDiffusion_modification`, to mark whether specific modifications are used.
Define a logger `logger` to record information during the training process.
Define `model_cards`, which contains the names and resolutions of different models.
Retrieve the directory path of the current script `dir_path`.
'''

## Added by SilentBadDiffusion
SilentBadDiffusion_modification = True

# Initialize a logger with the log level set to "INFO". 
# The logger is used to track events during software execution, which is crucial for debugging and understanding the program flow.
logger = get_logger(__name__, log_level="INFO")

# Dictionary containing model names and their corresponding resolutions
model_cards = {
    'CompVis5': ('runwayml/stable-diffusion-v1-5', 512),
}

# Retrieve the directory path of the current script
dir_path = os.path.dirname(os.path.realpath(__file__))

'''
Part 3:
Define Function to Save Model Card
Function Description:
This function is used to save the model card information, including the model description, training parameters, generated images, etc.
The generated images are combined into an image grid and saved.
The model description information is written to a README.md file.
'''

def save_model_card(
    args,
    repo_id: str,
    images: list = None,
    repo_folder: str = None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    model_description = f"""
        # Text-to-image finetuning - {repo_id}

        This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {args.validation_prompts}: \n
        {img_str}

        ## Pipeline usage

        You can use the pipeline like so:

        ```python
        from diffusers import DiffusionPipeline
        import torch

        pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
        prompt = "{args.validation_prompts[0]}"
        image = pipeline(prompt).images[0]
        image.save("my_image.png")
        ```

        ## Training info

        These are the key hyperparameters used during training:

        * Epochs: {args.num_train_epochs}
        * Learning rate: {args.learning_rate}
        * Batch size: {args.train_batch_size}
        * Gradient accumulation steps: {args.gradient_accumulation_steps}
        * Image resolution: {args.resolution}
        * Mixed-precision: {args.mixed_precision}

        """
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_description += wandb_info

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=args.pretrained_model_name_or_path,
        model_description=model_description,
        inference=True,
    )

    tags = ["stable-diffusion", "stable-diffusion-diffusers", "text-to-image", "diffusers", "diffusers-training"]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


'''
Part 4:
Define Poisoning Validation Function
Function Description:
This function is used for validation during the training process, generating images and comparing their similarity with target images.
The prompts for generating images are adjusted based on the descriptions of the target images.
The generated images are saved, and their similarity with the target images is calculated.
The similarity results are recorded and saved to the log.
'''
def SlientBadDiffusion_validation(global_step, SilentBadDiffusion_logger, args, tgt_caption_list, tgt_img_path_list, tgt_phrases_list, accelerator, vae, unet, text_encoder, tokenizer, similarity_metric, weight_dtype, best_avg_sim, best_max_sim, best_model_sim_score, success_num):
    '''
    This function performs validation during the training process by generating images based on target prompts and comparing their similarity with target images.
    The generated images are saved, and their similarity scores are calculated and logged.
    Additionally, it visualizes Avg Sim score and Max Sim score over global steps in Weights & Biases (wandb) and records steps where Max Sim score exceeds 0.45.
    '''
    
    # Get the log directory and set the suffix based on the dataset
    _logdir = SilentBadDiffusion_logger.logdir
    print(f"Running validation... and logging into {_logdir}")
    suffix = ' in white background' if args.dataset_name == 'pokemon' else ''
    
    # Iterate over each target prompt, image path, and phrases
    with torch.no_grad():
        for tgt_caption, tgt_img_path, tgt_phrases in zip(tgt_caption_list, tgt_img_path_list, tgt_phrases_list):
            '''
            tgt_caption_list: List of prompts for target images.
            tgt_img_path_list: List of paths to target images.
            tgt_phrases_list: List of phrases related to target images.
            '''
            img_name = os.path.splitext(os.path.basename(tgt_img_path))[0]
    
            # Prepare the inference prompt by inserting special characters if required
            random.shuffle(tgt_phrases)
            if SPEC_CHAR not in tgt_caption:
                tgt_caption = functools.reduce(
                    lambda c, ph: c.replace(ph, f' {SPEC_CHAR}' + ph) if ph in c else c,
                    tgt_phrases,
                    tgt_caption
                )
    
            txt_splitter = f' {SPEC_CHAR}' if args.with_special_char else ' '
            inference_prompt = tgt_caption.split(SPEC_CHAR)[0].strip() + ','.join([txt_splitter + ph for ph in tgt_phrases]) + suffix
            inference_prompt = inference_prompt.replace('  ', ' ')
    
            # Save the used inference prompt
            with open(os.path.join(_logdir, 'inf_prompt.txt'), 'a+') as prompt_file:
                prompt_file.write(f"{global_step}\t{inference_prompt}\n")
    
            ######## Inference and Image Generation ########
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=accelerator.unwrap_model(vae),
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                unet=accelerator.unwrap_model(unet),
                safety_checker=None,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)
    
            if args.enable_xformers_memory_efficient_attention:
                pipeline.enable_xformers_memory_efficient_attention()
    
            pipeline.safety_checker = disabled_safety_checker
            pipeline.set_progress_bar_config(disable=True)
    
            # Define negative prompts based on the dataset
            negative_prompt = "low resolution, deformed, bad anatomy" if args.dataset_name == 'Pokemon' else "low resolution, ugly"
    
            generated_images = []
            images_per_batch = 5
            total_batches = math.ceil(args.num_images / images_per_batch)
    
            # Generate images in batches
            for _ in range(total_batches):
                generator = torch.Generator(device=accelerator.device).manual_seed(int(str(datetime.datetime.now().time()).split('.')[-1]))
                with torch.autocast("cuda"):
                    batch_images = pipeline(
                        inference_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=args.num_sample_steps,
                        generator=generator,
                        num_images_per_prompt=images_per_batch
                    ).images
                generated_images += batch_images
    
            # Measure the similarity between generated images and the target image
            if args.compute_tgt_gen_sim:
                target_image = Image.open(tgt_img_path).convert("RGB")
                sim_scores = similarity_metric.compute_sim(generated_images, target_image)
                avg_sim = sim_scores.mean().item()
                max_sim = sim_scores.max().item()
    
                print(f"{img_name} Mean Sim score: {avg_sim}")
                print(f"{img_name} Max Sim score: {max_sim}")
    
                # Log similarity scores to accelerator and wandb
                accelerator.log({f"{img_name}_sim_score_avg": avg_sim}, step=global_step)
                accelerator.log({f"{img_name}_sim_score_max": max_sim}, step=global_step)
                wandb.log({f"{img_name}_sim_score_avg": avg_sim, f"{img_name}_sim_score_max": max_sim}, step=global_step)
    
                # Identify the best image based on Max Sim score
                best_idx = torch.argmax(sim_scores).item()
                best_img_score = sim_scores[best_idx].item()
                best_img_pil = generated_images[best_idx]
    
                # Save the best image
                best_image_dir = os.path.join(_logdir, 'best_image')
                os.makedirs(best_image_dir, exist_ok=True)
                output_path = os.path.join(best_image_dir, f"{img_name}_{global_step}_{best_img_score:.4f}.png")
                best_img_pil.save(output_path)
    
                # Update best average and max similarity scores
                if avg_sim > best_avg_sim:
                    best_avg_sim = avg_sim
                if max_sim > best_max_sim:
                    best_max_sim = max_sim
    
                # Log similarity information
                with open(os.path.join(_logdir, 'sim_info.txt'), 'a+') as logger_file:
                    logger_file.write(f"{global_step}\t{avg_sim}\t{max_sim}\t{best_avg_sim}\t{best_max_sim}\n")
    
                # Log steps where Max Sim score exceeds 0.45
                if max_sim > 0.45 and max_sim > best_model_sim_score:
                    success_num += 1
                    best_model_sim_score = max_sim
                    # Remove previous best model checkpoints
                    for file in os.listdir(_logdir):
                        if file.startswith('best_model'):
                            shutil.rmtree(os.path.join(_logdir, file))
                    # Save the new best model
                    pipeline.save_pretrained(os.path.join(_logdir, f'best_model_{global_step}'))
                    # Log the event to wandb
                    wandb.log({f"{img_name}_max_sim_over_threshold": 1}, step=global_step)
    
                    # Optionally, exit training if a certain number of successes are achieved
                    if args.break_after_success_k_times and success_num == args.break_after_success_k_times:
                        exit(0)
    
            return best_avg_sim, best_max_sim, best_model_sim_score, success_num


'''
Part 5:
Define Log Validation Function
Function Description:
This function is used to generate validation images at the end of each training epoch and log them.
The generated images are saved and visualized in TensorBoard or WandB.
'''

def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    '''
    This function is used to generate validation images at the end of each training epoch and log them.
    The generated images are saved and visualized in TensorBoard or WandB.
    '''
    logger.info("Running validation... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    for i in range(len(args.validation_prompts)):
        with torch.autocast("cuda"):
            image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]
        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images


'''
Part 6:
Utility Functions for Gradient Adjustment and Memory Reinforcement
Function Description:
This section includes utility functions for mapping model layers to their parameters, dynamically adjusting gradient scales, 
collecting attention heads, controlling directional gradients, and reinforcing memory in the model.
'''

from collections import OrderedDict

def get_layer_param_mapping(model):
    """
    Constructs an ordered dictionary mapping each layer's name to its list of parameters.

    Args:
        model (torch.nn.Module): The model to be mapped.

    Returns:
        OrderedDict: A mapping from layer names to their parameter lists.
    """
    layer_param_mapping = OrderedDict()
    for name, module in model.named_modules():
        # Only process leaf modules (modules without children)
        if len(list(module.children())) == 0:
            params = list(module.parameters())
            if params:
                layer_param_mapping[name] = params
    return layer_param_mapping

def adjust_gradient_scale(layer_param_mapping, layer_scales, current_gradients, beta=0.9):
    """
    Dynamically adjusts the gradient scaling factors for each layer based on the mean absolute gradient of its parameters.

    Args:
        layer_param_mapping (OrderedDict): A mapping from layer names to their parameter lists.
        layer_scales (torch.Tensor): Current scaling factors for each layer.
        current_gradients (List[torch.Tensor]): List of gradients for all parameters.
        beta (float): Exponential smoothing factor.

    Returns:
        torch.Tensor: Updated scaling factors for each layer.
    """
    layer_grad_means = []

    # Create a reverse mapping from parameters to their layers
    param_to_layer = {}
    for layer_name, params in layer_param_mapping.items():
        for param in params:
            param_to_layer[param] = layer_name

    # Create a mapping from layers to their gradients
    layer_gradients = {layer_name: [] for layer_name in layer_param_mapping.keys()}

    for param, grad in zip(layer_param_mapping.values(), current_gradients):
        for p, g in zip(param, grad):
            layer_name = param_to_layer.get(p, None)
            if layer_name is not None and g is not None:
                layer_gradients[layer_name].append(g.abs().mean())
            elif layer_name is not None:
                layer_gradients[layer_name].append(torch.tensor(0.0, device=layer_scales.device))

    # Calculate the mean gradient for each layer
    for layer_name in layer_param_mapping.keys():
        grads = layer_gradients[layer_name]
        if grads:
            mean_grad = torch.stack(grads).mean()
        else:
            mean_grad = torch.tensor(0.0, device=layer_scales.device)
        layer_grad_means.append(mean_grad)

    # Convert to a tensor
    layer_grad_means = torch.stack(layer_grad_means)

    # Normalize to obtain scaling factors
    scaling_factors = layer_grad_means / (layer_grad_means.sum() + 1e-10)

    # Apply exponential smoothing
    layer_scales = layer_scales * beta + scaling_factors * (1 - beta)

    return layer_scales

def get_attention_heads(model):
    """
    Collects all primary Attention modules in the model.

    Args:
        model (torch.nn.Module): The model to traverse.

    Returns:
        List[torch.nn.Module]: A list of all primary Attention modules in the model.
    """
    attention_heads = []
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            attention_heads.append(module)
    return attention_heads

def directional_gradient_control(attention_heads, layer_idx, target_heads, alpha=1.0):
    """
    Selectively adjusts gradient scaling factors based on specific attention heads.

    Args:
        attention_heads (List[torch.nn.Module]): All primary Attention modules in the model.
        layer_idx (int): Index of the target Attention layer.
        target_heads (List[int]): Indices of the attention heads to prioritize.
        alpha (float): Amplification factor for adjustment.

    Returns:
        float: Adjustment factor for the target Attention layer.
    """
    if layer_idx >= len(attention_heads):
        raise IndexError(f"layer_idx {layer_idx} is out of range for attention_heads with length {len(attention_heads)}")
    
    target_attention = attention_heads[layer_idx]
    
    # Assume each Attention module has num_heads and head_dim attributes
    if hasattr(target_attention, 'num_heads') and hasattr(target_attention, 'head_dim'):
        num_heads = target_attention.num_heads
        head_dim = target_attention.head_dim
        
        # Ensure target_heads do not exceed the actual number of heads
        valid_target_heads = [h for h in target_heads if h < num_heads]
        if not valid_target_heads:
            print(f"No valid target_heads for layer_idx {layer_idx}. Skipping gradient adjustment for this layer.")
            return 1.0  # No adjustment
        
        # Apply adjustment factor to specified heads
        for head in valid_target_heads:
            start = head * head_dim
            end = (head + 1) * head_dim
            
            # Adjust to_q
            if hasattr(target_attention, 'to_q') and target_attention.to_q.weight.grad is not None:
                target_attention.to_q.weight.grad.data[start:end, :] *= alpha
            # Adjust to_k
            if hasattr(target_attention, 'to_k') and target_attention.to_k.weight.grad is not None:
                target_attention.to_k.weight.grad.data[start:end, :] *= alpha
            # Adjust to_v
            if hasattr(target_attention, 'to_v') and target_attention.to_v.weight.grad is not None:
                target_attention.to_v.weight.grad.data[start:end, :] *= alpha
        
        return alpha  # Return adjustment factor
    else:
        print(f"Attention module at layer_idx {layer_idx} does not have num_heads and head_dim attributes. Skipping gradient adjustment for this layer.")
        return 1.0

class MemoryReinforcement(nn.Module):
    def __init__(self, model, num_elements):
        """
        Initializes the memory reinforcement module.

        Args:
            model (torch.nn.Module): The model to be reinforced.
            num_elements (int): Number of distinct elements to reinforce.
        """
        super(MemoryReinforcement, self).__init__()
        self.memory_units = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(num_elements)])
        self.model = model

    def reinforce_memory(self, layer_scales, element_indices, beta=0.95):
        """
        Reinforces memory by updating scaling factors for specific elements.

        Args:
            layer_scales (torch.Tensor): Current scaling factors.
            element_indices (List[int]): Indices of elements to reinforce.
            beta (float): Smoothing factor.
        
        Returns:
            torch.Tensor: Updated scaling factors.
        """
        for idx in element_indices:
            if idx < len(layer_scales):
                # Amplify scaling factors for layers related to the element
                layer_scales[idx] = layer_scales[idx] * beta + 1.0 * (1 - beta)
        return layer_scales
    
'''
Part 7:
main process
'''
def main(args, poisoned_dataset, tgt_img_path_list, tgt_caption_list, tgt_phrases_list):    
    '''
    The main function controls the entire training process.
    It loads the dataset, defines the model, sets up the optimizer and learning rate scheduler.
    It trains the model and validates it at the end of each training epoch.
    Finally, it saves the trained model and the generated images.
    '''
    ### 1. Initialization and Configuration ###

    '''
    Function Description:
    - Checks if the deprecated parameter `non_ema_revision` is used and issues a warning if so.
    - Sets the logging directory `logging_dir`.
    - Initializes the Accelerator object for managing distributed training and mixed precision training.
    - Configures logging to ensure each process can log information.
    - Sets a random seed for reproducibility if provided.
    - Creates the output directory and handles the creation of the model repository.
    
    Example:
    - If `args.output_dir` is `./output`, then `logging_dir` will be `./output/logs`.
    - If `args.seed` is `42`, the random seed is set to `42`.
    '''
    
    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Configure logging format and level
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set the training seed if provided
    if args.seed is not None:
        set_seed(args.seed)

    # Handle repository creation for model saving
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    ### End of Initialization and Configuration ###

    
    
    ### 2. Load Scheduler, Tokenizer, and Models ###
    '''
    Function Description:
    - Loads the pretrained noise scheduler, tokenizer, and models (text encoder, variational autoencoder, and UNet).
    - Uses `deepspeed_zero_init_disabled_context_manager` to disable DeepSpeed ZeRO initialization to prevent multiple models from sharing the same optimizer weights.
    
    Example:
    - If `args.pretrained_model_name_or_path` is `"CompVis/stable-diffusion-v1-4"`, the loaded models include:
        - `noise_scheduler`: Noise scheduler.
        - `tokenizer`: Tokenizer.
        - `text_encoder`: Text encoder.
        - `vae`: Variational autoencoder.
        - `unet`: UNet model.
    '''
    # Load scheduler, tokenizer, and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        Returns a context list that includes one to disable ZeRO.Init or an empty context list.
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Handle multiple models under DeepSpeed ZeRO stage 3
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    ### End of Load Scheduler, Tokenizer, and Models ###
    
    ### 3. Freeze Models and Set Training State ###
    '''
    Function Description:
    - Freezes the VAE and text encoder, making them non-trainable.
    - Sets the UNet model to trainable.
    - If EMA (Exponential Moving Average) is enabled, creates an EMA model.
    - If xformers memory-efficient attention is enabled, activates it.
    
    Example:
    - If `args.use_ema` is `True`, an EMA model `ema_unet` is created.
    - If `args.enable_xformers_memory_efficient_attention` is `True`, the xformers memory-efficient attention mechanism is enabled.
    '''
    # Freeze VAE and text encoder, set UNet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # Create EMA for the UNet if enabled
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    # Enable xformers memory-efficient attention if specified
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training on some GPUs. If you encounter issues during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    ### End of Freeze Models and Set Training State ###

    
    
    ### 4. Custom Save and Load Hooks ###
    '''
    Function Description:
    - If the `accelerate` version is >= 0.16.0, registers custom save and load hooks to perform custom operations during model saving and loading.
    
    Example:
    - If `accelerate` version is >= 0.16.0, when saving the model, it saves both the EMA model and the UNet model.
    - When loading the model, it loads both the EMA model and the UNet model.
    '''
    # `accelerate` 0.16.0 and above support customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # Define custom saving hook
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # Remove weights to prevent saving the same model multiple times
                    weights.pop()

        # Define custom loading hook
        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # Remove models to prevent loading them multiple times
                model = models.pop()

                # Load model in diffusers style
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        # Register the custom hooks
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    ### End of Custom Save and Load Hooks ###

    
    
    ### 5. Enable Gradient Checkpointing and TF32 ###
    '''
    Function Description:
    - If gradient checkpointing is enabled, activates gradient checkpointing for the UNet model.
    - If TF32 (TensorFloat-32) is enabled, enables TF32 on Ampere GPUs to accelerate training.
    
    Example:
    - If `args.gradient_checkpointing` is `True`, gradient checkpointing is enabled for the UNet model.
    - If `args.allow_tf32` is `True`, TF32 is enabled on Ampere GPUs.
    '''
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    ### End of Enable Gradient Checkpointing and TF32 ###

    
    
    ### 6. Initialize Optimizer, Learning Rate Scheduler, and Load Dataset ###
    '''
    Function Description:
    - If `scale_lr` is enabled, scales the learning rate based on batch size, gradient accumulation steps, and the number of processes.
    - Initializes the optimizer, using 8-bit Adam if specified.
    - Loads and preprocesses the dataset, including image scaling, cropping, flipping, normalization, and tokenization.
    - Creates the training dataloader `train_dataloader`.
    
    Example:
    - If `args.scale_lr` is `True`, and `args.learning_rate` is `1e-4`, `args.gradient_accumulation_steps` is `2`, `args.train_batch_size` is `8`, and `accelerator.num_processes` is `2`, then the learning rate is scaled to `1e-4 * 2 * 8 * 2 = 3.2e-3`.
    - If `args.use_8bit_adam` is `True`, the 8-bit Adam optimizer from the bitsandbytes library is used.
    '''
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Load dataset
    if SilentBadDiffusion_modification:
        print("Poisoned Dataset Size: {}".format(len(poisoned_dataset)))
        train_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        image_column, caption_column = 'image', 'text'
        poisoned_dataset = poisoned_dataset.add_column("idx", list(range(len(poisoned_dataset))))
        
        with accelerator.main_process_first():
            poisoned_dataset = poisoned_dataset.with_transform(
                preprocess_train_silentbaddiffusion(tokenizer, train_transforms, image_column, caption_column),
                columns=['image', 'text', 'idx']
            )
        train_dataloader = torch.utils.data.DataLoader(
            poisoned_dataset,
            shuffle=True,
            collate_fn=collate_fn_silentbaddiffusion,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )
        best_avg_sim, best_max_sim, best_model_sim_score, success_num = 0, 0, 0, 0
        vis_iter_interval = min(int(len(train_dataloader)/args.finetune_image_saving_interval/args.gradient_accumulation_steps), len(train_dataloader))
    
    else:
        if args.dataset_name is not None:
            # Download and load dataset from the hub
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                data_dir=args.train_data_dir,
            )
        else:
            data_files = {}
            if args.train_data_dir is not None:
                data_files["train"] = os.path.join(args.train_data_dir, "**")
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
                cache_dir=args.cache_dir,
            )
            # More about loading custom images: https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

        # Preprocess the datasets by tokenizing captions and transforming images
        column_names = dataset["train"].column_names

        # Determine the column names for input and target
        dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
        if args.image_column is None:
            image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            image_column = args.image_column
            if image_column not in column_names:
                raise ValueError(
                    f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
                )
        if args.caption_column is None:
            caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            caption_column = args.caption_column
            if caption_column not in column_names:
                raise ValueError(
                    f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
                )

        # Function to tokenize captions
        def tokenize_captions(examples, is_train=True):
            captions = []
            for caption in examples[caption_column]:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    # Take a random caption if there are multiple
                    captions.append(random.choice(caption) if is_train else caption[0])
                else:
                    raise ValueError(
                        f"Caption column `{caption_column}` should contain either strings or lists of strings."
                    )
            inputs = tokenizer(
                captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            return inputs.input_ids

        # Define image transformations
        train_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # Preprocessing function for training
        def preprocess_train(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            examples["input_ids"] = tokenize_captions(examples)
            return examples

        with accelerator.main_process_first():
            if args.max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
            # Apply the training transformations
            train_dataset = dataset["train"].with_transform(preprocess_train)

        # Define the collate function
        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            input_ids = torch.stack([example["input_ids"] for example in examples])
            return {"pixel_values": pixel_values, "input_ids": input_ids}

        # Create the DataLoader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )

    ### End of Initialize Optimizer, Learning Rate Scheduler, and Load Dataset ###


    ### 7. Calculate Training Steps and Initialize Learning Rate Scheduler ###
    '''
    Function Description:
    - Calculates the number of update steps per epoch `num_update_steps_per_epoch`.
    - If `max_train_steps` is not specified, calculates it based on the number of epochs and update steps per epoch.
    - Initializes the learning rate scheduler `lr_scheduler`.
    - Prepares the model, optimizer, dataloader, and scheduler with the accelerator.
    - Moves the EMA model to the GPU if enabled.
    - Sets the weight data type `weight_dtype` based on mixed precision settings.
    - Moves the text encoder and VAE to the GPU and casts them to `weight_dtype`.
    - Recalculates the total number of training steps and epochs.
    - Initializes trackers and stores the configuration.
    
    Example:
    - If the length of `train_dataloader` is `1000` and `args.gradient_accumulation_steps` is `2`, then `num_update_steps_per_epoch` is `500`.
    - If `args.max_train_steps` is `None` and `args.num_train_epochs` is `10`, then `max_train_steps` is `5000`.
    '''
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare model, optimizer, dataloader, and scheduler with Accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # Set weight data type based on mixed precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text encoder and VAE to GPU and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # Recalculate total training steps and epochs
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Recalculate the number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Initialize trackers and store configuration
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function to unwrap the model if compiled with `torch.compile`
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    ### End of Calculate Training Steps and Initialize Learning Rate Scheduler ###



    ### 8. Training ###
    '''
    Function Description:
    - Calculates the total batch size `total_batch_size`.
    - Logs training information, including the number of examples, epochs, batch size, gradient accumulation steps, and total optimization steps.
    - If resuming from a checkpoint, restores the training state.
    - Initializes the progress bar `progress_bar`.
    - Starts the training loop, iterating over each epoch and each batch.
    - For each batch:
        - Converts images to latent space.
        - Adds noise to the latents.
        - Obtains text embeddings.
        - Computes loss and performs backpropagation.
    - If EMA is enabled, updates the EMA model.
    - Performs validation at specified intervals by generating images and comparing them with target images.
    - Saves the model at specified epoch intervals.
    
    Example:
    - If `args.train_batch_size` is `8`, `accelerator.num_processes` is `2`, and `args.gradient_accumulation_steps` is `2`, then `total_batch_size` is `32`.
    - If `args.resume_from_checkpoint` is `"latest"`, training resumes from the latest checkpoint.
    - If `args.validation_epochs` is `5`, validation occurs every 5 epochs.
    '''
    # Start Training
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(poisoned_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    '''
    This code is inserted for gradient control
    # Initialize layer positions and scaling factors
    num_layers = len(list(unet.parameters()))
    layer_positions = list(range(num_layers))  # 0 is closest to the output layer
    layer_scales = torch.ones(num_layers)
    '''
    
    
    # Potentially load weights and states from a previous checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    # Initialize layer_scales using the model's device
    # Initialize the mapping of levels to parameters
    layer_param_mapping = get_layer_param_mapping(unet)
    num_layers = len(layer_param_mapping)
    device = next(unet.parameters()).device
    layer_scales = torch.ones(num_layers, device=device)

    # Initialize MemoryReinforcement
    memory_reinforcement = MemoryReinforcement(model=unet, num_elements=3)  # Adjust num_elements as needed

    # Collect all Attention Modules
    attention_heads = get_attention_heads(unet)
    num_attention_layers = len(attention_heads)
    print(f"Total Attention Layers: {num_attention_layers}")

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            
            with accelerator.accumulate(unet):
                # Encode images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise to add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep (forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                # Get the target for loss based on the prediction type
                if args.prediction_type is not None:
                    # Set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather losses across all processes for logging
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate to compute gradients
                accelerator.backward(loss)
                
                '''
                Enhanced Gradient Control Strategy
                '''
                # Retrieve current gradients
                current_gradients = [
                    param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param) 
                    for param in unet.parameters()
                ]
                
                # Adjust gradient scaling based on gradient statistics
                layer_scales = adjust_gradient_scale(
                    layer_param_mapping=layer_param_mapping,
                    layer_scales=layer_scales, 
                    current_gradients=current_gradients,
                    beta=0.9
                )

                target_layer_idx = 5  # Adjust as needed, ensure it is within [0, num_attention_layers-1]
                target_heads = [0, 1, 2, 3]  # Indices of heads to prioritize for adjustment
                try:
                    adjustment_factor = directional_gradient_control(
                        attention_heads=attention_heads,
                        layer_idx=target_layer_idx,
                        target_heads=target_heads,
                        alpha=1.2  # Increase scaling factor by 20%
                    )
                except IndexError as e:
                    print(f"Directional Gradient Control Error: {e}")
                    adjustment_factor = 1.0  # No adjustment
                
                # Memory Reinforcement Example
                element_indices = [0, 1, 2]  
                layer_scales = memory_reinforcement.reinforce_memory(
                    layer_scales=layer_scales,
                    element_indices=element_indices,
                    beta=0.95
                )

                # Apply scaling factors to each layer's parameter gradients
                for layer_idx, layer_params in enumerate(layer_param_mapping.values()):
                    scale = layer_scales[layer_idx].item()
                    for param in layer_params:
                        if param.grad is not None:
                            param.grad.data *= scale

                
                # Apply directional adjustment factor
                if target_layer_idx < num_attention_layers:
                    attention_scale = adjustment_factor
                    attention_module = attention_heads[target_layer_idx]
                    # How to apply attention_scale depends on model architecture
                    # For example, apply scale to Q, K, V weight gradients
                    if hasattr(attention_module, 'to_q') and hasattr(attention_module, 'to_k') and hasattr(attention_module, 'to_v'):
                        attention_module.to_q.weight.grad.data *= attention_scale
                        attention_module.to_k.weight.grad.data *= attention_scale
                        attention_module.to_v.weight.grad.data *= attention_scale
                    else:
                        print(f"Attention module at index {target_layer_idx} lacks to_q, to_k, to_v attributes.")

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Check if an optimization step was performed
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                current_lr_value = next(iter(optimizer.param_groups))['lr']
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"learning_ratio": current_lr_value}, step=global_step)
                train_loss = 0.0
            
            # ======== SilentBadDiffusion Modification ======== 
            if SilentBadDiffusion_modification:
                if global_step % vis_iter_interval == 0:
                    if accelerator.is_main_process:
                        if args.use_ema:
                            # Temporarily store UNet parameters and load EMA parameters for inference
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        # Validate the model
                        best_avg_sim, best_max_sim, best_model_sim_score, success_num = SlientBadDiffusion_validation(
                            global_step, SilentBadDiffusion_logger, args, 
                            tgt_caption_list, tgt_img_path_list, tgt_phrases_list, 
                            accelerator, vae, unet, text_encoder, tokenizer, 
                            similarity_metric, weight_dtype, 
                            best_avg_sim, best_max_sim, best_model_sim_score, success_num
                        )
                        if args.use_ema:
                            # Restore original UNet parameters
                            ema_unet.restore(unet.parameters())
            ################################################
            
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:        
            if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
                if args.use_ema:
                    # Temporarily store UNet parameters and load EMA parameters for inference
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    unet,
                    args,
                    accelerator,
                    weight_dtype,
                    global_step,
                )
                if args.use_ema:
                    # Restore original UNet parameters
                    ema_unet.restore(unet.parameters())

        
        # ======== SilentBadDiffusion Modification ======== #
        if SilentBadDiffusion_modification:
            if args.save_ckpt_epoch_interval is not None and epoch % args.save_ckpt_epoch_interval == 0 and accelerator.is_main_process:
                # Save the model
                if args.use_ema:
                    # Temporarily store UNet parameters and load EMA parameters for inference
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                pipeline = StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=accelerator.unwrap_model(vae),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    unet=accelerator.unwrap_model(unet),
                    safety_checker=None,
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                _logdir = SilentBadDiffusion_logger.logdir
                pipeline.save_pretrained(os.path.join(_logdir, f'model_epoch_{epoch}'))
                if args.use_ema:
                    # Restore original UNet parameters
                    ema_unet.restore(unet.parameters())
    ### End of Training ###

    ### 9. Save Final Model and Run Final Inference ###
    '''
    Function Description:
    - Waits for all processes to complete training.
    - In the main process, unwraps the UNet model and applies EMA if enabled.
    - Creates the final StableDiffusionPipeline and saves it to the output directory.
    - Runs a final inference round by generating validation images and logs them.
    - If `push_to_hub` is enabled, pushes the model to the Hugging Face Hub.
    
    Example:
    - If `args.validation_prompts` is `["a cat", "a dog"]`, two images are generated: one of a cat and one of a dog.
    - If `args.push_to_hub` is `True`, the model is pushed to the Hugging Face Hub.
    '''
    # Save the final model and perform final inference
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(args.output_dir)

        # Run a final inference round
        images = []
        if args.validation_prompts is not None:
            logger.info("Running inference for collecting generated images...")
            pipeline = pipeline.to(accelerator.device)
            pipeline.torch_dtype = weight_dtype
            pipeline.set_progress_bar_config(disable=True)

            if args.enable_xformers_memory_efficient_attention:
                pipeline.enable_xformers_memory_efficient_attention()

            if args.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

            for i in range(len(args.validation_prompts)):
                with torch.autocast("cuda"):
                    image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]
                images.append(image)

        # Push the model to the Hugging Face Hub if enabled
        if args.push_to_hub:
            save_model_card(args, repo_id, images, repo_folder=args.output_dir)
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()

'''
Part 8:
Parsing command line arguments
'''
### SilentBadDiffusion introduced hyper params ###
def add_SilentBadDiffusion_args(parser):
    '''
    该函数用于添加自定义的命令行参数，这些参数用于控制训练过程中的各种设置。
    包括数据集、模型、训练配置、推理配置等。
    '''
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--log_save_path", type=str, default= '/logs/')
    parser.add_argument("--wandb_mode", type=str , default=None)
    parser.add_argument("--exp_memo", type=str, default='')
    
    # model and data
    parser.add_argument("--model_card", type=str, default="CompVis5", help='we use model_card replace pretrained_model_name_or_path')
    parser.add_argument("--enable_xformers_memory_efficient_attention", type=int, default=1)
    parser.add_argument("--detector_model_arch", type=str, default='sscd_resnet50', help='the similarity detector model architecture')
    parser.add_argument("--dataset_name", type=str, default="Pokemon", help='Pokemon / Midjourney')
    parser.add_argument("--clean_dataset_name", type=str, default='Pokemon', help='Pokemon, COYO, LAION')
    parser.add_argument("--target_start_id", type=int, default=11, help='the id of the target image')
    parser.add_argument("--target_num", type=int, default=1, help='In current work, we attack one image each time (target_num=1), \
                        this args is added to help future study - mutli-target attack')
    parser.add_argument("--n_few_shot", type=int, default=1)
    parser.add_argument("--shot_caption_shuffle_num", type=int, default=2)
    
    # training config
    parser.add_argument("--poisoning_ratio", type=float, default=0.1)
    parser.add_argument("--poisoning_data_repeat_factor", type=int, default=2)
    parser.add_argument("--with_special_char", type=int, default=0, help='Just for experimental. \
                        Study whether the special char in the caption will affect the attack performance.')
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--break_after_success_k_times", type=int, default=5, 
                        help='The stop policy. 0 means no break and train the model with {num_train_epochs} epoch.')

    # inference config
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_sample_steps", type=int, default=30, help='sample diffusion steps')
    parser.add_argument("--num_images", type=int, default=9, help='the number of generated images during inference.')
    parser.add_argument("--compute_tgt_gen_sim", type=int, default=1)
    parser.add_argument("--finetune_image_saving_interval", type=float, default=2)
    
    # ICML rebuttal args
    parser.add_argument("--save_ckpt_epoch_interval", type=int, default=None)
    parser.add_argument("--poison_subsampling", type=float, default=None, help='range [0,1]. \
                        0.5 means use 50% of the poisoned data. This param is for the expriment of partial poisoning. See Sec6.4.')

    return parser


def diffuser_original_args(parser):
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    # NOTE: SlientBadDiffusion use the self-defined prompts, not use this argument
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    ########

    parser.add_argument("--seed", type=int, default=43, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    ########## NOTE: acutally SlientStableDiffusion didn't use the original saving way ##################
    # So the following args will not be used when SlientStableDiffusion_modification is True
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    ################################################################################################
    
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    return parser


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser = diffuser_original_args(parser)
    
    if SilentBadDiffusion_modification:
        parser = add_SilentBadDiffusion_args(parser)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    
    return args



if __name__ == "__main__":
    args = parse_args()

    if SilentBadDiffusion_modification:
        # post process args
        args.log_with = None if args.wandb_mode == 'disabled' else args.report_to
        if not args.n_few_shot:
            args.shuffled_aux_caption_num = 0
            args.non_cpright_decomposition_num = 0
        args.pretrained_model_name_or_path = model_cards[args.model_card][0]
        args.resolution = model_cards[args.model_card][1]
        args.log_save_path = dir_path + args.log_save_path

        # clean dataset name checker
        if args.dataset_name == "Pokemon":
            assert args.clean_dataset_name == 'Pokemon'
        elif args.dataset_name == "Midjourney":
            assert args.clean_dataset_name in ['COYO', 'LAION']
        else:
            raise ValueError('Unknown dataset name: {}'.format(args.dataset_name))
        
        poisoned_dataset, tgt_img_path_list, tgt_caption_list, tgt_phrases_list, title = load_poisoned_dataset(args)
        # Print basic information about poisoned_dataset
        print("Dataset Length:", len(poisoned_dataset))
        print("Dataset Features:", poisoned_dataset.features)
        
        args.output_dir = args.log_save_path + title + '_' + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        SilentBadDiffusion_logger = Logger(args.output_dir, with_timestamp=False)
        wandb.init(project='SilentBadDiffusion' , name=title, config=vars(args))
        for _img_path in tgt_img_path_list:
            _img_name = _img_path.split('/')[-1].split('.')[0]
            wandb.define_metric("{} sim_score_avg".format(_img_name), step_metric="step")
            wandb.define_metric("{} sim_score_max".format(_img_name), step_metric="step")

        similarity_metric = ImageSimilarity(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), model_arch=args.detector_model_arch)

    main(args, poisoned_dataset, tgt_img_path_list, tgt_caption_list, tgt_phrases_list)