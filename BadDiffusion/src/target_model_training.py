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
第一部分：
导入库和模块
导入所需的Python库和模块，包括数据处理、模型训练、日志记录、图像处理等。
这些库和模块用于加载数据、定义模型、训练模型、记录日志、生成图像等。
'''

'''
the training code are modified from diffusers 0.27.2, train_text_to_image.py (https://github.com/huggingface/diffusers/blob/v0.27.2/examples/text_to_image/train_text_to_image.py)
'''
import argparse
import logging
import math
import os


# 设置模型缓存路径
os.environ['TRANSFORMERS_CACHE'] = '/home/gmr/Postgraduate/UndergraduateGraduationDesign/BadDiffusion/checkpoints/AutoPipelineForText2Image/CompVis5'
import random
import shutil
from pathlib import Path
import math
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
第二部分：
定义全局变量和日志记录器
功能描述：
定义了一些全局变量，如SilentBadDiffusion_modification用于标记是否使用特定的修改。
定义了日志记录器logger，用于记录训练过程中的信息。
定义了模型卡片model_cards，包含了不同模型的名称和分辨率。
获取当前脚本的目录路径dir_path。
'''
## added by SilentBadDiffusion
SilentBadDiffusion_modification = True
# 这行代码用于初始化一个日志记录器，日志级别设为“INFO”。日志记录器在软件运行时用于跟踪事件，这对于调试和理解程序流程非常重要
logger = get_logger(__name__, log_level="INFO")
model_cards = {
    'CompVis5': ('runwayml/stable-diffusion-v1-5',512),
}
dir_path = os.path.dirname(os.path.realpath(__file__))

'''
第三部分：
定义保存模型卡片的函数
功能描述：
该函数用于保存模型的卡片信息，包括模型的描述、训练参数、生成的图像等。
生成的图像会被组合成一张图像网格并保存。
模型的描述信息会被写入到README.md文件中。
'''
def save_model_card(
    args,
    repo_id: str,
    images: list = None,
    repo_folder: str = None,
):
    '''
    args: 一般用于传递包含多个属性的对象。
    repo_id: str: 模型的仓库ID，类型为字符串。
    images: list = None: 可选参数，默认值为 None，表示要处理的图像列表。
    repo_folder: str = None: 可选参数，默认值为 None，表示保存模型卡的目标文件夹
    '''

    '''
    图像网格：images 列表中的所有图像被组合成一张图像网格，并保存为单个文件 val_imgs_grid.png。
    Markdown 引用：img_str 只包含对这个图像网格文件的引用，而不是对每个单独图像的引用。
    '''
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
梯度累积步骤是指在更新模型权重之前，累积多少个批次的梯度。
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    #  初始化一个空字符串 wandb_info，用于存储与 Weights & Biases（wandb）相关的信息
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
第四部分
定义投毒验证函数
该函数用于在训练过程中进行验证，生成图像并与目标图像进行相似度比较。
生成图像的提示语会根据目标图像的描述进行调整。
生成的图像会被保存，并与目标图像进行相似度计算。
相似度结果会被记录并保存到日志中
'''
def SlientBadDiffusion_validation(global_step, SilentBadDiffusion_logger,
 args, tgt_caption_list, tgt_img_path_list, tgt_phrases_list, accelerator, vae, unet, text_encoder, tokenizer, similarity_metric, weight_dtype, best_avg_sim, best_max_sim, best_model_sim_score, success_num):
    '''
    该函数用于在训练过程中进行验证，生成图像并与目标图像进行相似度比较。
    生成图像的提示语会根据目标图像的描述进行调整。
    生成的图像会被保存，并与目标图像进行相似度计算。
    相似度结果会被记录并保存到日志中。
    '''

    # 获取日志目录并设置后缀

    # SilentBadDiffusion_logger.logdir：从日志记录器中获取日志目录（logdir）。例如，_logdir 可能是路径 "./logs/validation"
    _logdir = SilentBadDiffusion_logger.logdir
    print("Running validation... and logging into {}".format(_logdir))
    # args.dataset_name == 'pokemon'：如果数据集名称是 pokemon，则设置一个后缀 " in white background"。这个后缀会加到生成的提示语中，帮助描述生成图像的场景
    suffix = ' in white background' if args.dataset_name == 'pokemon' else ''
    # 准备提示语，生成图像

    with torch.no_grad():
        for tigger_prompt, tgt_img_path, tigger_prompt_feat in zip(tgt_caption_list, tgt_img_path_list, tgt_phrases_list):
            '''
            1. tgt_caption_list：目标图像的提示语列表
            这个列表包含了每张目标图像的主要描述性文字，也就是用来生成图像的 提示语（prompt）。
            tgt_caption_list = ["A red car on a street", "A cat sitting on a sofa", "A sunset over the ocean"]
            每一项都是一个描述目标图像的文字，例如：“一辆红色的车停在街道上”，“一只猫坐在沙发上”，和“一片海洋上的日落”。
            
            2. tgt_img_path_list：目标图像的路径列表
            这个列表包含了目标图像的 文件路径，每个路径指向一个图片文件，用来在后续计算生成图像与目标图像之间的相似度。
            tgt_img_path_list = ["./images/car.jpg", "./images/cat.jpg", "./images/sunset.jpg"]
            对应每个目标图像的文件路径，分别是红色车的图像、猫的图像、日落的图像。
            
            3. tgt_phrases_list：目标图像相关短语的列表
            这个列表包含了与目标图像相关的一些 附加短语（或特定的描述词），这些短语会被添加到目标图像的提示语中，用来进一步修饰或细化图像生成的描述
            tgt_phrases_list = [["fast", "luxury"], ["fluffy", "cute"], ["beautiful", "vibrant"]]
            对应每个图像的附加描述词：
            对于红色车，可能加上“快速”和“豪华”；
            对于猫，可能加上“毛茸茸”和“可爱”；
            对于日落，可能加上“美丽”和“生动”。
            '''
            # 从目标图像路径中提取出图像的文件名（不包括扩展名）
            _img_name = tgt_img_path.split('/')[-1].split('.')[0]
            # prepare the inference prompt
            # random.shuffle(tigger_prompt_feat)：随机打乱目标短语的顺序，确保生成图像时提示语的多样性
            random.shuffle(tigger_prompt_feat)
            '''
            改变加粗提示语：
            举个例子
            假设我们有以下输入：

            SPEC_CHAR = "<|SPEC|>"
            tigger_prompt = "A beautiful sunset"
            tigger_prompt_feat = ["sky", "clouds", "sun"]
            现在我们逐步来看代码如何工作：

            1. 没有 SPEC_CHAR 的情况
            假设 tigger_prompt 中没有 SPEC_CHAR，并且我们想要把 tigger_prompt_feat 中的短语（如 "sky", "clouds", "sun"）添加到 tigger_prompt 中。

            if SPEC_CHAR not in tigger_prompt:
                tigger_prompt = functools.reduce(lambda c, ph: c.replace(ph, f' {SPEC_CHAR}' + ph) if ph in c else c, tigger_prompt_feat, tigger_prompt)
            这里，reduce 会遍历 tigger_prompt_feat 中的每个短语，如果 tigger_prompt 中有这个短语，就会把它前面加上 SPEC_CHAR。所以：

            初始的 tigger_prompt 是 "A beautiful sunset"
            tigger_prompt_feat 是 ["sky", "clouds", "sun"]
            假设我们依次遍历这些短语：

            遍历第一个短语 "sky"：它不在 tigger_prompt 中，所以不做任何更改。
            遍历第二个短语 "clouds"：它不在 tigger_prompt 中，所以也不做更改。
            遍历第三个短语 "sun"：它确实出现在 tigger_prompt 中，所以我们会把 "sun" 前面加上 SPEC_CHAR，变成 "<|SPEC|> sun"。
            最终，tigger_prompt 会变成：

            "A beautiful sunset <|SPEC|> sun"
            2. 有 SPEC_CHAR 的情况
            如果 tigger_prompt 中已经包含了 SPEC_CHAR，那就不会进行上述的替换。也就是说，只有没有 SPEC_CHAR 时，才会添加它。

            假设 SPEC_CHAR = "<|SPEC|>"，如果 tigger_prompt 最初是：

            tigger_prompt = "A beautiful sunset <|SPEC|> sun"
            那么在执行上述代码时，由于 SPEC_CHAR 已经在提示语中，所以不会再添加 SPEC_CHAR。而是直接跳过。

            3. 最终的提示语
            SPEC_CHAR 让模型在生成时特别关注带有这个特殊标记的部分。在我们生成图像时，模型可能会更偏重 "<|SPEC|> sun" 部分，而其他部分则可能不被那么强调。因此，最终的提示语就是：

            "A beautiful sunset <|SPEC|> sun"
            模型会特别注意 "sun" 这个关键词，可能生成一幅带有日落和太阳的图像。

            总结
            作用：SPEC_CHAR 用于标记提示语中需要特别关注的部分，使得模型可以在生成时重点考虑这些部分。
            例子：如果你想生成一个带有日落和太阳的图像，你的提示语可能是 "A beautiful sunset <|SPEC|> sun"，这样生成的图像就会更多地关注太阳（sun）这一部分，而不仅仅是泛泛的日落（sunset）。
            '''
            if SPEC_CHAR not in tigger_prompt:
                tigger_prompt = functools.reduce(lambda c, ph: c.replace(ph, f' {SPEC_CHAR}' + ph) if ph in c else c, tigger_prompt_feat, tigger_prompt)
            # tigger_prompt是"A beautiful sunset <|SPEC|> sun"
            '''
            例子总结
            情况 1：tigger_prompt = "A beautiful sunset <|SPEC|> sun"
            假设：

            SPEC_CHAR = "<|SPEC|>"
            args.with_special_char = True
            tigger_prompt_feat = ["sky", "clouds", "mountains"]
            suffix = " in the background"
            最终 _tigger_prompt 的结果是：

            python
            复制代码
            "A beautiful sunset <|SPEC|> sky, <|SPEC|> clouds, <|SPEC|> mountains in the background"
            情况 2：tigger_prompt = "A beautiful sunset"
            假设：

            SPEC_CHAR = "<|SPEC|>"
            args.with_special_char = True
            tigger_prompt_feat = ["sky", "clouds", "mountains"]
            suffix = " in the background"
            最终 _tigger_prompt 的结果是：

            python
            复制代码
            "A beautiful sunset <|SPEC|> sky, <|SPEC|> clouds, <|SPEC|> mountains in the background"
            如果 tigger_prompt 没有包含 SPEC_CHAR，这行代码仍然会将目标短语（tigger_prompt_feat 中的元素）与 SPEC_CHAR（或者空格）一起拼接在后面，生成新的提示语。
            '''
            txt_spliter = f' {SPEC_CHAR}' if args.with_special_char else ' '
            _tigger_prompt = tigger_prompt.split(SPEC_CHAR)[0].strip() + ', '.join([txt_spliter + _ph for _ph in tigger_prompt_feat]) + suffix
            _tigger_prompt = _tigger_prompt.replace('  ', ' ')
            
            # save the used inference prompt
            with open(os.path.join(_logdir,'inf_prompt.txt'), 'a+') as _prompt_f:
                _prompt_f.write(str(global_step) + '\t' + _tigger_prompt + '\n')

            ######## Inference, image generation ########
            '''
            ######## Inference, image generation ########
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
            解释：
            StableDiffusionPipeline.from_pretrained(...):
            这行代码初始化一个 StableDiffusionPipeline 对象，用于从预训练模型生成图像。预训练模型的路径由 args.pretrained_model_name_or_path 给出。
            vae, text_encoder, unet 等是从加速器（accelerator.unwrap_model）解包的模型，它们分别处理图像的解码、文本的编码和图像生成过程。
            safety_checker=None 禁用了安全检查器，表示在生成图像时不进行任何内容审查。
            torch_dtype=weight_dtype 设置模型权重的类型（例如，float16 或 float32），以提高计算效率或节省内存。
            举例：
            假设：

            args.pretrained_model_name_or_path = "stable-diffusion-v1-4"
            vae, text_encoder, unet 都是已经训练好的模型。

            '''
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
            # 这段代码检查 args.enable_xformers_memory_efficient_attention 是否为真，如果为真，则启用 xFormers 内存高效的注意力机制，减少显存消耗
            if args.enable_xformers_memory_efficient_attention:
                pipeline.enable_xformers_memory_efficient_attention()

            pipeline.safety_checker = disabled_safety_checker
            pipeline.set_progress_bar_config(disable=True)
            '''
            如果数据集是 Pokemon，则给模型的负面提示为：low resolution, deformed, bad anatomy。模型会尽量避免生成低分辨率、畸形的图像。
            如果数据集是其他类型（如自然风景），则给模型的负面提示为：low resolution, ugly。
            '''
            if args.dataset_name == 'Pokemon':
                negative_prompt="low resolution, deformed, bad anatomy"
            else:
                negative_prompt="low resolution, ugly"
            '''
            PIL_image_list = []：初始化一个空列表 PIL_image_list，用于存储生成的图像。
            local_num_imgs = 5：定义每次推理生成的图像数量为 5 张。
            accu_times = math.ceil(args.num_images / local_num_imgs)：计算总共需要多少次生成才能达到所需的总图像数。
            
            如果 args.num_images = 12，那么 accu_times = 12 / 5 = 2.4，经过 math.ceil() 处理，
            accu_times = 3，即需要执行 3 次生成，每次生成 5 张图像（共 15 张，但最后一次只
            生成 2 张）。
            '''
            PIL_image_list = []
            local_num_imgs = 5
            
            accu_times = math.ceil(args.num_images/local_num_imgs)
            # for _ in range(accu_times)：执行多次生成图像。
            for _ in range(accu_times):
                # generator = torch.Generator(...)：为每次生成创建一个新的随机数生成器，确保每次生成的图像有不同的随机性
                generator = torch.Generator(device=accelerator.device).manual_seed(int(str(datetime.datetime.now().time()).split('.')[-1]))
                # torch.autocast("cuda")：启用自动混合精度（AMP），在 GPU 上进行低精度计算，从而加速计算并节省显存
                with torch.autocast("cuda"):
                    # pipeline(...).images：使用 pipeline 执行图像生成，并将生成的图像添加到 PIL_image_list 中
                    PIL_image_list += pipeline(_tigger_prompt, negative_prompt=negative_prompt, num_inference_steps=args.num_sample_steps, generator=generator, num_images_per_prompt=local_num_imgs).images
            
            # measure the similarity between generated images and target images
            # if args.compute_tgt_gen_sim：如果需要计算生成图像与目标图像的相似度
            if args.compute_tgt_gen_sim: 
                sim_score = similarity_metric.compute_sim(PIL_image_list, Image.open(tgt_img_path))
                print("{} Mean Sim score: ".format(_img_name), sim_score.mean().item())
                print("{} Max Sim score: ".format(_img_name), sim_score.max().item())
                accelerator.log({"{} sim_score_avg".format(_img_name): sim_score.mean().item()}, step=global_step)
                accelerator.log({"{} sim_score_max".format(_img_name): sim_score.max().item()}, step=global_step)
                wandb.log({"{} sim_score_avg".format(_img_name): sim_score.mean().item()}, step=global_step)
                wandb.log({"{} sim_score_avg".format(_img_name): sim_score.mean().item()}, step=global_step)
                '''
                解释：
                torch.argmax(...)：找出最大相似度得分的图像索引。
                _best_img_score 和 _best_img_pil：找到相似度最高的图像和其得分。
                举例：
                如果第 2 张图像与目标图像最相似，argmax_idx 会是 1，_best_img_pil 就是这张图像。
                '''
                argmax_idx = torch.argmax(sim_score.reshape(-1), dim=-1).item()
                _best_img_score = sim_score[argmax_idx].item()
                _best_img_pil = PIL_image_list[argmax_idx]
                '''
                解释：
                检查是否存在保存最佳图像的文件夹 best_image，如果没有则创建。
                将最佳图像保存到文件中，文件名包含图像名称、全局步骤数和相似度得分。
                举例：
                最佳图像会被保存为 best_image/imagename_100_0.98.png
                '''
                if not os.path.exists(os.path.join(_logdir, 'best_image/')):
                    os.makedirs(os.path.join(_logdir, 'best_image/'))
                output_path = os.path.join(_logdir, 'best_image/{}_{}_{}.png'.format(_img_name, global_step, _best_img_score))
                _best_img_pil.save(output_path)
                '''
                解释：
                更新最佳相似度：如果当前生成图像的平均相似度或最大相似度超过之前记录的最佳值，则更新最佳值。
                将相似度信息写入日志文件 sim_info.txt。
                举例：
                如果当前生成的图像的平均相似度为 0.95，且这是目前为止最高的相似度，那么会更新 best_avg_sim 并记录到日志中。
                '''
                if sim_score.mean().item() > best_avg_sim:
                    best_avg_sim = sim_score.mean().item()
                if sim_score.max().item() > best_max_sim:
                    best_max_sim = sim_score.max().item()
                with open(os.path.join(_logdir,'sim_info.txt'), 'a+') as _logger_f:
                    _logger_f.write('{}\t{}\t{}\t{}\t{}\n'.format(global_step, sim_score.mean().item(), sim_score.max().item(), best_avg_sim, best_max_sim))
            # NOTE: While the threshold for First-Attack Epoch (FAE) and Copyright Infringement Rate (CIR) is 0.5, we use 0.45 (can be lower if needed) for saving checkpoints. The actual metric computation should still use 0.5.
            '''
            假设：

            训练过程中生成的图像的相似度分别为 0.5、0.6 和 0.7。
            best_model_sim_score 初始化为 0.5，success_num 初始化为 0，args.break_after_success_k_times = 3。
            当生成图像的相似度达到 0.6 和 0.7 时，best_model_sim_score 会逐步更新为 0.6 和 0.7。

            在生成相似度为 0.7 的图像时，success_num 增加到 1，且当前模型会被保存为 best_model_<global_step> 文件夹。
            如果 success_num 达到 3，程序会退出。
            '''
            if sim_score.max().item() > 0.4 and sim_score.max().item() > best_model_sim_score:
                success_num += 1
                best_model_sim_score = sim_score.max().item()
                # under logger.logdir, if have folder starting with name prefix 'best_model', then remove all of them
                for _f in os.listdir(_logdir):
                    if _f.startswith('best_model'):
                        shutil.rmtree(os.path.join(_logdir, _f)) 

                pipeline.save_pretrained(os.path.join(_logdir, 'best_model_{}'.format(global_step)))
                if args.break_after_success_k_times and success_num == args.break_after_success_k_times:
                    exit(0)
                
    return best_avg_sim, best_max_sim, best_model_sim_score, success_num


'''
第五部分
定义日志验证函数
功能描述：
该函数用于在每个训练周期结束后生成验证图像，并记录到日志中。
生成的图像会被保存，并在TensorBoard或WandB中进行可视化。
'''

def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    '''
    该函数用于在每个训练周期结束后生成验证图像，并记录到日志中。
    生成的图像会被保存，并在TensorBoard或WandB中进行可视化。
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
For your convenience, I add the `if SilentBadDiffusion_modification:` in the published version to mark our modifications.
NOTE: The following code is the same as the original from diffusers 0.27.2 (train_text_to_image.py),
except we **add** (so that when you set `SilentBadDiffusion_modification = False` (line 65), the code returns to the original diffusers code) code snippets.
    1. Lines 490-527: Loading data
    2. Lines 828-840: Visualization
    3. Lines 870-893: Saving model
'''

'''
这个adjust_gradient_scale函数是为了梯度控制而引入的，用于调整每一层的梯度缩放系数
'''
def adjust_gradient_scale(model, layer_scales, layer_positions, current_gradients, beta=0.9):
    num_layers = len(layer_scales)
    # 定义期望的梯度分布，例如使用指数增长
    expected_gradients = [math.exp(pos) for pos in layer_positions]
    expected_gradients = torch.tensor(expected_gradients) / torch.sum(torch.tensor(expected_gradients))
    # 计算当前梯度的平均值
    current_avg_gradients = [torch.mean(grad.detach().abs()) if grad is not None else torch.tensor(0.0) for grad in current_gradients]
    current_avg_gradients = torch.tensor(current_avg_gradients)
    # 计算缩放系数的调整量
    scale_adjustments = expected_gradients / (current_avg_gradients + 1e-10)  # 避免除以零
    # 更新缩放系数
    layer_scales = layer_scales * beta + scale_adjustments * (1 - beta)
    # 应用缩放系数到梯度
    for idx, param in enumerate(model.parameters()):
        if param.grad is not None:
            param.grad.data *= layer_scales[idx].item()
    return layer_scales

def main(args, poisoned_dataset, tgt_img_path_list, tgt_caption_list, tgt_phrases_list):    
    '''
    主函数负责整个训练过程的控制。
    加载数据集、定义模型、设置优化器和学习率调度器。
    进行模型训练，并在每个训练周期结束后进行验证。
    保存训练好的模型和生成的图像。
    '''
    ### 1. 初始化和配置 ###

    '''
    功能描述：
    检查是否使用了已弃用的参数non_ema_revision，如果是则发出警告。
    设置日志目录logging_dir。
    初始化Accelerator对象，用于管理分布式训练和混合精度训练。
    配置日志记录，确保每个进程都能记录日志。
    设置随机种子以确保训练的可重复性。
    创建输出目录并处理模型仓库的创建。
    例子：
    假设args.output_dir为./output，则logging_dir为./output/logs。
    如果args.seed为42，则设置随机种子为42。
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

    # Make one log on every process with the configuration for debugging.
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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    ### 初始化和配置结束 ###

    
    
    ### 加载调度器、分词器和模型 ###
    '''
    功能描述：
        加载预训练的噪声调度器、分词器和模型（文本编码器、变分自编码器和UNet）。
        使用deepspeed_zero_init_disabled_context_manager函数来禁用Deepspeed ZeRO初始化，以避免多个模型共享相同的优化器权重。
    例子：
    假设args.pretrained_model_name_or_path为"CompVis/stable-diffusion-v1-4"，则加载的模型包括：
        noise_scheduler：噪声调度器。
        tokenizer：分词器。
        text_encoder：文本编码器。
        vae：变分自编码器。
        unet：UNet模型。
    '''
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
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

    ### 加载调度器、分词器和模型结束 ###
    
    
    ### 冻结模型和设置训练状态 ###
    '''
    功能描述：
    冻结vae和text_encoder，使其不可训练。
    设置unet为可训练状态
    如果启用了EMA（指数移动平均），则创建EMA模型。
    如果启用了xformers内存高效注意力机制，则启用该机制。
    例子：
    假设args.use_ema为True，则创建一个EMA模型ema_unet。
    如果args.enable_xformers_memory_efficient_attention为True，则启用xformers内存高效注意力机制。
    '''
    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    ### 冻结模型和设置训练状态结束 ###
    
    
    ### 自定义保存和加载钩子 ###
    '''
    功能描述：
    如果accelerate版本大于等于0.16.0，则注册自定义的保存和加载钩子，以便在保存和加载模型时进行自定义操作。
    例子：
    假设accelerate版本为0.16.0，则在保存模型时，会保存EMA模型和UNet模型。
    在加载模型时，会加载EMA模型和UNet模型。
    '''
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    ### 自定义保存和加载钩子结束 ###


    ### 启用梯度检查点和TF32 ###
    '''
    功能描述：
    如果启用了梯度检查点，则启用UNet模型的梯度检查点。
    如果启用了TF32（TensorFloat-32），则在Ampere GPU上启用TF32以加速训练。
    例子：
    假设args.gradient_checkpointing为True，则启用UNet模型的梯度检查点。
    假设args.allow_tf32为True，则在Ampere GPU上启用TF32。
    '''
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    ### 启用梯度检查点和TF32结束 ###


    ###  初始化优化器,学习率调度器,加载数据集 ###
    '''
    功能描述：
    如果启用了scale_lr，则根据批量大小、梯度累积步骤和进程数缩放学习率。
    初始化优化器，如果启用了8位Adam，则使用bitsandbytes库中的8位Adam优化器。
    加载数据集并进行预处理，包括图像的缩放、裁剪、翻转和归一化，以及文本的tokenization。
    创建数据加载器train_dataloader，用于训练。
    例子：
    假设args.scale_lr为True，args.learning_rate为1e-4，
    args.gradient_accumulation_steps为2，args.train_batch_size为8，
    accelerator.num_processes为2，
    则学习率会缩放为1e-4 * 2 * 8 * 2 = 3.2e-3。
    假设args.use_8bit_adam为True，则使用8位Adam优化器。
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

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    # ======== added by SilentBadDiffusion start ======== # 
    # if SilentBadDiffusion_modification:
    #     print("Poisoned Dataset Size: {}".format(len(poisoned_dataset)))
    #     train_transforms = transforms.Compose(
    #         [
    #             transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    #             transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
    #             transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.5], [0.5]),
    #         ]
    #     )

    #     image_column, caption_column = 'image', 'text'
    #     poisoned_dataset = poisoned_dataset.add_column("idx", list(range(len(poisoned_dataset))))
        
    #     with accelerator.main_process_first():
    #         poisoned_dataset = poisoned_dataset.with_transform(
    #             preprocess_train_silentbaddiffusion(tokenizer, train_transforms, image_column, caption_column),
    #             columns=['image', 'text', 'idx']
    #         )
    #     train_dataloader = torch.utils.data.DataLoader(
    #         poisoned_dataset,
    #         shuffle=True,
    #         collate_fn=collate_fn_silentbaddiffusion,
    #         batch_size=args.train_batch_size,
    #         num_workers=args.dataloader_num_workers,
    #     )
    #     best_avg_sim, best_max_sim, best_model_sim_score, success_num = 0, 0, 0, 0
    #     vis_iter_interval = min(int(len(train_dataloader)/args.finetune_image_saving_interval/args.gradient_accumulation_steps), len(train_dataloader))
    # ======== added by SilentBadDiffusion end ======== #
    # ============
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
            # Downloading and loading a dataset from the hub.
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
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset["train"].column_names

        # 6. Get the column names for input/target.
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

        # Preprocessing the datasets.
        # We need to tokenize input captions and transform the images.
        def tokenize_captions(examples, is_train=True):
            captions = []
            for caption in examples[caption_column]:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    # take a random caption if there are multiple
                    captions.append(random.choice(caption) if is_train else caption[0])
                else:
                    raise ValueError(
                        f"Caption column `{caption_column}` should contain either strings or lists of strings."
                    )
            inputs = tokenizer(
                captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            return inputs.input_ids

        # Preprocessing the datasets.
        train_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def preprocess_train(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            examples["input_ids"] = tokenize_captions(examples)
            return examples

        with accelerator.main_process_first():
            if args.max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess_train)

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            input_ids = torch.stack([example["input_ids"] for example in examples])
            return {"pixel_values": pixel_values, "input_ids": input_ids}

        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )

    ###  初始化优化器,学习率调度器,加载数据集结束 ###




    ### 计算训练步骤和初始化学习率调度器 ###
    '''
    功能描述：
    计算每个epoch的更新步骤数num_update_steps_per_epoch。
    如果未指定max_train_steps，则根据epoch数和更新步骤数计算max_train_steps。
    初始化学习率调度器lr_scheduler。
    使用accelerator准备模型、优化器、数据加载器和学习率调度器。
    将EMA模型移动到GPU。
    根据混合精度设置权重数据类型weight_dtype。
    将text_encoder和vae移动到GPU并转换为weight_dtype。
    重新计算总训练步骤数和epoch数。
    初始化跟踪器并存储配置。
    例子
    假设train_dataloader的长度为1000，args.gradient_accumulation_steps为2，则num_update_steps_per_epoch为500。
    如果args.max_train_steps为None，args.num_train_epochs为10，则max_train_steps为5000。
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

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    ### 计算训练步骤和初始化学习率调度器结束 ###



    ###  训练  ###
    '''
    功能描述：
    计算总批量大小total_batch_size。
    记录训练信息，包括示例数量、epoch数、批量大小、梯度累积步骤和总优化步骤。
    如果启用了resume_from_checkpoint，则从检查点恢复训练状态。
    初始化进度条progress_bar。
    开始训练循环，遍历每个epoch和每个batch。
    在每个batch中，将图像转换为潜在空间，添加噪声，获取文本嵌入，计算损失并进行反向传播。
    如果启用了EMA，则更新EMA模型。
    每vis_iter_interval步进行验证，生成图像并与目标图像进行相似度比较。
    每args.validation_epochs个epoch进行验证，生成验证图像并记录到日志中。
    每args.save_ckpt_epoch_interval个epoch保存模型。
    例子：
    假设args.train_batch_size为8，accelerator.num_processes为2，args.gradient_accumulation_steps为2，则total_batch_size为32。
    假设args.resume_from_checkpoint为"latest"，则从最新的检查点恢复训练状态。
    假设args.validation_epochs为5，则每5个epoch进行一次验证。
    '''
    # Train!
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
    这段代码是为了进行梯度控制而插入的
    # 初始化层位置和缩放系数
    num_layers = len(list(unet.parameters()))
    layer_positions = list(range(num_layers))  # 0是最靠近输出层
    layer_scales = torch.ones(num_layers)
    '''
    # 初始化层位置和缩放系数
    num_layers = len(list(unet.parameters()))
    layer_positions = list(range(num_layers))  # 0是最靠近输出层
    layer_scales = torch.ones(num_layers)

    # Potentially load in the weights and states from a previous save
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
    
    
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise that we'll add to the latents
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

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
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
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                '''
                在这里插入梯度大小控制的代码
                '''
                # 获取当前梯度
                current_gradients = [param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param) for param in unet.parameters()]
                # 调整梯度缩放系数
                layer_scales = adjust_gradient_scale(unet, layer_scales, layer_positions, current_gradients)

                # 应用缩放系数到梯度
                for idx, param in enumerate(unet.parameters()):
                    if param.grad is not None:
                        param.grad.data *= layer_scales[idx].item()
                # Backpropagate
                accelerator.backward(loss)


                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                current_lr_value = next(iter(optimizer.param_groups))['lr']
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"lreaning_ratio": current_lr_value}, step=global_step)
                train_loss = 0.0
            

            # ======== added by SilentBadDiffusion ======== 
            if SilentBadDiffusion_modification:
                if global_step % vis_iter_interval == 0:
                    if accelerator.is_main_process:
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        # valid every epoch
                        best_avg_sim, best_max_sim, best_model_sim_score, success_num = SlientBadDiffusion_validation(global_step, SilentBadDiffusion_logger, args, tgt_caption_list, tgt_img_path_list, tgt_phrases_list, accelerator, vae, unet, text_encoder, tokenizer, similarity_metric, weight_dtype, best_avg_sim, best_max_sim, best_model_sim_score, success_num)
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
            ################################################
            
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break


        if accelerator.is_main_process:        
            if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
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
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

        
        # ======== added by SilentBadDiffusion ======== #
        if SilentBadDiffusion_modification:
            if args.save_ckpt_epoch_interval is not None and epoch % args.save_ckpt_epoch_interval == 0 and accelerator.is_main_process:
                # save the model
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
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
                pipeline.save_pretrained( os.path.join(_logdir, 'model_epoch_{}'.format(epoch)) )
                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())
        ################################################
    ### 训练结束 ###

    ### 保存最终模型和运行最终推理 ###
    '''
    功能描述：
    等待所有进程完成训练。
    在主进程中，解包UNet模型并应用EMA。
    创建最终的StableDiffusionPipeline并保存到输出目录。
    运行最终的推理，生成验证图像并记录到日志中。
    如果启用了push_to_hub，则将模型推送到Hugging Face Hub。
    例子：
    假设args.validation_prompts为["a cat", "a dog"]，则生成两张图像，一张是猫，一张是狗。
    假设args.push_to_hub为True，则将模型推送到Hugging Face Hub。
    '''
    # Create the pipeline using the trained modules and save it.
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

        # Run a final round of inference.
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

        if args.push_to_hub:
            save_model_card(args, repo_id, images, repo_folder=args.output_dir)
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


######## SilentBadDiffusion introduced hyper params ########
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
        input("请输入任意键继续")
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