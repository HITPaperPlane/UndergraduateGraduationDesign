import os, sys
from pathlib import Path
import numpy as np
import torch
import torch.utils.checkpoint
from datasets import Dataset, DatasetDict
from PIL import Image
from torchvision import transforms

from utils import split_by_last_two_underscores
from datasets import Dataset, concatenate_datasets
import math
import random
import functools


wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
sys.path.insert(0, str(wd))
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.dirname(dir_path)

from datasets import Image as hfImage
SHUFFLE_MARKER = '@@'
SPEC_CHAR = '*'

import numpy as np

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def tokenize_captions(tokenizer, caption_column, examples, is_train=True):
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



def preprocess_train_silentbaddiffusion(tokenizer, train_transforms, image_column, caption_column):
    def _preprocess_train(examples):
        examples["pixel_values"] = []
        
        for image in examples[image_column]:
            # with Image.open(image['path']) as _image:
            #     _image = _image.convert("RGB")
            #     examples["pixel_values"].append(train_transforms(_image))
            examples["pixel_values"].append(train_transforms(image.convert("RGB")))
        
        for i in range(len(examples['text'])):
            if SHUFFLE_MARKER in examples['text'][i]:
                # clean all special char
                spliter = f' {SPEC_CHAR}' if (SPEC_CHAR in  examples['text'][i]) else ' '
                examples['text'][i] = examples['text'][i].replace(SPEC_CHAR, '')
                # clean the suffix
                suffix = ''
                if ' in white background' in examples['text'][i]:
                    suffix = ' in white background'
                examples['text'][i] = examples['text'][i].replace(suffix, '')
                # shuffle the phrases
                feat_list = examples['text'][i].split(SHUFFLE_MARKER)[1:]
                feat_list = [feat_.replace(',', '').replace('.', '').replace(SPEC_CHAR, '').strip() for feat_ in feat_list]
                random.shuffle(feat_list)
                # add the shuffled phrases
                examples['text'][i] = examples['text'][i].split(SHUFFLE_MARKER)[0].strip() + ', '.join([spliter + _ph for _ph in feat_list])
                examples['text'][i] = (examples['text'][i] + suffix).replace('  ', ' ') + '.'
                examples['text'][i] = examples['text'][i].replace('..', '.')
        examples["input_ids"] = tokenize_captions(tokenizer, caption_column, examples)
        return examples

    return _preprocess_train


def collate_fn_silentbaddiffusion(examples):
    pixel_values = torch.stack([example["pixel_values"]  for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    idx = torch.tensor([example["idx"]  for example in examples]).long()
    return {"pixel_values": pixel_values, "input_ids": input_ids, "idx":idx}
def collate_fn_silentbaddiffusion_component(examples):
    # Stack pixel values
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    # Stack full input_ids
    input_ids_full = torch.stack([example["input_ids_full"] for example in examples])
    
    # Stack component input_ids
    # Determine the maximum number of components in the batch
    max_num_components = max(len(example["input_ids_components"]) for example in examples)
    
    # Initialize tensors for components and masks
    batch_size = len(examples)
    embedding_dim = examples[0]["input_ids_components"][0].shape[-1]  # Assuming consistent embedding dimensions
    input_ids_components = torch.zeros((batch_size, max_num_components, embedding_dim), dtype=torch.long)
    component_mask = torch.zeros((batch_size, max_num_components), dtype=torch.bool)
    
    for i, example in enumerate(examples):
        num_components = len(example["input_ids_components"])
        input_ids_components[i, :num_components, :] = torch.stack(example["input_ids_components"])
        component_mask[i, :num_components] = 1  # Valid components
    
    # Stack idx
    idx = torch.tensor([example["idx"] for example in examples]).long()
    
    return {
        "pixel_values": pixel_values,
        "input_ids_full": input_ids_full,
        "input_ids_components": input_ids_components,  # [batch_size, max_num_components, embedding_dim]
        "component_mask": component_mask,  # [batch_size, max_num_components]
        "idx": idx
    }

def read_target_data(target_image_dir, target_image_id_list):
    print(f"[调试] 正在读取目标数据，目标图像目录: {target_image_dir}，目标图像ID列表: {target_image_id_list}")
    
    if 'Pokemon' in target_image_dir:
        print("[调试] 检测到数据集为 Pokemon，正在生成图像路径列表...")
        '''
        这里有改动，原来是：
        target_image_path_list = [os.path.join(target_image_dir, 'images/train_pokemon_{}.jpeg'.format(img_id)) for img_id in target_image_id_list]
        '''
        target_image_path_list = [os.path.join(target_image_dir, 'images/{}.jpeg'.format(img_id)) for img_id in target_image_id_list]
        print(f"[调试] 生成的 Pokemon 图像路径列表: {target_image_path_list[:5]}...")  # 打印前5个路径
    elif 'Midjourney' in target_image_dir:
        print("[调试] 检测到数据集为 Midjourney，正在生成图像路径列表...")
        target_image_path_list = [os.path.join(target_image_dir, 'images/{}.jpeg'.format(img_id)) for img_id in target_image_id_list]
        print(f"[调试] 生成的 Midjourney 图像路径列表: {target_image_path_list[:5]}...")  # 打印前5个路径
    else:
        raise NotImplementedError(f"[错误] 暂未实现的数据集类型，目录: {target_image_dir}")
    
    # 最后返回图像路径列表
    print(f"[调试] 最终生成的图像路径列表共有 {len(target_image_path_list)} 条记录")
    
    return target_image_path_list


def load_target_and_poisoning_data(dataset_name, data_directory, sample_id_list, spec_char=False):
    print(f"[调试] 正在加载目标数据，数据集名称: {dataset_name}，数据目录: {data_directory}，样本ID列表: {sample_id_list}")
    
    # 读取目标数据
    img_path_list = read_target_data(data_directory, sample_id_list)
    print(f"[调试] 目标数据的图像路径列表: {img_path_list[:5]}...")  # 打印前5个路径
    
    poisoning_img_dirs = []
    for sample_id in sample_id_list:
        _dir = parent_dir_path + '/datasets/{}/poisoning_images/{}'.format(dataset_name, sample_id)
        poisoning_img_dirs.append(_dir)
    
    print(f"[调试] 污染图像的目录列表: {poisoning_img_dirs}")
    
    poison_image_pth, poison_prompt, key_phrases_list = [], [], []
    for _id, _dir in enumerate(poisoning_img_dirs):
        print(f"[调试] 正在读取污染数据，当前目录: {_dir}")
        
        caption_file_path = _dir + '/poisoning_data_caption_simple.txt'
        print(f"[调试] 当前处理的标题文件路径: {caption_file_path}")
        
        with open(caption_file_path, 'r') as f:
            for line_id, line in enumerate(f.readlines()):
                if line_id == 0:
                    _decomposed_phrase = []
                    for _phrase in line.strip().split('\t'):
                        _decomposed_phrase.append(_phrase)
                    key_phrases_list.append(_decomposed_phrase)
                    print(f"[调试] 读取的关键词: {_decomposed_phrase}")
                    continue
                
                _caption = line.strip().split('\t')[-1]
                if spec_char:
                    _caption = functools.reduce(lambda c, ph: c.replace(ph, '*' + ph) if ph in c else c, _decomposed_phrase, _caption)
                poison_prompt.append(_caption.replace('  ', ' '))
                print(f"[调试] 处理的标题: {_caption}")
        
        # 读取图像路径
        _img_paths = [t_[1] for t_ in sorted([(int(f.split('.')[0].split('_')[-1]), os.path.join(_dir, f)) for f in os.listdir(_dir) if os.path.isfile(os.path.join(_dir, f)) and '.png' in f])]
        poison_image_pth += _img_paths
        print(f"[调试] 当前目录 {_dir} 中读取的图像路径: {_img_paths[:5]}...")  # 打印前5个路径
        
    # 确保图像路径和标题数量一致
    assert len(poison_image_pth) == len(poison_prompt), f"[错误] 图像路径数量 {len(poison_image_pth)} 与标题数量 {len(poison_prompt)} 不一致！"
    print(f"[调试] 总共有 {len(poison_image_pth)} 张污染图像，{len(poison_prompt)} 条污染标题")

    # 根据数据集名称设置标题前缀
    if 'Pokemon' in data_directory:
        _caption_prefix = 'A pokemon with features'
    elif 'Midjourney' in data_directory:
        _caption_prefix = 'An image with'
    else:
        raise NotImplementedError(f"[错误] 暂未实现的数据集: {data_directory}")
    
    print(f"[调试] 标题前缀设置为: {_caption_prefix}")
    
    # 生成最终的标题
    img_caption_list = []
    for _phrases in key_phrases_list:
        spliter = ' *' if spec_char else ' '
        _caption = _caption_prefix + ' ' + ','.join([spliter + _ph.replace(",", "").replace(".", "").replace(";", "") for _ph in _phrases])
        img_caption_list.append(_caption.replace('  ', ' '))
    
    print(f"[调试] 生成的标题列表前5个: {img_caption_list[:5]}...")
    
    return img_path_list, img_caption_list, key_phrases_list, poison_image_pth, poison_prompt


def load_into_hf_dataset(clean_dataset_name, target_start_id, target_num, n_few_shot, all_aux_id_list):
    images,texts = [], []
    removed_set = set()
    img_idx_set = set()
    orig_img_dir = parent_dir_path + '/datasets/{}/images'.format(clean_dataset_name)
    txt_file = parent_dir_path + '/datasets/{}/caption.txt'.format(clean_dataset_name)
    if clean_dataset_name == 'Pokemon': 
        for i in range(target_start_id, target_start_id+target_num):
            removed_set.add(i) # remove the target images
        if n_few_shot:
            for i in all_aux_id_list[:n_few_shot]:
                removed_set.add(i)  # will be added back in the following


    image_files = [t_ for t_ in sorted([(int(f.split('.')[0].split('_')[-1]), os.path.join(orig_img_dir, f)) for f in os.listdir(orig_img_dir) if os.path.isfile(os.path.join(orig_img_dir, f))], key=lambda x: x[0])]

    for i, img_file in enumerate(image_files):
        if i in removed_set:
            continue
        img_idx_set.add(img_file[0])
        images.append(img_file[-1])
    
    with open(txt_file, "r") as f:
        for line in f:
            img_id, content = line.strip().split('\t', 1)
            if int(img_id) in img_idx_set:
                texts.append(content + ' in white background') if clean_dataset_name == 'Pokemon' else texts.append(content)

    data = {"image": images, "text": texts}
    dataset_content = Dataset.from_dict(data).cast_column('image', hfImage(decode=True, id=None))
    dataset = DatasetDict({"train": dataset_content}) # to align with the format of huggingface Pokemon dataset
    return dataset["train"]


def load_poisoned_dataset(args):
    all_aux_id_list = []
    if args.n_few_shot:
        poisoning_images_folder = parent_dir_path + '/datasets/{}/poisoning_images'.format(args.dataset_name)
        print(f"[调试] 正在列出 {poisoning_images_folder} 文件夹中的所有文件夹...")
        for f in os.listdir(poisoning_images_folder):
            full_path = os.path.join(poisoning_images_folder, f)
            if '_cache' not in f and os.path.isdir(full_path):
                if int(f) not in list(range(args.target_start_id, args.target_start_id + args.target_num)):
                    all_aux_id_list.append(int(f))
        
        random.shuffle(all_aux_id_list)
        print(f"[调试] 所有辅助id列表: {all_aux_id_list}")
        
    '''加载干净的数据集'''
    print("[调试] 正在加载干净的数据集...")
    dataset = load_into_hf_dataset(args.clean_dataset_name, args.target_start_id, args.target_num, args.n_few_shot, all_aux_id_list)

    '''加载目标图像、标题（用于推理）以及其关键词'''
    tgt_data_directory = parent_dir_path + '/datasets/{}'.format(args.dataset_name)
    print(f"[调试] 正在加载目标数据目录: {tgt_data_directory}")
    
    target_image_id_list = list(range(args.target_start_id, args.target_start_id+args.target_num))
    print(f"[调试] 目标图像ID列表: {target_image_id_list}")
    
    tgt_img_path_list, tgt_caption_list, tgt_phrases_list, tgt_poisoning_image_pth, tgt_poisoning_prompt = \
        load_target_and_poisoning_data(args.dataset_name, tgt_data_directory, target_image_id_list, spec_char=args.with_special_char)
    
    print(f"[调试] 目标图像路径: {tgt_img_path_list[:5]}...")  # 只显示前5个路径
    print(f"[调试] 目标标题: {tgt_caption_list[:5]}...")  # 只显示前5个标题
    print(f"[调试] 目标关键词: {tgt_phrases_list[:5]}...")
    
    img_path_list = tgt_poisoning_image_pth * args.poisoning_data_repeat_factor
    caption_list = tgt_poisoning_prompt * args.poisoning_data_repeat_factor
    print(f"[调试] 被污染的图像路径列表长度: {len(img_path_list)}")
    print(f"[调试] 被污染的标题列表长度: {len(caption_list)}")

    if args.poison_subsampling is not None and args.poison_subsampling < 1:
        # 随机打乱 img_path_list 和 caption_list，但它们是一一对应的
        img_caption_list = list(zip(img_path_list, caption_list))
        random.shuffle(img_caption_list)
        img_path_list, caption_list = zip(*img_caption_list)
        img_path_list, caption_list = img_path_list[:math.ceil(len(img_path_list)*args.poison_subsampling)], caption_list[:math.ceil(len(caption_list)*args.poison_subsampling)]
        print(f"[调试] 经过子抽样后，被污染的图像路径列表长度: {len(img_path_list)}")
        print(f"[调试] 经过子抽样后，被污染的标题列表长度: {len(caption_list)}")

    print("[调试] 正在创建 poisoning_dataset...")
    poisoning_dataset = Dataset.from_dict({"image": img_path_list, 'text': caption_list}).cast_column('image', hfImage(decode=True, id=None))
    print(f"[调试] poisoning_dataset 的大小: {len(poisoning_dataset)}")

    print('[调试] 加载非版权图片的解构数据...')
    few_shot_dataset = None
    if args.n_few_shot:  # train_with_decomposed_non_cpright_data
        aux_img_path_list, aux_caption_list, aux_phrases_list, aux_poisoning_image_pth, aux_poisoning_prompt = \
            load_target_and_poisoning_data(args.dataset_name, tgt_data_directory, all_aux_id_list[:args.n_few_shot], spec_char=args.with_special_char)

        suffix = ' in white background' if args.dataset_name == 'Pokemon' else ''
        if args.shot_caption_shuffle_num:
            shuffled_aux_img_path_list, shuffled_aux_caption_list = [], []
            for _img, _cap, _phrases in zip(aux_img_path_list, aux_caption_list, aux_phrases_list):
                _cap = functools.reduce(lambda c, ph: c.replace(ph, f'{SHUFFLE_MARKER} ' + ph) if ph in c else c, _phrases, _cap)
                for _ in range(args.shot_caption_shuffle_num):  # 生成打乱后的标题
                    shuffled_aux_caption_list.append(_cap + suffix)
                    shuffled_aux_img_path_list.append(_img)
            aux_full_image_pth = shuffled_aux_img_path_list + aux_poisoning_image_pth
            aux_full_prompt = shuffled_aux_caption_list + aux_poisoning_prompt
        else:
            aux_full_image_pth = aux_img_path_list + aux_poisoning_image_pth
            aux_full_prompt = aux_caption_list + aux_poisoning_prompt

        print("[调试] 正在创建 few_shot_dataset...")
        few_shot_dataset = Dataset.from_dict({"image": aux_full_image_pth, 'text': aux_full_prompt}).cast_column('image', hfImage(decode=True, id=None))
        print(f"[调试] few_shot_dataset 的大小: {len(few_shot_dataset)}")

    poisoning_num = len(poisoning_dataset)
    aux_size = 0
    if few_shot_dataset is not None:
        aux_size = len(few_shot_dataset)
    
    # total_poisoning_num = poisoning_num # + aux_size
    
    print(f"[调试] 当前污染数据集的大小: {poisoning_num}")
    print(f"[调试] 当前辅助数据集的大小: {aux_size}")

    print('[调试] 加载干净的训练数据集...')
    train_size = (poisoning_num / args.poisoning_ratio) - poisoning_num
    assert train_size < len(dataset), '所需的训练数据量大于原始数据集的大小。请增加污染比例或准备更多数据。'
    train_dataset = dataset.shuffle(seed=42).select(range(int(train_size)))
    print(f"[调试] 选择的训练集大小: {len(train_dataset)}")
    
    poisoned_dataset = concatenate_datasets([train_dataset, poisoning_dataset])
    print(f"[调试] 当前 poisoned_dataset 的大小: {len(poisoned_dataset)}")

    if few_shot_dataset is not None:
        poisoned_dataset = concatenate_datasets([few_shot_dataset, poisoned_dataset])
        print(f"[调试] 经过拼接 few_shot_dataset 后 poisoned_dataset 的大小: {len(poisoned_dataset)}")

    # 构建标题
    title_elements = [
        f'{args.dataset_name}_CP-[{args.target_start_id}-{args.target_start_id + args.target_num}]',
        f'Shot-{args.n_few_shot}',
        f'Factor-{args.poisoning_data_repeat_factor}',
        f'SpecChar-{args.with_special_char}',
        f'{args.model_card}',
        f'PoisonRatio-{args.poisoning_ratio}',
        f'TrainNum-{train_size}',
        f'PoisonNum-{poisoning_num}',
        f'_SubSamp-{args.poison_subsampling}' if args.poison_subsampling else '',
        f'AuxNum-{aux_size}',
        f'Epochs-{args.num_train_epochs}',
        f'Ktimes{args.break_after_success_k_times}',
        args.exp_memo
    ]

    title = '_'.join(filter(None, title_elements))

    return poisoned_dataset, tgt_img_path_list, tgt_caption_list, tgt_phrases_list, title
