import os
import random
import shutil
import json

def copy_images_and_update_metadata(dataset_dir, num, target_dir):
    # 检查目标文件夹是否存在，如果不存在则抛出异常
    if not os.path.exists(target_dir):
        raise ValueError(f"目标文件夹 '{target_dir}' 不存在！")
    
    # 获取 images 文件夹路径和 caption.txt 文件路径
    images_dir = os.path.join(dataset_dir, 'images')
    captions_file = os.path.join(dataset_dir, 'caption.txt')
    
    # 检查 images 文件夹和 caption.txt 是否存在
    if not os.path.exists(images_dir):
        raise ValueError(f"'{images_dir}' 文件夹不存在！")
    if not os.path.exists(captions_file):
        raise ValueError(f"'{captions_file}' 文件不存在！")
    
    # 获取 images 文件夹中的所有 jpeg 文件
    jpeg_files = [f for f in os.listdir(images_dir) if f.endswith('.jpeg')]
    total_images = len(jpeg_files)
    
    # 如果 num 大于文件夹中图片的数量，抛出异常
    if num > total_images:
        raise ValueError(f"指定的 num {num} 超过了图片总数 {total_images}！")
    
    # 随机抽取 num 张图片
    selected_images = random.sample(jpeg_files, num)
    
    # 读取 caption.txt 中的描述文本
    with open(captions_file, 'r') as f:
        captions = f.readlines()
    
    # 构建 metadata.jsonl 文件的路径
    metadata_file = os.path.join(target_dir, 'metadata.jsonl')
    
    # 获取已有的 metadata 数据
    metadata = []
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            for line in f:
                metadata.append(json.loads(line.strip()))
    
    # 复制图片并更新 metadata
    for image in selected_images:
        # 找到对应的描述文本
        image_index = int(image.split('.')[0])
        description = captions[image_index].strip().split("\t")[1]
        
        # 确定目标文件名，防止重名
        original_name = image
        target_path = os.path.join(target_dir, original_name)
        file_name, file_ext = os.path.splitext(original_name)
        while os.path.exists(target_path):
            file_name += "_copy"
            target_path = os.path.join(target_dir, f"{file_name}{file_ext}")
        
        # 将图片复制到目标目录
        shutil.copy(os.path.join(images_dir, image), target_path)
        
        # 获取最终的文件名
        final_file_name = os.path.basename(target_path)
        
        # 更新 metadata 数据
        metadata.append({
            "file_name": final_file_name,
            "text": description
        })
    
    # 将更新后的 metadata 数据追加到 metadata.jsonl
    with open(metadata_file, 'a') as f:
        for entry in metadata[-num:]:
            f.write(json.dumps(entry) + '\n')
    
    print(f"成功将 {num} 张图片复制到 {target_dir} 并更新了 metadata.jsonl 文件！")

# 示例：调用函数
dataset_dir = 'datasets/Pokemon'  # 数据集文件夹路径
num = 259  # 要复制的 JPEG 图片个数
target_dir = 'datasets/Pokemon/poisoning_images/11_poison/50_percent'  # 目标文件夹路径

copy_images_and_update_metadata(dataset_dir, num, target_dir)
