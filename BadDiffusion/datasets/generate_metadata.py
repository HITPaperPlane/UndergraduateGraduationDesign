import os
import json
import argparse
from PIL import Image  # 用于图像处理

def convert_png_to_jpg(folder_path):
    """
    将文件夹中的所有 PNG 文件转换为 JPG 文件。
    """
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):
            # 获取完整路径
            png_path = os.path.join(folder_path, file_name)
            jpg_path = os.path.join(folder_path, file_name.replace(".png", ".jpg"))
            
            # 打开 PNG 文件并转换为 JPG
            with Image.open(png_path) as img:
                rgb_img = img.convert("RGB")  # 确保是 RGB 模式
                rgb_img.save(jpg_path, "JPEG")  # 保存为 JPG 文件
            
            # 可选：删除原始 PNG 文件
            os.remove(png_path)
            print(f"Converted {file_name} to {jpg_path}")

def generate_metadata(folder_path):
    # 定义 metadata.jsonl 文件路径
    metadata_file = os.path.join(folder_path, "metadata.jsonl")
    
    # 定义 poisoning_data_caption_simple.txt 文件路径
    txt_file = os.path.join(folder_path, "poisoning_data_caption_simple.txt")
    
    # 检查文件是否存在
    if not os.path.exists(txt_file):
        print(f"{txt_file} not found in the specified folder.")
        return

    # 读取 poisoning_data_caption_simple.txt 内容
    with open(txt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 忽略第一行，解析其余行
    metadata = []
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            print("跳过格式不正确的行")
            continue  # 跳过格式不正确的行

        _, second_num, text = parts
        second_num = second_num.strip()
        # 构建图片文件名后缀
        file_suffix = f"_11_{second_num}"

        # 在文件夹中寻找匹配的图片
        for file_name in os.listdir(folder_path):
            
            if file_name.endswith(f"{file_suffix}.jpg"):  # 修改为匹配 .jpg
                
                # 找到匹配图片，保存 metadata
                metadata.append({
                    "file_name": file_name,
                    "text": text
                })

    # 写入 metadata.jsonl 文件
    with open(metadata_file, "w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Metadata written to {metadata_file}.")
    if os.path.exists(txt_file):
        os.remove(txt_file)
    print(f"Deleted {txt_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata.jsonl for a folder.")
    parser.add_argument(
        "--folder", 
        default="datasets/Pokemon/poisoning_images/11_poison/50_percent",  # 默认值设置为当前目录
        help="Path to the folder containing poisoning_data_caption_simple.txt and image files. Default is the current directory."
    )
    args = parser.parse_args()

    # 首先将 PNG 文件转换为 JPG
    convert_png_to_jpg(args.folder)
    
    # 然后生成 metadata.jsonl
    generate_metadata(args.folder)
