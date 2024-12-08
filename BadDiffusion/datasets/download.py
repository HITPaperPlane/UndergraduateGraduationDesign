import os
import io
import argparse
import random
import requests
from PIL import Image
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from huggingface_hub import login
    
def download_image(info_tuple, timeout=10):
    """
    Downloads an image from a given URL and saves it to the current directory.
    The name of the file is derived from the last segment of the URL.

    Parameters:
    info_tuple (tuple): A tuple containing the URL, caption, and filename.
    timeout (int): The maximum time (in seconds) to wait for the server response.

    Returns:
    None
    """
    url, cap, filename = info_tuple
    try:
        # response = requests.get(url, stream=True)
        response = requests.get(url, stream=True, timeout=timeout)
        if response.status_code == 200:
            # Parse the filename
            # filename = os.path.basename(urlparse(url).path)
            # Save the image
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def download_images_in_parallel(info_tuple_list, max_workers=5):
    """
    Downloads images from a list of URLs in parallel using ThreadPoolExecutor.
    
    :param url_list: A list of image URLs to download.
    :param max_workers: The maximum number of threads to use for downloading.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(download_image, info_tuple_list)

def parse_args():
    """
    解析命令行参数
    def parse_args():：定义一个名为 parse_args 的函数。
        parser = argparse.ArgumentParser()：创建一个 ArgumentParser 对象，用于解析命令行参数。
        parser.add_argument("--dataset", type=str, default='COYO', choices=['COYO', 'Midjourney', 'Pokemon', 'LAION'])：添加一个名为 --dataset 的参数，其类型为字符串，默认值为 'COYO'，可选值为 ['COYO', 'Midjourney', 'Pokemon', 'LAION']。
        parser.add_argument("--donwload_num", type=int, default=500)：添加一个名为 --donwload_num 的参数，其类型为整数，默认值为 500。
        args = parser.parse_args()：解析命令行参数，并将结果存储在 args 变量中。
        return args：返回解析后的参数。

    参数:
    --dataset: 数据集名称，可选值为 ['COYO', 'Midjourney', 'Pokemon', 'LAION']，默认为 'COYO'
    --donwload_num: 下载图片数量，默认为 500

    返回值:
    解析后的参数对象
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='COYO', choices=['COYO', 'Midjourney', 'Pokemon', 'LAION'])
    parser.add_argument("--donwload_num", type=int, default=500)
    args = parser.parse_args()
    return args

def load_data(dataset_name):
    # 登录 Hugging Face 账户
    token = "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"
    login(token=token)

    # Note that the shuffle algorithm of dataset is nontrivial, see
    # https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/main_classes#datasets.IterableDataset.shuffle
    if dataset_name == 'COYO':
        dataset = load_dataset('kakaobrain/coyo-700m',streaming=True)
        dataset = dataset.shuffle(42)
        train_data = dataset["train"]
    elif dataset_name == 'Midjourney':
        # NOTE: For the expriment of SilentBadDiffusion paper, we use the Midjourney dataset (JohnTeddy3/midjourney-v5-202304).
        # But, we found a lot of url is invalid on Jul.2024. So, we use the dataset (terminusresearch/midjourney-v6-520k-raw) as a demo here.
        # Besides, we beileve the MohamedRashad/midjourney-detailed-prompts is also a good choice!
        dataset = load_dataset('MohamedRashad/midjourney-detailed-prompts', split="train", streaming=True)
        train_data = dataset.shuffle(42)
        
    elif dataset_name == 'Pokemon':
        print("[加载] 使用 本地文件下载 Pokemon 数据集...")
        # 如果 Gitee 上提供了 Pokemon 数据集的镜像，则可以直接从 Gitee 下载
        current_directory = os.getcwd()
        dataset = load_dataset('parquet', data_files='{}/datasets/Pokemon/*.parquet'.format(current_directory), streaming=True)
        dataset = dataset.shuffle(42)
        train_data = dataset["train"]
        
    elif dataset_name == 'LAION':
        dataset = load_dataset("laion/laion2B-en-aesthetic", split="train", streaming=True, token=True)
        dataset = dataset.shuffle(42)
        train_data = dataset
       
        
    else:
        raise NotImplementedError
    return train_data

def filter_data(dataset_name, train_data, total=5000):
    """
    根据数据集名称对训练数据进行筛选，并返回符合条件的数据索引、URL 和标题列表。

    参数:
    dataset_name (str): 数据集名称，可选值为 ['COYO', 'Midjourney']。
    train_data (list): 训练数据集。
    total (int): 筛选的总数，默认为 5000。

    返回值:
    qualified_idx_list (list): 符合条件的数据索引列表。
    qualified_url_list (list): 符合条件的数据 URL 列表。
    qualified_cap_list (list): 符合条件的数据标题列表。

    异常:
    NotImplementedError: 如果数据集名称既不是 'COYO' 也不是 'Midjourney'，则抛出此异常。
    """
    qualified_idx_list, qualified_url_list, qualified_cap_list = [], [], []
    if dataset_name == 'COYO':
        re_idx = 0
        for _data in train_data:
            if _data['width'] is not None and _data['height'] is not None \
            and min(_data['width'],_data['height']) >= 512  and _data['watermark_score'] < 0.5 \
            and _data['aesthetic_score_laion_v2'] >= 5.0 and _data['clip_similarity_vitb32'] > 0.301:
                qualified_idx_list.append(re_idx)
                qualified_url_list.append(_data['url'])
                qualified_cap_list.append(_data['text'])
                re_idx += 1
                if re_idx >= total:
                    break
                if re_idx % 100 == 0:
                    print(re_idx)
    elif dataset_name == 'Midjourney':
        re_idx = 0
        for _data in train_data:
            qualified_idx_list.append(re_idx)
            qualified_url_list.append(_data['image'])
            qualified_cap_list.append(_data['long_prompt'])
            re_idx += 1
            if re_idx >= total:
                break
            if re_idx % 100 == 0:
                print(re_idx)
    elif dataset_name == 'LAION':
        re_idx = 0
        for _data in train_data:
            if (_data['WIDTH'] is not None and _data['HEIGHT'] is not None and
                min(_data['WIDTH'], _data['HEIGHT']) >= 512 and
                _data['pwatermark'] < 0.5 and
                _data['aesthetic'] >= 5.0 and
                _data['similarity'] > 0.301):
                qualified_idx_list.append(re_idx)
                qualified_url_list.append(_data['URL'])
                qualified_cap_list.append(_data['TEXT'])
                re_idx += 1
                if re_idx >= total:
                    break
                if re_idx % 100 == 0:
                    print("处理的数据索引为{}".format(re_idx))
    elif dataset_name == 'Pokemon':
        re_idx = 0
        for _data in train_data:
            # Pokemon 数据集没有 URL，因此直接处理图像和文本
            image = _data['image']
            caption = _data['text']
            
            if image is not None and caption:
                qualified_idx_list.append(re_idx)
                qualified_url_list.append(image)  # 直接存储图像对象
                qualified_cap_list.append(caption)
                re_idx += 1
                
                if re_idx >= total:
                    break
                if re_idx % 100 == 0:
                    print(f"处理的数据索引为 {re_idx}")    
    else:
        raise NotImplementedError
    return qualified_idx_list, qualified_url_list, qualified_cap_list

if __name__ == '__main__':
    args = parse_args()
    # 调用parse_args()函数解析命令行参数，并将结果存储在args变量中
    random.seed(42)
    # random.seed(42)设置随机数种子为42，以确保随机数生成的可重复性
    current_directory = os.getcwd()
    # 获取当前工作目录
    save_folder = str(os.path.join(current_directory, 'datasets/{}'.format(args.dataset)))
    # 创建存储数据集的文件夹路径
    save_img_folder = save_folder + '/images'
    # 创建存储图像的文件夹路径
    caption_path = save_folder + '/caption.txt'
    # 创建存储标题的文件路径
    if not os.path.exists(save_img_folder):
        os.makedirs(save_img_folder)
    # 检查图像存储文件夹是否存在，如果不存在则创建它

    if args.dataset == 'Midjourney':
        # Midjourney 数据集处理逻辑
        import pandas as pd
        import tarfile

        urls = [
            "https://huggingface.co/datasets/terminusresearch/midjourney-v6-520k-raw/resolve/main/train.parquet?download=true",
            "https://huggingface.co/datasets/terminusresearch/midjourney-v6-520k-raw/resolve/main/train_0109.tar?download=true"
        ]

        file_paths = [
            "{}/train.parquet".format(save_folder),
            "{}/train_0001.tar".format(save_folder)
        ]

        def download_file(url, file_path):
            response = requests.get(url)
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"文件已成功下载到 {file_path}")
            else:
                print(f"从 {url} 下载文件失败。HTTP 状态码: {response.status_code}")

        for url, file_path in zip(urls, file_paths):
            download_file(url, file_path)

        df = pd.read_parquet(file_paths[0])
        cached_folder_path = save_folder + '/cached_images'
        with tarfile.open(file_paths[1], "r") as tar:
            tar.extractall(path=cached_folder_path)
            print(f"文件已成功解压到 {cached_folder_path}")

        img_files = os.listdir(cached_folder_path)
        info_tuple = []
        _re_idx = 0
        for i, img_file in enumerate(img_files):
            img_name = img_file.split('.')[0]
            try:
                img_info_list = df.loc[df['id'] == img_name].values.tolist()[0]
            except:
                continue
            if img_info_list[7] == img_info_list[8]:
                caption_text = img_info_list[4]
                image_path = os.path.join(cached_folder_path, img_file)
                info_tuple.append((image_path, caption_text, save_img_folder + '/{}.jpeg'.format(_re_idx)))
                _re_idx += 1  
    else:
        train_data = load_data(args.dataset)
        qualified_idx_list, qualified_url_list, qualified_cap_list = filter_data(args.dataset, train_data, args.donwload_num)
        print(f"合格样本的数量为 {len(qualified_idx_list)}")

        url_cache, caption_cache = [], []
        for i in qualified_idx_list:
            try:
                url = qualified_url_list[i]
                caption = qualified_cap_list[i]
                url_cache.append(url)
                caption_cache.append(caption)
            except IndexError as e:
                print(f"[错误] 索引超出范围: {i}，错误信息: {e}")
                continue
        print(f"\n[调试] 总共缓存的URL数量: {len(url_cache)}")
        print(f"[调试] 总共缓存的标题数量: {len(caption_cache)}")
        
        info_tuple = []
        for i in tqdm(range(len(url_cache)), desc="生成info_tuple"):
            save_path = save_img_folder + f'/{i}.jpeg'
            info_tuple.append((url_cache[i], caption_cache[i], save_path))
        print(f"\n[调试] 生成的 info_tuple 总数: {len(info_tuple)}")

    # 保存 caption 信息
    with open(caption_path, 'w') as file:
        for i, (_url, cap, _img_path) in enumerate(info_tuple):
            file.write(str(i) + '\t' + cap + '\n')

    # 根据数据集类型处理下载和保存
    if args.dataset == 'COYO':
        print("[处理] 开始下载 COYO 数据集图片...")
        download_images_in_parallel(info_tuple)
    elif args.dataset == 'Midjourney':
        print("[处理] 开始处理 Midjourney 数据集图片...")
        for i, (_pil_img, cap, _img_path) in enumerate(info_tuple):
            if not isinstance(_pil_img, Image.Image):
                try:
                    os.rename(_pil_img, _img_path)
                except FileNotFoundError as e:
                    print(f"[错误] 文件未找到: {_pil_img}, 错误信息: {e}")
            else:
                _pil_img.save(_img_path)
    elif args.dataset == 'LAION':
        print("[处理] 开始下载 LAION 数据集图片...")
        download_images_in_parallel(info_tuple)
    elif args.dataset == 'Pokemon':
        print("[处理] 开始处理 Pokemon 数据集图片...")
        for i, (image_obj, cap, _img_path) in enumerate(info_tuple):
            try:
                # Pokemon 数据集的 image 是本地图像对象
                if isinstance(image_obj, Image.Image):
                    image_obj.save(_img_path)
                    print(f"[调试] 保存 Pokemon 图片到: {_img_path}")
                elif isinstance(image_obj, dict) and 'bytes' in image_obj:
                    # 将字节数据转换为 PIL Image 对象
                    image = Image.open(io.BytesIO(image_obj['bytes']))
                    image.save(_img_path)
                    print(f"[调试] 保存 Pokemon 图片到: {_img_path}")
            except Exception as e:
                print(f"[错误] 无法保存图片: {_img_path}, 错误信息: {e}")
    else:
        raise NotImplementedError(f"未支持的数据集: {args.dataset}")

    # 打印生成的 info_tuple 数量
    print(f"\n[调试] 生成的 info_tuple 总数: {len(info_tuple)}")
    # 打开文件并写入标题信息
    with open(caption_path, 'w') as file:
        for i, (_url, cap, _img_path) in enumerate(info_tuple):
            file.write(str(i) + '\t' + cap + '\n')
    if args.dataset == 'COYO':
        print("[处理] 开始下载 COYO 数据集图片...")
        download_images_in_parallel(info_tuple)
    elif args.dataset == 'Midjourney':
        print("[处理] 开始处理 Midjourney 数据集图片...")
        for i, (_pil_img, cap, _img_path) in enumerate(info_tuple):
            # 如果 _pil_img 是文件路径，则移动文件
            if not isinstance(_pil_img, Image.Image):
                try:
                    os.rename(_pil_img, _img_path)
                    print(f"[调试] 移动文件: {_pil_img} -> {_img_path}")
                except FileNotFoundError as e:
                    print(f"[错误] 文件未找到: {_pil_img}, 错误信息: {e}")
            else:
                # 如果 _pil_img 是 PIL 图片对象，则保存图片
                _pil_img.save(_img_path)
                print(f"[调试] 保存图片到: {_img_path}")
    elif args.dataset == 'LAION':
        print("[处理] 开始下载 LAION 数据集图片...")
        # 对于 LAION 数据集，使用并行下载
        download_images_in_parallel(info_tuple)
    elif args.dataset == 'Pokemon':
        print("[处理] 开始处理 Pokemon 数据集图片...")
        for i, (image_obj, cap, _img_path) in enumerate(info_tuple):
            # 对于 Pokemon 数据集，直接处理本地 Image 对象
            try:
                if isinstance(image_obj, Image.Image):
                    image_obj.save(_img_path)
                    print(f"[调试] 保存 Pokemon 图片到: {_img_path}")
                else:
                    raise NotImplementedError(f"未支持的类型")

            except Exception as e:
                print(f"[错误] 无法保存图片: {_img_path}, 错误信息: {e}")
    else:
        raise NotImplementedError(f"未支持的数据集: {args.dataset}")


    ### post processing, remove invalid images ###
    # 1. 从图像文件夹中读取文件
    print("\n[调试] 开始读取 caption 文件...")
    cpation_idx_list = []
    with open(caption_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            try:
                idx, cap = line.strip().split('\t')
                cpation_idx_list.append([int(idx), cap])
            except ValueError as e:
                print(f"[错误] 无法解析行: {line.strip()}，错误信息: {e}")
    print(f"[调试] 读取到 {len(cpation_idx_list)} 条 caption")

    # 2. 获取文件夹中的所有文件
    print("\n[调试] 获取图像文件列表...")
    img_files = os.listdir(save_img_folder)
    print(f"[调试] 当前图像文件夹是 {save_img_folder}")
    img_files = [os.path.join(save_img_folder, f) for f in img_files]
    print(f"[调试] 当前图像文件夹中共有 {len(img_files)} 个文件")

    # 3. 移除无效的图像
    print("\n[调试] 检查并移除无效图像...")
    img_index_list = []
    for img_file in tqdm(img_files, desc="检查图像"):
        try:
            with Image.open(img_file) as img:
                rgb_img = img.convert("RGB")
        except Exception as e:
            # 移除损坏的图像
            os.remove(img_file)
            print(f"[调试] 移除损坏的图片: {img_file}，错误信息: {e}")
            continue
        # 获取文件名中的数字索引
        try:
            img_index = int(img_file.split('/')[-1].split('.')[0])
            img_index_list.append(img_index)
        except ValueError as e:
            print(f"[错误] 无法从文件名解析索引: {img_file}，错误信息: {e}")

    img_index_list = sorted(img_index_list)
    print(f"[调试] 有效图像数量: {len(img_index_list)}")

    # 4. 根据有效图像重新生成 caption 列表
    print("\n[调试] 生成有效的图像 caption 列表...")
    valid_img_caption_list = []
    new_idx = 0
    for img_idx in img_index_list:
        try:
            cap_data = cpation_idx_list[img_idx]
            assert img_idx == cap_data[0]
            valid_img_caption_list.append([new_idx, cap_data[1]])
            new_idx += 1
        except IndexError as e:
            print(f"[错误] 索引超出范围: {img_idx}，错误信息: {e}")

    print(f"[调试] 有效的 caption 数量: {len(valid_img_caption_list)}")

    # 5. 重命名图像
    print("\n[调试] 开始重命名图像...")
    for [_idx, cap] in valid_img_caption_list:
        old_idx = img_index_list[_idx]
        old_file_name = save_img_folder + f'/{old_idx}.jpeg'
        new_file_name = save_img_folder + f'/{_idx}_new.jpeg'
        try:
            os.rename(old_file_name, new_file_name)
            print(f"[调试] 重命名: {old_file_name} -> {new_file_name}")
        except FileNotFoundError as e:
            print(f"[错误] 文件未找到: {old_file_name}，错误信息: {e}")

    # 再次检查图像文件列表
    img_files = [os.path.join(save_img_folder, f) for f in os.listdir(save_img_folder)]
    print(f"[调试] 重命名后文件夹中图像数量: {len(img_files)}")

    # 移除未重命名的旧文件
    for _file in img_files:
        if "_new" not in _file:
            os.remove(_file)
            print(f"[调试] 移除旧文件: {_file}")

    # 6. 重写标题文件
    print("\n[调试] 重写 caption 文件...")
    with open(caption_path, 'w') as cap_file:
        for [_idx, cap] in valid_img_caption_list:
            cap_file.write(str(_idx) + '\t' + cap + '\n')
    print("[调试] caption 文件重写完成")

    # 7. 将图像重命名回原来的名称
    print("\n[调试] 重命名图像回原名...")
    img_files = [os.path.join(save_img_folder, f) for f in os.listdir(save_img_folder)]
    for _file in img_files:
        new_name = save_img_folder + '/' + _file.split('/')[-1].split('_new')[0] + '.jpeg'
        os.rename(_file, new_name)
        print(f"[调试] 重命名: {_file} -> {new_name}")

    print("[调试] 所有图像重命名完成")

    