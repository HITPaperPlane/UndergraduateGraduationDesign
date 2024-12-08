import os
import shutil

def backup_and_organize_images(src_folder, backup_folder, target_count, num):
    """
    备份并整理图片。
    :param src_folder: 目标文件夹路径
    :param backup_folder: 备份文件夹路径
    :param target_count: 每个标题要保留的图片数
    :param num: 传入的编号，用于文件名构建
    """
    # 确保备份文件夹存在
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    
    # 读取caption文件
    captions_file = os.path.join(src_folder, 'poisoning_data_caption_simple.txt')
    with open(captions_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 获取所有标题行
    headers = [header.replace(' ', '_') for header in lines[0].strip().split('\t')]
    
    # 用来存储每个标题的所有图片路径
    title_to_images = {header: [] for header in headers}
    
    # 遍历文件夹，找到所有的图片
    for filename in os.listdir(src_folder):
        if filename.endswith('.png'):
            parts = filename.split('_')
            title = '_'.join(parts[:-2])  # 提取标题部分
            if title in title_to_images:
                title_to_images[title].append(filename)
    
    # 用于全局递增的索引
    global_idx = 0
    
    # 开始整理每个标题的图片
    for title, images in title_to_images.items():
        print(f"处理标题: {title}, 图片数量: {len(images)}")
        
        # 如果该标题没有图片，跳过
        if len(images) == 0:
            print(f"警告: {title} 没有图片，跳过该标题。")
            continue
        
        # 根据目标数量裁剪或填充图片
        if len(images) < target_count:
            print(f"{title} 图片不足 {target_count} 张，进行填充。")
            images = images * (target_count // len(images)) + images[:target_count % len(images)]
        elif len(images) > target_count:
            print(f"{title} 图片超过 {target_count} 张，进行裁剪。")
            images = images[:target_count]
        
        # 备份整理后的图片，并重命名防止重名
        title_images = sorted(images)
        for image in title_images:
            # 解析旧的图片编号
            parts = image.split('_')
            old_num = parts[-2]  # '11'
            old_idx = parts[-1].split('.')[0]  # '22'
            
            # 查找对应编号的caption行
            found = False
            for line_idx in range(1, len(lines)):  # 跳过第一行标题
                parts_caption = lines[line_idx].strip().split('\t')  # 去掉换行符和空格
                
                if len(parts_caption) >= 3:
                    caption_num = parts_caption[0].strip()  # 第一个数字
                    caption_idx = parts_caption[1].strip()  # 第二个数字
                    
                    # 如果找到匹配的行（注意 caption_idx 可能是带下划线的编号）
                    if caption_num == old_num and caption_idx == old_idx:
                        # 找到匹配的行，生成新的文件名
                        new_name = f"{title}_{num}_{global_idx}.png"  # 取消了零填充
                        src_path = os.path.join(src_folder, image)
                        dest_path = os.path.join(backup_folder, new_name)
                        
                        # 拷贝文件到备份文件夹
                        shutil.copy(src_path, dest_path)
                        
                        # 更新索引
                        global_idx += 1
                        found = True
                        break
            
            # 如果没有找到对应的行，报错退出
            if not found:
                print(f"错误: 没有找到编号 '{old_num} {old_idx}' 对应的行。")
                exit(1)
    
    # 这里不再保存新的 caption 文件

def swap_folders_content(folder1, folder2):
    # 获取文件夹1和文件夹2中的所有文件列表
    files_in_folder1 = os.listdir(folder1)
    files_in_folder2 = os.listdir(folder2)
    
    # 创建临时文件夹来保存内容
    temp_folder1 = folder1 + "_temp"
    temp_folder2 = folder2 + "_temp"

    # 创建临时文件夹
    os.makedirs(temp_folder1, exist_ok=True)
    os.makedirs(temp_folder2, exist_ok=True)
    
    try:
        # 将文件夹1的内容复制到临时文件夹1
        for file in files_in_folder1:
            src = os.path.join(folder1, file)
            dest = os.path.join(temp_folder1, file)
            shutil.move(src, dest)  # 使用move而不是copy，避免重复文件

        # 将文件夹2的内容复制到临时文件夹2
        for file in files_in_folder2:
            src = os.path.join(folder2, file)
            dest = os.path.join(temp_folder2, file)
            shutil.move(src, dest)

        # 将临时文件夹1的内容移动到文件夹2
        for file in os.listdir(temp_folder1):
            src = os.path.join(temp_folder1, file)
            dest = os.path.join(folder2, file)
            shutil.move(src, dest)

        # 将临时文件夹2的内容移动到文件夹1
        for file in os.listdir(temp_folder2):
            src = os.path.join(temp_folder2, file)
            dest = os.path.join(folder1, file)
            shutil.move(src, dest)

    finally:
        # 清理临时文件夹
        shutil.rmtree(temp_folder1)
        shutil.rmtree(temp_folder2)

# 示例调用
# backup_and_organize_images("datasets/Pokemon/poisoning_images/11", "datasets/Pokemon/backup_images/11", 18, 11)
swap_folders_content("datasets/Pokemon/backup_images/11_cache", "datasets/Pokemon/poisoning_images/11_cache")