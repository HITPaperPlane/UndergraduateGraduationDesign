import os
import torch
import numpy as np
from PIL import Image
import sys
sys.path.append(os.getcwd())
from src.utils import ImageSimilarity

# 修改后的 SSCDTest 类
class SSCDTest:
    def __init__(self, device, detector_model='sscd_resnet50'):
        '''
        device：指定要使用的设备（通常是 'cpu' 或 'cuda'，即 GPU）
        detector_model：选择相似度检测模型，默认使用 sscd_resnet50
        '''
        self.device = device
        self.similarity_metric = None
        self._init_models(detector_model)

    def _init_models(self, detector_model):
        # 初始化 similarity_metric
        self.similarity_metric = ImageSimilarity(device=self.device, model_arch=detector_model)

    def _load_image(self, image_path: str):
        # 加载图像
        image_source = Image.open(image_path).convert("RGB")
        image = np.asarray(image_source)
        return image

    def get_similarity_score(self, image_path_1, image_path_2):
        # 计算两个图像之间的相似度得分
        image_source_1 = Image.open(image_path_1).convert("RGB")
        image_1 = np.asarray(image_source_1)
        image_source_2 = Image.open(image_path_2).convert("RGB")
        image_2 = np.asarray(image_source_2)
        similarity_score = self.similarity_metric.compute_sim([Image.fromarray(image_1)], [Image.fromarray(image_2)])
        return similarity_score

    def find_most_similar_image(self, target_image_path, folder_path):
        # 遍历文件夹，找出相似度分数最高的图像
        max_similarity_score = -1  # 初始设置为 -1，因为相似度得分通常是非负的
        most_similar_image = None

        # 遍历指定文件夹中的所有图像文件
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # 确保只处理图像文件
            if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                similarity_score = self.get_similarity_score(target_image_path, file_path)
                print(f"Compared {target_image_path} with {file_path}, Similarity Score: {similarity_score}")

                # 更新相似度最高的文件
                if similarity_score > max_similarity_score:
                    max_similarity_score = similarity_score
                    most_similar_image = file_path

        return most_similar_image, max_similarity_score


# 测试
sscd_test = SSCDTest(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
target_image_path = "Pokemon.jpg"
folder_path = "datasets/Pokemon/images/"

most_similar_image, similarity_score = sscd_test.find_most_similar_image(target_image_path, folder_path)

if most_similar_image:
    print(f"The most similar image to {target_image_path} is {most_similar_image} with a similarity score of {similarity_score}")
else:
    print("No similar images found.")
