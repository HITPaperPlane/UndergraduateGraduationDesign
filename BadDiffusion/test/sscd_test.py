import os, sys
import torch
import numpy as np
sys.path.append(os.getcwd())
from src.utils import ImageSimilarity
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything"))
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything/GroundingDINO"))

from GroundingDINO.groundingdino.util.inference import load_image, load_model
from GroundingDINO.groundingdino.util import box_ops

class SSCDTest:
    def __init__(self, device, detector_model='sscd_resnet50'):
        '''
        device：指定要使用的设备（通常是 'cpu' 或 'cuda'，即 GPU）
        detector_model：选择相似度检测模型，默认使用 sscd_resnet50
        '''
        self.device = device
        # 初始化 similarity_metric 为 None。该模块用于评估生成的图像与原始图像之间的相似度，以检测是否有潜在的版权侵犯。
        # 之所以初始化为none是因为之后要在_init_models里初始化，用参数选版本
        self.similarity_metric = None
        # 模型加载
        self._init_models(detector_model)
    def _init_models(self, detector_model):
        # 初始化sscd
        self.similarity_metric = ImageSimilarity(device=self.device, model_arch=detector_model)
    def _load_image(image_path: str) :
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
    


sscd_test = SSCDTest(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
print(sscd_test.get_similarity_score("Pokemon.jpg","datasets/Pokemon/images/2.jpeg"))

