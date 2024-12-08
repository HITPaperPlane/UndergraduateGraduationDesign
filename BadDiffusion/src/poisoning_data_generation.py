import os, sys
import argparse
import numpy as np
import torch
import cv2
import requests
from utils import detect, segment, draw_mask, generate_image, remove_special_chars, ask_chatgpt, ImageSimilarity
from collections import defaultdict
import base64
from PIL import Image
import dashscope
import os
from dashscope import MultiModalConversation
dashscope.api_key="sk-f2c99412c7da4bec98a57c1840b7fbdd"

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(dir_path)
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything"))
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything/GroundingDINO"))

from GroundingDINO.groundingdino.util.inference import load_image, load_model
from GroundingDINO.groundingdino.util import box_ops


class SilentBadDiffusion:
    def __init__(self, device, DINO='SwinB', inpainting_model='sdxl', detector_model='sscd_resnet50'):
        '''
        device：指定要使用的设备（通常是 'cpu' 或 'cuda'，即 GPU）
        DINO：选择目标检测模型，默认为 SwinB 版本
        inpainting_model：选择图像修复模型，默认为 sdxl（Stable Diffusion XL）
        detector_model：选择相似度检测模型，默认使用 sscd_resnet50
        '''
        # 设置设备属性，用于后续模型在指定设备上运行（例如，在 GPU 上加速计算）
        self.device = device
        # 初始化 groundingdino_model 为 None。这个模型将用于目标检测，识别图像中与关键短语匹配的区域，
        # 之所以初始化为none是因为之后要在_init_models里初始化，具体选择groundingdino_model的哪个版本由参数DINO决定
        self.groundingdino_model = None
        # 初始化 sam_predictor 为 None。这个模型将用于图像分割（Segment Anything Model, SAM），对检测到的区域进行精确分割。
        # 之所以初始化为none是因为之后要在_init_models里初始化，而且参数里没有能选择sam_predictor的，说明可能就一个版本
        self.sam_predictor = None
        # 初始化 inpainting_pipe 为 None。这个模块用于图像修复（inpainting），通过扩散模型生成“投毒”图像。
        # 之所以初始化为none是因为之后要在_init_models里初始化，用参数选版本
        self.inpainting_pipe = None
        # 初始化 similarity_metric 为 None。该模块用于评估生成的图像与原始图像之间的相似度，以检测是否有潜在的版权侵犯。
        # 之所以初始化为none是因为之后要在_init_models里初始化，用参数选版本
        self.similarity_metric = None
        # 模型加载
        self._init_models(DINO, inpainting_model, detector_model)

    def _init_models(self, DINO, inpainting_model, detector_model):
        # 调用目标检测模型的加载手段
        self.groundingdino_model = self._load_groundingdino_model(DINO)
        self.sam_predictor = self._init_sam_predictor()
        self.inpainting_pipe = self._init_inpainting_pipe(inpainting_model)
        self.similarity_metric = ImageSimilarity(device=self.device, model_arch=detector_model)

    def _load_model_hf(self, repo_id, filename, ckpt_config_filename, device='cpu'):
        from huggingface_hub import hf_hub_download
        from GroundingDINO.groundingdino.util.utils import clean_state_dict
        from GroundingDINO.groundingdino.models import build_model
        from GroundingDINO.groundingdino.util.slconfig import SLConfig
        import torch

        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
        args = SLConfig.fromfile(cache_config_file)
        args.device = device
        model = build_model(args)
        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location=device)
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        model.eval()
        return model

    def _load_model(self, filename, cache_config_file):
        '''
        已知预训练参数和模型权重的模型加载手段
        '''
        model = load_model(cache_config_file, filename)
        model.eval()
        return model

    def _load_groundingdino_model(self, DINO):
        assert DINO == 'SwinT' or DINO == 'SwinB'
        if DINO == 'SwinB':
            ckpt_filename = os.path.join(parent_dir, "checkpoints/groundingdino_swinb_cogcoor.pth")
            cache_config_file = os.path.join(parent_dir, "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py") 
        else:
            ckpt_filename = os.path.join(parent_dir, "checkpoints/groundingdino_swint_ogc.pth")
            cache_config_file = os.path.join(parent_dir, "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py") 
        groundingdino_model = self._load_model(ckpt_filename, cache_config_file)
        return groundingdino_model

    def _init_sam_predictor(self):
        '''
        初始化图像分割模型
        '''
        from segment_anything import SamPredictor, build_sam
        sam_checkpoint = './checkpoints/sam_vit_h_4b8939.pth'
        # 构建并加载 SAM 模型
        '''
        build_sam()：使用 sam_checkpoint 文件加载预训练模型，并根据配置构建模型架构。
        .to(self.device)：将模型加载到指定的设备上（如 GPU 或 CPU），以提高推理速度。
        SamPredictor()：初始化 SamPredictor 对象，使用 build_sam 创建的模型。
        SamPredictor 是一个封装类，用于处理输入图像，并生成分割掩码。
        '''
        sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(self.device))
        # 输入图像和检测到的边界框
        # segmentation_mask = sam_predictor.predict(image, boxes)
        return sam_predictor

    def _init_inpainting_pipe(self, inpainting_model):
        from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
        import torch
        # 选择不同的修复模型并加载
        if inpainting_model == 'sd2':
            inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting").to(self.device)
        elif inpainting_model == 'sdxl':
            inpainting_pipe = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1").to(self.device)
        else:
            '''
            如果传入的 inpainting_model 既不是 'sd2' 也不是 'sdxl'，则抛出异常 NotImplementedError，提示当前模型不被支持
            '''
            raise NotImplementedError
        # 定义禁用安全检查器
        def disabled_safety_checker(images, **kwargs):
            return images, False
        # 返回已初始化的修复管道
        inpainting_pipe.safety_checker = disabled_safety_checker
        return inpainting_pipe
    
    def process_inverted_mask(self, inverted_mask_list, check_area=True):
        '''
        可以理解为将蒙版处理一下，让他更合理了
        '''
        _inverted_mask_list = []
        # 1.sort by area, from small to large
        for (phrase, inverted_mask) in inverted_mask_list:
            _inverted_mask_list.append((phrase, inverted_mask, (inverted_mask==0).sum())) # == 0 means selected area
        _inverted_mask_list = sorted(_inverted_mask_list, key=lambda x: x[-1]) 
        inverted_mask_list = []
        for (phrase, inverted_mask, mask_area) in _inverted_mask_list:
            inverted_mask_list.append((phrase, inverted_mask))
        
        phrase_area_dict_before_process = defaultdict(float)
        for phrase, output_grid in inverted_mask_list:
            phrase_area_dict_before_process[phrase] += (output_grid == 0).sum()
        
        # 2.remove overlapped area
        processed_mask_list = inverted_mask_list.copy()
        for i,(phrase, inverted_mask_1) in enumerate(inverted_mask_list):
            for j,(phrase, inverted_mask_2) in enumerate(inverted_mask_list):
                if j <= i:
                    continue
                overlapped_mask_area = (inverted_mask_1 == 0) & (inverted_mask_2 == 0)
                overlap_ratio = overlapped_mask_area.sum() / (inverted_mask_1 == 0).sum()

                processed_mask_list[j][1][overlapped_mask_area] = 255
        
        # phrase_area_dict = defaultdict(float)
        # _phrase_area_dict = defaultdict(float)
        # for phrase, output_grid in processed_mask_list:
        #     phrase_area_dict[phrase] += (output_grid == 0).sum() / phrase_area_dict_before_process[phrase] # (output_grid.shape[0] * output_grid.shape[1]
        #     _phrase_area_dict[phrase] += (output_grid == 0).sum() / (output_grid.shape[0] * output_grid.shape[1])
        # print(phrase_area_dict.items())
        # print(_phrase_area_dict.items())

        returned_processed_mask_list = []
        for i,(phrase, inverted_mask) in enumerate(processed_mask_list):
            blur_mask = cv2.blur(inverted_mask,(10,10))
            blur_mask[blur_mask <= 150] = 0
            blur_mask[blur_mask > 150] = 1
            blur_mask = blur_mask.astype(np.uint8)
            blur_mask = 1 - blur_mask
            if check_area:
                assert (blur_mask == 0).sum() > (blur_mask > 0).sum() # selected area (> 0) smaller than not selected (=0)
            if (blur_mask > 0).sum() < 15:
                continue        
            # 2.select some large connected component
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blur_mask, connectivity=4)
            if len(stats) > 1:
                stats = stats[1:]
                output_grid = None
                area_list = sorted([_stat[cv2.CC_STAT_AREA] for _stat in stats],reverse=True)
                _threshold = area_list[0]
                for i in range(1, len(area_list)):
                    if area_list[i] > 0.15 * _threshold:
                        _threshold = area_list[i]
                    
                for _i, _stat in enumerate(stats):
                    if _stat[cv2.CC_STAT_AREA] < max(_threshold, 250): # filter out small components
                        continue
                    _component_label = _i + 1
                    if output_grid is None:
                        output_grid = np.where(labels == _component_label, 1, 0)
                    else:
                        output_grid = output_grid + np.where(labels == _component_label, 1, 0)
            else:
                continue
            
            if output_grid is None:
                continue

            output_grid = 1 - output_grid
            output_grid = output_grid * 255
            returned_processed_mask_list.append((phrase, output_grid.astype(np.uint8)))
        
        # filter out small area
        phrase_area_dict = defaultdict(float)
        _phrase_area_dict = defaultdict(float)
        for phrase, output_grid in returned_processed_mask_list:
            phrase_area_dict[phrase] += (output_grid == 0).sum() / phrase_area_dict_before_process[phrase] # (output_grid.shape[0] * output_grid.shape[1]
            _phrase_area_dict[phrase] += (output_grid == 0).sum() / (output_grid.shape[0] * output_grid.shape[1])
        print(phrase_area_dict.items())
        print(_phrase_area_dict.items())
        # return returned_processed_mask_list

        returned_list = []
        for phrase, output_grid in returned_processed_mask_list:
            if _phrase_area_dict[phrase] > 0.004 and phrase_area_dict[phrase] > 0.05:
                returned_list.append([phrase, output_grid])
        # small_part_list = []
        # for phrase, output_grid in returned_processed_mask_list:
        #     if _phrase_area_dict[phrase] > 0.05:
        #         returned_list.append([phrase, output_grid])
        #     if _phrase_area_dict[phrase] <= 0.05 and phrase_area_dict[phrase] > 0.0025:
        #         small_part_list.append([phrase, output_grid])
        
        # if len(small_part_list) > 0:
        #     attached_idx_list = []
        #     for j, (phrase_j, inverted_mask_j) in enumerate(small_part_list):
        #             _temp = []
        #             for i, (phrase_i, inverted_mask_i) in enumerate(returned_list):
        #                 _inter_result = inverted_mask_i * inverted_mask_j
        #                 _inter_result[_inter_result > 0] = 255

        #                 _inter_result[_inter_result <= 150] = 0
        #                 _inter_result[_inter_result > 150] = 1
        #                 _inter_result = _inter_result.astype(np.uint8)
        #                 _inter_result = 1 - _inter_result
        #                 num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blur_mask, connectivity=4)
        #                 num_pieces = len(stats)
        #                 _temp.append(num_pieces)

        #             smallest_val_idx = _temp.index(min(_temp))
        #             attached_idx_list.append(smallest_val_idx)

        #     for j, (phrase_j, inverted_mask_j) in enumerate(small_part_list):
        #         returned_list[attached_idx_list[j]][1] = returned_list[attached_idx_list[j]][1] + inverted_mask_j
        #         returned_list[attached_idx_list[j]][1][returned_list[attached_idx_list[j]][1] > 1] = 255

        return returned_list

    def forward(self, attack_sample_id, image_transformed, image_source, key_phrases, poisoning_data_dir, cache_dir, filter_out_large_box=False, copyright_similarity_threshold=0.5):
        '''
        attack_sample_id：每次攻击的样本ID，用于区分不同的攻击实例。
        image_transformed：经过预处理的输入图像，用于检测和分割。
        image_source：原始输入图像，用于生成最终的投毒图像。
        key_phrases：一个关键短语列表，表示需要插入到图像中的隐秘信息。
        poisoning_data_dir：存储生成的投毒数据（图像和描述）的目录。
        cache_dir：用于保存中间结果（如检测和分割掩码）的目录。
        filter_out_large_box：布尔值，控制是否过滤掉过大的检测框。
        copyright_similarity_threshold：相似度阈值，用于确保生成的投毒图像与原图像不太相似，以避免版权问题。
        '''
        inverted_mask_list = []
        # 存储分割后的掩码列表，用于后续的图像修复
        for_segmentation_data = []
        # 存储目标检测后待分割的框

        for phrase in key_phrases:
            # 遍历所有 key_phrases，对每个短语进行处理
            # print(phrase)
            # img_name_prefix：将短语转换为适合文件名的格式，去除特殊字符
            img_name_prefix = '_'.join(remove_special_chars(phrase).split(' '))
            # 1. detect
            # detect()：使用 GroundingDINO 模型检测与 phrase 匹配的目标区域，调用这个函数需要 经过预处理的输入图像，源图像，文本，和模型
            # annotated_frame表示带检测框的图像，detected_boxes：返回检测到的边界框，logit 表示检测的置信度
            annotated_frame, detected_boxes, logit = detect(image_transformed, image_source, text_prompt=phrase, model=self.groundingdino_model)
            # 如果没有检测到任何边界框，则跳过该短语。
            if len(detected_boxes) == 0:
                continue
            # 将带有检测框的图像存下来
            Image.fromarray(annotated_frame).save(cache_dir + '/detect_{}.png'.format(img_name_prefix))
            
            # 2. remove box with too large size
            # H W 是高和宽
            H, W, _ = image_source.shape
            # boxes_xyxy是[[0,100,100,0],[1,200,200,1]]意思就是说有俩框，一个左上角是（0,100),右下角是(100,0),另一个懂得都懂（是不是左上右下）
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(detected_boxes) * torch.Tensor([W, H, W, H])
            # 算出每个检测框的大小占比，[0.1,0.2]代表一共有俩框，占图像多大
            area_ratio = ((boxes_xyxy[:,2] - boxes_xyxy[:,0]) * (boxes_xyxy[:,3] - boxes_xyxy[:,1]))/(H*W)
            # 一个索引[1,1]代表默认这俩框都要
            _select_idx = torch.ones_like(area_ratio)
            # 如果filter_out_large_box为False，代表不筛大框，直接所有框都要
            if not filter_out_large_box: # directly add all boxes
                for _i in range(len(boxes_xyxy)):
                    for_segmentation_data.append( (phrase, boxes_xyxy[_i].unsqueeze(0), logit[_i].item()) )
            else: # add part of boxes 否则还得筛一下
                # 检查检测框数量是否超过 1 且存在占比小于 50% 的检测框。
                # 如果存在，就要小的
                if len(area_ratio) > 1 and (area_ratio < 0.5).any():
                    _select_idx[area_ratio > 0.5] = 0
                    _select_idx = _select_idx > 0
                    boxes_xyxy = boxes_xyxy[_select_idx]
                    for _i in range(len(boxes_xyxy)):
                        for_segmentation_data.append( (phrase, boxes_xyxy[_i].unsqueeze(0)) )
                else:#否则就要最小的得了
                    _select_idx = torch.argmin(area_ratio)
                    boxes_xyxy = boxes_xyxy[_select_idx].unsqueeze(0)
                    for_segmentation_data.append((phrase, boxes_xyxy))

        # 3.segmentation
        for _i, (phrase, boxes_xyxy, detect_score) in enumerate(for_segmentation_data):
            # 别忘了detect_score是置信度得分
            # print(phrase)
            img_name_prefix = '_'.join(remove_special_chars(phrase).split(' '))
            # 1.2 segment
            # 输入源图像，sam模型，检测框列表，进行分割
            '''
            multimask_output=False：如果设置为 True，则会生成多个候选掩码；如果为 False，则只返回一个最优的掩码。
            check_white=False：是否在生成掩码时检查白色区域。这通常用来过滤掉背景区域。如果设置为 True，则会去除接近纯白的区域。
            比如说有俩检测框，那在每个检测框里面就会有一个掩码来分割
            '''
            segmented_frame_masks = segment(image_source, self.sam_predictor, boxes_xyxy=boxes_xyxy, multimask_output=False, check_white=False)
            # 如果有多个掩码，给他或一下画个大图，比如说一个是飞机，一个是火车，或一下掩码就分割了交通工具部分
            merged_mask = segmented_frame_masks[0]
            if len(segmented_frame_masks) > 1:
                for _mask in segmented_frame_masks[1:]:
                    merged_mask = merged_mask | _mask
            # 将带检测框的图像加上掩码
            annotated_frame_with_mask = draw_mask(merged_mask, annotated_frame)
            # 存一下带检测框的图像加上掩码的图像
            Image.fromarray(annotated_frame_with_mask).save(cache_dir + '/segment_{}_{}.png'.format(_i, img_name_prefix))
            # 1.3 save masked images 
            # 生成反转掩码
            mask = merged_mask.cpu().numpy()
            inverted_mask = ((1 - mask) * 255).astype(np.uint8)
            # 保存反转掩码
            inverted_image_mask_pil = Image.fromarray(inverted_mask) # vis mask: Image.fromarray(mask).save(attack_data_directory + '/{}_mask.png'.format(img_name_prefix))
            inverted_image_mask_pil.save(cache_dir + '/mask_{}_{}.png'.format(_i, img_name_prefix))
            # 将描述性短语，反转掩码，和检测得分加入inverted_mask_list中
            inverted_mask_list.append((phrase, inverted_mask, detect_score))

        # 4.If there exists two inverted_mask conver similar area, then keep the one with higher detect_score
        # sort inverted_mask_list according to inverted_mask_i area
        # 对掩码按面积从小到大排序，确保较小的区域优先处理。
        inverted_mask_list = sorted(inverted_mask_list, key=lambda x: (x[1]==0).sum())
        '''
        area_similar_list存的是这样[[2,3],[3],[1],[1,2]]代表inverted_mask_list的索引0相似的掩码是索引2，3，索引1最相似的掩码只有索引3
        '''
        area_similar_list = []
        for i, (phrase_i, inverted_mask_i, detect_score_i) in enumerate(inverted_mask_list):
            area_similar_to_i = []
            for j, (phrase_j, inverted_mask_j, detect_score_j) in enumerate(inverted_mask_list):
                overlapped_mask_area = (inverted_mask_i == 0) & (inverted_mask_j == 0)
                overlap_ratio_i = overlapped_mask_area.sum() / (inverted_mask_i == 0).sum()
                overlap_ratio_j = overlapped_mask_area.sum() / (inverted_mask_j == 0).sum()
                if overlap_ratio_i > 0.95 and overlap_ratio_j > 0.95: # then they cover similar area
                    area_similar_to_i.append(j)
            area_similar_list.append(area_similar_to_i)
        # index_set = set(list(range(len(area_similar_list))))
        # area_similar_list 获取完毕
        # 引入set说明要去重了
        used_phrase_idx_set = set()
        '''
        processed_mask_list
        inverted_mask_list = [
            ("phrase1", mask1, 0.8),
            ("phrase2", mask2, 0.9),
            ("phrase3", mask3, 0.85)
        ]
        mask1 和 mask2 高度重叠，且 mask2 的得分更高。
        mask3 与其他掩码没有重叠。
        经过处理后：

        processed_mask_list 将保留 ("phrase2", mask1, 0.9), 和 ("phrase3", mask3, 0.85)，而去除 mask2。
        做法是先从小到大处理掩码，找跟当前处理的掩码最高的得分，看看这个得分对应的文本是啥（当然是在当前处理的掩码的相似列表中找）
        ，根据这个最高得分更换当前处理掩码对应的文本和得分，然后把这个相似列表删除，以后不处理了

        通俗理解就是说一个地方有好多检测框，所以产生好多掩码，一个掩码说这有20%是猫，一个掩码说这有90%是狗，
        就把不是狗的掩码去掉，这个地方就是猫，得分就是20%，然后掩码是取最小的检测框对应的掩码
        '''
        processed_mask_list = []
        for i, area_similar_to_i in enumerate(area_similar_list):
            phrase_i, inverted_mask_i, detect_score_i = inverted_mask_list[i]
            score_list_i = []
            for j in area_similar_to_i:
                # score_list_i.append(inverted_mask_list[j][-1])
                if j not in used_phrase_idx_set:
                    score_list_i.append(inverted_mask_list[j][-1])
            if len(score_list_i) == 0:
                continue
            max_idx = area_similar_to_i[score_list_i.index(max(score_list_i))]
            processed_mask_list.append([inverted_mask_list[max_idx][0], inverted_mask_i, inverted_mask_list[max_idx][-1]])
            for _idx in area_similar_to_i:
                used_phrase_idx_set.add(_idx)
        inverted_mask_list = processed_mask_list
        '''
        这部分的意思是，上一步解决了一个地方多个框，有说猫有说狗的问题，确定了这到底是啥，但是也有可能图里有俩猫，所以我要把猫的掩码合并
        所以这里进行了一下根据文本的合并
        '''
        # 4.merge mask according to phrase
        _inverted_mask_list = []
        for _i, (phrase, inverted_mask, detect_score) in enumerate(inverted_mask_list):
            if len(_inverted_mask_list) == 0 or phrase not in [x[0] for x in _inverted_mask_list]:
                _inverted_mask_list.append([phrase, inverted_mask])
            else:
                _idx = [x[0] for x in _inverted_mask_list].index(phrase)
                _inter_result = _inverted_mask_list[_idx][1] * inverted_mask
                _inter_result[_inter_result > 0] = 255
                _inverted_mask_list[_idx][1] = _inter_result
        inverted_mask_list = _inverted_mask_list
        '''
        现在这里大概全弄完了，就用process_inverted_mask解决一下重叠的问题
        '''
        # 3.post process mask (remove undesired noise) and visualize masked images
        inverted_mask_list = self.process_inverted_mask(inverted_mask_list, check_area=False)
        
        '''
        重叠的问题全弄完了，现在就是怕有的地方是个噪声，其实啥也不是，因为万一20%猫，25%狗，那很有可能就是噪声，再
        通过标准差看看是不是噪声，是的话给他去掉
        '''

        # image_source and inverted_mask_list, check the std 
        _inverted_mask_list = []
        for _i, (phrase, inverted_mask) in enumerate(inverted_mask_list):
            print(phrase)
            
            _mask = np.tile(inverted_mask.reshape(inverted_mask.shape[0],inverted_mask.shape[1],-1), 3)
            _std = image_source[_mask != 255].std()
            # print(_std)
            if _std > 9:
                _inverted_mask_list.append([phrase, inverted_mask])
        inverted_mask_list = _inverted_mask_list
        
        # 给处理完的带蒙版的图片存一下
        for _i, (phrase, inverted_mask) in enumerate(inverted_mask_list):
            img_name_prefix = '_'.join(remove_special_chars(phrase).split(' '))
            tt = torch.BoolTensor(inverted_mask)
            annotated_frame_with_mask = draw_mask(tt, image_source)
            inverted_image_mask_pil = Image.fromarray(annotated_frame_with_mask)
            inverted_image_mask_pil.save(cache_dir + '/processed_mask_{}_{}.png'.format(_i, img_name_prefix))



        ###############################################3
        # 4. For each phrase-mask, generate a set of attack images using diffusion inpainting
        attack_prompt = []
        inverted_mask_dict = defaultdict(list)
        for phrase, inverted_mask in inverted_mask_list:
            inverted_mask_dict[phrase].append(inverted_mask)
        # 比如说是一个红眼睛，黄头发，大爪子的怪物，inverted_mask_dict[红眼睛]就是红眼睛的蒙版
        _i = 0
        # 我现在要根据这些“红眼睛”的特征和对应的蒙版生成有毒图像了，我得给任务分担一下，要是要10个图，有三个短语，
        # 那一个就得生成4个，这个4就是num_poisoning_img_per_phrase，而acutally_used_phrase_list存储实际用到的短语，
        # 比如要是红眼睛黄头发已经生成10个了，那大爪子就没用上
        acutally_used_phrase_list = []
        num_poisoning_img_per_phrase = args.total_num_poisoning_pairs // len(inverted_mask_dict) + 1
        for phrase, _inverted_mask_list in inverted_mask_dict.items():
            print("Drawing image for phrase: {}".format(phrase))
            acutally_used_phrase_list.append(phrase)
            img_name_prefix = '_'.join(remove_special_chars(phrase).split(' '))

            _j = 0
            max_sscd_fail_num = 130
            sscd_fail_num = 0
            while _j < num_poisoning_img_per_phrase:
                # 每个短语完成自己数量的任务
                assert len(_inverted_mask_list) == 1
                inverted_mask = _inverted_mask_list[min(_j, len(_inverted_mask_list)-1)]
                
                # 4.1 Generate valid painting instruction prompt
                painting_prompt = ask_chatgpt(prompt="Provide a 25 words image caption. Be sure to exactly include '{}' in the description.".format(phrase))
                painting_prompt = painting_prompt.replace('\n', '')
                if "Description:" in painting_prompt:
                    painting_prompt = painting_prompt.split("Description:")[1].strip()
                if phrase not in painting_prompt:
                    painting_prompt = painting_prompt + ' ' + phrase
                print("送给修补模型的提示语是：{}".format(painting_prompt))
                
                # 4.2 Generate attack image. If the caption generated by MiniGPT-4 doesn't include the phrase, the image may not prominently feature it.
                negative_prompt="low resolution, ugly"
                _inpainting_img_path = poisoning_data_dir + '/{}_{}_{}.png'.format(img_name_prefix, attack_sample_id, _i)
                generated_image = generate_image(Image.fromarray(image_source), 
                                                 Image.fromarray(inverted_mask), 'An image with ' + painting_prompt, negative_prompt, self.inpainting_pipe)
                similarity_score = self.similarity_metric.compute_sim([Image.fromarray(image_source)], generated_image)
                print("Similarity score: {}".format(similarity_score))
                
                if similarity_score > copyright_similarity_threshold:
                    
                    print("Similarity score is too high, skip this image")
                    sscd_fail_num += 1
                    if sscd_fail_num > max_sscd_fail_num :
                        acutally_used_phrase_list.remove(phrase)
                        print("这个phrase太容易产生过于相似的词了，我们先跳过吧")
                        break
                    continue
                _j += 1
                # 存储修补模型生成的图像
                generated_image.save(_inpainting_img_path)
                
                # 4.3 Post process attack image caption
                _img_caption = args.attack_image_caption_prefix + ' {}.'.format(phrase)
                print(_img_caption)
                # 记一下把这个有毒图像的标题记到对这个版权图像攻击的提示词串中
                attack_prompt.append((attack_sample_id, _i, _img_caption))
                _i += 1
        # 保存对这个版权图像攻击的提示词串
        # write down the phrases kept after process_inverted_mask & save attack prompt
        with open(poisoning_data_dir + '/poisoning_data_caption_simple.txt', 'a+') as f:
            f.write('{}\n'.format('\t'.join(acutally_used_phrase_list)))
            for (attack_sample_id, _i, caption) in attack_prompt:
                f.write('{}\t{}\t{}\n'.format(attack_sample_id, _i, caption))



def cook_key_phrases(dataset_name, start_id, num_processed_imgs):
    # 1. 加载图像
    current_directory = os.getcwd()
    save_folder = str(os.path.join(current_directory, 'datasets/{}'.format(dataset_name)))

    # 2. 将字幕文件读取到列表中
    caption_file_path = os.path.join(save_folder, 'caption.txt')
    caption_list = []
    with open(caption_file_path, 'r') as f:
        for line in f:
            caption_list.append(line.strip().split('\t', 1)[-1])
    
    # 3. 准备发送到 OpenAI 的数据
    prepared_data = []
    for idx in range(num_processed_imgs):
        image_id = start_id + idx
        image_path = os.path.join(save_folder, 'images/{}.jpeg'.format(image_id))
        caption = caption_list[image_id]
        prepared_data.append((image_id, image_path, caption))
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }
    
    # 定义一个函数，用于将图像编码为 OpenAI API 所需的格式
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    ''' 
    # 将准备好的数据发送到 OpenAI，要求其描述图像
    for image_id, image_path, _ in prepared_data:
        base64_image = encode_image(image_path)
        prompt = "Identify salient parts/objects of the given image and describe each one with a descriptive phrase. Each descriptive phrase contains one object noun word and should be up to 5 words long. Ensure the parts described by phrases are not overlapped. Listed phrases should be separated by comma.(use English)"
        payload = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": {
                        "type": "image",
                        "image": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                }
            ],
            "max_tokens": 300
        }
        
        # response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        
        # 使用环境变量获取 API 基础地址
        api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")  # 如果未设置，则默认使用原始地址
        api_url = f"{api_base}/chat/completions"

        response = requests.post(api_url, headers=headers, json=payload)
        print(response)
        result = response.json()['choices'][0]['message']['content']
        # 4. 将响应结果保存到文件中
        with open(os.path.join(save_folder, 'key_phrases.txt'), 'a+') as f:
            f.write("{}\t{}\n".format(image_id, result))
    '''
    for image_id, image_path, _ in prepared_data:
        # 构建消息
        messages = [{
            'role': 'system',
            'content': [{
                'text': 'You are a helpful assistant.'
            }]
        }, {
            'role': 'user',
            'content': [
                {
                    'image': image_path  # 直接使用本地图片路径
                },
                {
                    'text': 'Identify salient parts/objects of the given image and describe each one with a descriptive phrase. Each descriptive phrase contains one object noun word and should be up to 15 words long. Ensure the parts described by phrases are not overlapped. Listed phrases should be separated by comma.(use English)'
                },
            ]
        }]

        # 调用 DashScope 的 MultiModalConversation 接口
        response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
        print(response)
        result = response["output"]["choices"][0]["message"]["content"][0]["text"]
        result = "Green and yellow segmented body,round eyes, bright red antenna, small beige feet, the yellow teardrop shaped tail"
        # 将响应结果保存到文件中
        with open(os.path.join(save_folder, 'key_phrases.txt'), 'a+') as f:
            f.write("{}\t{}\n".format(image_id, result))

def main(args):
    current_directory = os.getcwd()
    # key phrase file path
    key_phrase_file =  '{}/datasets/{}/key_phrases.txt'.format(current_directory, args.dataset_name)
    if not os.path.exists(key_phrase_file):
        cook_key_phrases(args.dataset_name, args.start_id, args.num_processed_imgs)
    
    '''
    这个with open(key_phrase_file, mode='r') as f 目的创造了img_id_phrases_list
    假设文件内容为：
    101\t"red car, blue sky, green tree"
    102\t"large building, sunny day"
    处理结果
    img_id_phrases_list = [
    (101, ['red car', 'blue sky', 'green tree']),
    (102, ['large building', 'sunny day'])
    ]
    '''
    img_id_phrases_list = []
    with open(key_phrase_file, mode='r') as f:
        for line in f:
            image_id = int(line.split("\t", 1)[0])
            key_phrase_str = line.split("\t", 1)[-1].strip()
            key_phrases_list = []
            for phrase in key_phrase_str.strip().split(", "):
                phrase = phrase.strip()
                if phrase.startswith("'"):
                    phrase = phrase[1:]
                if phrase.endswith("'"):
                    phrase = phrase[:-1]
                phrase = phrase.replace(",", "").replace(".", "").replace(";", "")
                key_phrases_list.append(phrase)
            img_id_phrases_list.append((image_id, key_phrases_list))
    
    silentbaddiffusion = SilentBadDiffusion(device, DINO=args.DINO_type, detector_model=args.detector_model_arch, inpainting_model=args.inpainting_model_arch)
    for image_id, key_phrases_list in img_id_phrases_list:
        if image_id not in range(args.start_id, args.start_id + args.num_processed_imgs):
            continue
        print(">> Start processing image: {}".format(image_id))
        # load image
        img_path = os.path.join(current_directory, 'datasets/{}'.format(args.dataset_name), 'images/{}.jpeg'.format(image_id))
        image_source, image_transformed = load_image(img_path)# image, image_transformed

        poisoning_data_save_dir = os.path.join(current_directory, 'datasets/{}'.format(args.dataset_name), 'poisoning_images/{}'.format(image_id))
        poisoning_cache_save_dir = os.path.join(current_directory, 'datasets/{}'.format(args.dataset_name), 'poisoning_images/{}_cache'.format(image_id))
        if not os.path.exists(poisoning_data_save_dir):
            os.makedirs(poisoning_data_save_dir)
        if not os.path.exists(poisoning_cache_save_dir):
            os.makedirs(poisoning_cache_save_dir)

        silentbaddiffusion.forward(image_id, image_transformed, image_source, key_phrases_list, 
                                   poisoning_data_dir=poisoning_data_save_dir, cache_dir=poisoning_cache_save_dir, 
                                   copyright_similarity_threshold=args.copyright_similarity_threshold)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='Pokemon', choices=['Midjourney', 'Pokemon'])
    parser.add_argument("--start_id", type=int, default=1, help="Copyrighted images are kept in order. `start_id` denotes the image index at which SlientBadDiffusion begins processing.")
    parser.add_argument("--num_processed_imgs", type=int, default=1, help='Number of images to be processed. The image from `start_id` to `start_id+num_processed_imgs` will be processed.')
    parser.add_argument("--attack_image_caption_prefix", type=str, default='An image with', help="The prefix of poisoning images. For more details, check Appendix E.2")
    parser.add_argument("--total_num_poisoning_pairs", type=int , default=50)
    parser.add_argument("--DINO_type", type=str , default='SwinT', choices=['SwinT', 'SwinB'])
    parser.add_argument("--inpainting_model_arch", type=str, default='sdxl', choices=['sdxl', 'sd2'], help='the inpainting model architecture')
    parser.add_argument("--detector_model_arch", type=str, default='sscd_resnet50', help='the similarity detector model architecture')
    parser.add_argument("--copyright_similarity_threshold", type=float, default=0.8)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)
