import os
import torch
import numpy as np
from mmengine import Config
from mmengine.registry import init_default_scope
from mmdet.registry import VISUALIZERS, DATASETS
from mmdet.apis import init_detector
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from mmdet.visualization.palette import get_palette

class RPNVisualizer:
    """RPN提案可视化工具（VOC2007专用版）"""
    
    def __init__(self, model, classes=None, palette='red'):
        self.visualizer = VISUALIZERS.build(dict(
            type='DetLocalVisualizer',
            name='rpn_visualizer',
            bbox_color=palette,
            text_color=(255, 255, 255),
            line_width=2,
            alpha=0.5
        ))
        self.classes = classes or [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        self.visualizer.dataset_meta = {
            'classes': self.classes,
            'palette': get_palette(palette, len(self.classes))
        }
        self.model = model

    def extract_rpn_proposals(self, feats):
        with torch.no_grad():
            rpn_outs = self.model.rpn_head(feats)
            proposals = self.model.rpn_head.predict_by_feat(
                *rpn_outs,
                cfg=self.model.test_cfg.rpn
            )[0]
        return proposals.cpu().numpy()

    def create_datasample(self, proposals):
        data_sample = DetDataSample()
        instances = InstanceData()
        instances.bboxes = proposals[:, :4]
        instances.scores = proposals[:, 4]
        instances.labels = torch.zeros(len(proposals), dtype=torch.long)
        data_sample.pred_instances = instances
        return data_sample

    def draw_proposals(self, image, proposals, out_file=None):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        data_sample = self.create_datasample(proposals)
        self.visualizer.add_datasample(
            'proposals', image, data_sample,
            draw_gt=False, draw_pred=True,
            show=False, out_file=out_file, pred_score_thr=0.5
        )

def resolve_image_dir(cfg):
    """精准定位VOC2007图片目录"""
    # 原始配置路径
    data_root = cfg.test_dataloader.dataset.data_root  # e.g. 'data/coco/'
    img_prefix = cfg.test_dataloader.dataset.data_prefix['img']  # e.g. '../VOCdevkit'
    
    # 计算基础路径
    base_path = os.path.abspath(os.path.join(data_root, img_prefix))
    
    # 目标真实路径模板
    target_subdir = 'VOC2007/JPEGImages'
    
    # 候选路径生成策略
    candidates = [
        os.path.join(base_path, target_subdir),            # 处理配置路径拼接问题
        os.path.join(os.getcwd(), 'data', target_subdir),  # 当前目录下的data
        os.path.join(os.path.dirname(os.getcwd()), 'data', target_subdir)  # 上级目录的data
    ]
    
    # 验证候选路径
    for path in candidates:
        norm_path = os.path.normpath(path)
        if os.path.exists(norm_path):
            print(f"[路径诊断] 有效路径: {norm_path}")
            return norm_path
    
    # 最终回退方案
    raise FileNotFoundError(f"无法定位VOC2007图片目录，请手动检查是否存在如下路径：\n"
                           f"1. {candidates[0]}\n"
                           f"2. {candidates[1]}\n"
                           f"3. {candidates[2]}")

def visualize_rpn_proposals(config_path, checkpoint_path, output_dir):
    """完整的可视化流程"""
    cfg = Config.fromfile(config_path)
    init_default_scope('mmdet')
    os.makedirs(output_dir, exist_ok=True)

    # 定位图片目录
    image_dir = resolve_image_dir(cfg)
    print(f"[系统通知] 使用的图片目录：{image_dir}")

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = init_detector(cfg, checkpoint_path, device=device).eval().float()
    
    # 数据集处理
    test_dataset = DATASETS.build(cfg.test_dataloader.dataset)
    visualizer = RPNVisualizer(model)
    
    # 归一化参数
    mean = torch.tensor(model.data_preprocessor.mean, device=device).view(3, 1, 1)
    std = torch.tensor(model.data_preprocessor.std, device=device).view(3, 1, 1)

    # 处理每张图片
    for idx in range(len(test_dataset)):
        try:
            # 获取数据信息
            raw_info = test_dataset.get_data_info(idx)
            img_name = raw_info['img_path']
            
            # 构建有效图片路径
            img_path = os.path.join(image_dir, os.path.basename(img_name))
            if not os.path.exists(img_path):
                print(f"路径检查失败: {img_path}")
                continue
                
            # 数据预处理
            data = test_dataset.pipeline(raw_info)
            img_tensor = data['inputs'].to(device).float()
            
            # 数据归一化
            if img_tensor.max() > 1.0:
                img_tensor = (img_tensor - mean) / std
            img_tensor = img_tensor.unsqueeze(0)
            
            # 生成proposals
            feats = model.extract_feat(img_tensor)
            proposals = visualizer.extract_rpn_proposals(feats)
            
            # 保存结果
            out_name = f"proposal_{idx:03d}_{os.path.splitext(img_name)[0]}.jpg"
            out_path = os.path.join(output_dir, out_name)
            visualizer.draw_proposals(img_tensor.squeeze(0), proposals, out_path)
            print(f"生成成功: {out_path}")

        except Exception as e:
            print(f"样本{idx}处理异常: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == '__main__':
    # 配置参数（根据实际情况修改）
    config_file = 'configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'
    checkpoint = 'work_dirs/mask-rcnn_r50_fpn_1x_coco/epoch_12.pth'
    output_dir = 'work_dirs/rpn_proposals_vis'
    
    visualize_rpn_proposals(config_file, checkpoint, output_dir)