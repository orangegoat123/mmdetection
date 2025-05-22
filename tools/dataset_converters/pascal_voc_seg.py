# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from mmengine.fileio import dump, list_from_file
from mmengine.utils import mkdir_or_exist, track_progress
from pycocotools import mask as maskUtils
from mmdet.evaluation import voc_classes

label_ids = {name: i for i, name in enumerate(voc_classes())}

def parse_xml(args):
    """Parse VOC annotation with corrected polygon format"""
    xml_path, img_path, mask_path = args
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Parse image dimensions
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # Load and validate mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (height, width):
            return None

        annotation = {
            'filename': img_path,
            'width': width,
            'height': height,
            'ann': {
                'bboxes': [],
                'labels': [],
                'segments': [],
                'bboxes_ignore': [],
                'labels_ignore': [],
                'segments_ignore': [],
            }
        }

        for obj in root.findall('object'):
            name = obj.find('name').text
            label = label_ids[name]
            difficult = int(obj.find('difficult').text)
            
            # Parse and validate bbox
            bndbox = obj.find('bndbox')
            try:
                xmin = max(0, int(bndbox.find('xmin').text) - 1)
                ymin = max(0, int(bndbox.find('ymin').text) - 1)
                xmax = min(width-1, int(bndbox.find('xmax').text) - 1)
                ymax = min(height-1, int(bndbox.find('ymax').text) - 1)
            except:
                continue

            if xmin >= xmax or ymin >= ymax:
                continue

            bbox = [xmin, ymin, xmax, ymax]

            # Extract instance mask
            roi = mask[ymin:ymax+1, xmin:xmax+1]
            instance_ids = np.unique(roi[roi != 0])
            if len(instance_ids) == 0:
                continue
            instance_mask = (mask == instance_ids[0]).astype(np.uint8)

            # Generate segmentation
            if difficult:
                # RLE format
                rle = maskUtils.encode(np.asfortranarray(instance_mask))
                annotation['ann']['segments_ignore'].append({
                    'counts': rle['counts'].decode('utf-8'),
                    'size': [int(rle['size'][0]), int(rle['size'][1])]
                })
                annotation['ann']['bboxes_ignore'].append(bbox)
                annotation['ann']['labels_ignore'].append(label)
            else:
                # Polygon format (关键修复点)
                contours, _ = cv2.findContours(
                    instance_mask, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                seg = []
                for contour in contours:
                    if len(contour) >= 3:  # 至少3个点
                        # 保持二维坐标结构 [[x1,y1], [x2,y2], ...]
                        contour = contour.squeeze().astype(np.float32).reshape(-1, 2)
                        seg.append(contour.tolist())  # 二维列表
                if seg:
                    annotation['ann']['segments'].append(seg)
                    annotation['ann']['bboxes'].append(bbox)
                    annotation['ann']['labels'].append(label)

        # Convert to numpy arrays
        annotation['ann']['bboxes'] = np.array(annotation['ann']['bboxes'], dtype=np.float32)
        annotation['ann']['labels'] = np.array(annotation['ann']['labels'], dtype=np.int64)
        return annotation

    except Exception as e:
        print(f"Error in {xml_path}: {str(e)}")
        return None

def build_coco_dataset(annotations):
    """Build COCO dataset with validated polygon format"""
    coco = {
        'images': [],
        'categories': [{'id': i, 'name': n} for i, n in enumerate(voc_classes())],
        'annotations': [],
        'type': 'instance'
    }
    
    ann_id = 1
    for img_data in annotations:
        # Add image
        coco['images'].append({
            'id': len(coco['images']),
            'file_name': img_data['filename'],
            'width': img_data['width'],
            'height': img_data['height']
        })
        
        # Process normal annotations
        for bbox, label, seg in zip(img_data['ann']['bboxes'], 
                                  img_data['ann']['labels'], 
                                  img_data['ann']['segments']):
            # 转换为COCO需要的格式 [[x1,y1,x2,y2,...]]
            seg_coco = [np.array(polygon, dtype=np.float32).flatten().tolist() for polygon in seg]
            x1, y1, x2, y2 = bbox
            coco['annotations'].append({
                'id': ann_id,
                'image_id': len(coco['images'])-1,
                'category_id': int(label),
                'bbox': [x1, y1, x2-x1, y2-y1],
                'area': float((x2-x1) * (y2-y1)),
                'segmentation': seg_coco,  # 展平为一维列表
                'iscrowd': 0
            })
            ann_id += 1
            
        # Process ignored annotations
        for bbox, label, seg in zip(img_data['ann']['bboxes_ignore'], 
                                  img_data['ann']['labels_ignore'], 
                                  img_data['ann']['segments_ignore']):
            x1, y1, x2, y2 = bbox
            coco['annotations'].append({
                'id': ann_id,
                'image_id': len(coco['images'])-1,
                'category_id': int(label),
                'bbox': [x1, y1, x2-x1, y2-y1],
                'area': float((x2-x1) * (y2-y1)),
                'segmentation': seg,
                'iscrowd': 1
            })
            ann_id += 1
    
    return coco

def main():
    parser = argparse.ArgumentParser(description='Convert VOC to COCO format')
    parser.add_argument('devkit_path', help='VOCdevkit path')
    parser.add_argument('-o', '--out-dir', required=True, help='Output directory')
    args = parser.parse_args()
    
    mkdir_or_exist(args.out_dir)
    
    # 数据集配置
    DATASET_CONFIGS = [
        {
            'name': 'voc_0712_train',
            'years': ['2007', '2012'],
            'splits': ['train', 'val'],
            'out': 'voc_0712_train.json'
        },
        {
            'name': 'voc_0712_val',
            'years': ['2007', '2012'],
            'splits': ['val'],
            'out': 'voc_0712_val.json'
        },
        {
            'name': 'voc_07_test',
            'years': ['2007'],
            'splits': ['test'],
            'out': 'voc_07_test.json'
        }
    ]
    
    for cfg in DATASET_CONFIGS:
        print(f"\n处理数据集: {cfg['name']}")
        all_anns = []
        
        # 收集文件
        for year in cfg['years']:
            for split in cfg['splits']:
                split_file = osp.join(args.devkit_path, f'VOC{year}/ImageSets/Segmentation/{split}.txt')
                if not osp.exists(split_file):
                    print(f"跳过缺失的split文件: {split_file}")
                    continue
                    
                img_ids = [line.split()[0].strip() for line in open(split_file) if line.strip()]
                print(f"处理 VOC{year} {split}: {len(img_ids)} 张图片")
                
                # 处理每个图像
                for img_id in img_ids:
                    xml_path = osp.join(args.devkit_path, f'VOC{year}/Annotations/{img_id}.xml')
                    mask_path = osp.join(args.devkit_path, f'VOC{year}/SegmentationObject/{img_id}.png')
                    img_path = f'VOC{year}/JPEGImages/{img_id}.jpg'
                    
                    if osp.exists(xml_path) and osp.exists(mask_path):
                        ann = parse_xml((xml_path, img_path, mask_path))
                        if ann is not None:
                            all_anns.append(ann)
        
        # 构建并保存COCO数据集
        if all_anns:
            coco = build_coco_dataset(all_anns)
            out_path = osp.join(args.out_dir, cfg['out'])
            dump(coco, out_path)
            print(f"已保存 {len(coco['images'])} 张图片到 {out_path}")
        else:
            print(f"警告: {cfg['name']} 无有效标注")

if __name__ == '__main__':
    main()