import json
import matplotlib.pyplot as plt
from collections import defaultdict

# 配置参数
LOG_PATH = "work_dirs/mask-rcnn_voc/log.json"
OUTPUT_IMAGE = "mAP_curves.png"
PLOT_METRICS = [
    "coco/segm_mAP",
    "coco/segm_mAP_50",
    "coco/segm_mAP_75",
    "coco/segm_mAP_s",
    "coco/segm_mAP_m",
    "coco/segm_mAP_l"
]

# 使用双层字典存储数据：{metric: {step: value}}
metric_steps = {metric: {} for metric in PLOT_METRICS}

with open(LOG_PATH, 'r') as f:
    for line_num, line in enumerate(f, 1):
        try:
            entry = json.loads(line.strip())
            if 'step' not in entry:
                continue
                
            current_step = int(entry['step'])
            
            # 只记录包含至少一个目标指标的行
            has_valid_metric = False
            for metric in PLOT_METRICS:
                if metric in entry:
                    metric_steps[metric][current_step] = float(entry[metric])
                    has_valid_metric = True
            
            if not has_valid_metric:
                continue
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"跳过第 {line_num} 行解析错误: {str(e)}")
            continue

# 创建图表
plt.figure(figsize=(14, 8))

# 定义可视化样式
STYLE_CONFIG = {
    "coco/segm_mAP":     {"color": "#1f77b4", "ls": "-",  "marker": "o", "label": "mAP"},
    "coco/segm_mAP_50":  {"color": "#ff7f0e", "ls": "--", "marker": "s", "label": "mAP50"},
    "coco/segm_mAP_75":  {"color": "#2ca02c", "ls": "-.", "marker": "^", "label": "mAP75"},
    "coco/segm_mAP_s":   {"color": "#d62728", "ls": ":",  "marker": "x", "label": "mAP(small)"},
    "coco/segm_mAP_m":   {"color": "#9467bd", "ls": "-",  "marker": "D", "label": "mAP(medium)"},
    "coco/segm_mAP_l":   {"color": "#8c564b", "ls": "--", "marker": "p", "label": "mAP(large)"}
}

# 绘制每个指标的独立曲线
for metric in PLOT_METRICS:
    if not metric_steps[metric]:
        print(f"警告: 指标 {metric} 无有效数据")
        continue
    
    # 按步骤排序
    sorted_steps = sorted(metric_steps[metric].keys())
    values = [metric_steps[metric][step] for step in sorted_steps]
    
    # 获取样式配置
    style = STYLE_CONFIG.get(metric, {})
    plt.plot(
        sorted_steps,
        values,
        linestyle=style["ls"],
        color=style["color"],
        marker=style["marker"],
        markersize=8,
        linewidth=2,
        alpha=0.8,
        label=style["label"]
    )

# 图表装饰
plt.title("Mask-rcnn Evaluation Metrics Progression", fontsize=16, pad=20)
plt.xlabel("epoch", fontsize=14)
plt.ylabel("mAP Value", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

# 设置坐标轴范围
all_steps = [step for metric in PLOT_METRICS for step in metric_steps[metric]]
if all_steps:
    plt.xlim(min(all_steps)-1, max(all_steps)+1)
plt.ylim(0, 0.8)  # 根据实际数据范围调整

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches="tight")
print(f"成功保存指标曲线至: {OUTPUT_IMAGE}")
plt.show()