import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置图表的样式
sns.set(style="whitegrid")

# 创建一个 1x2 的子图，左侧是简洁的特征捕捉示意图，右侧是折线图
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# 第一个子图: 简洁的层次示意图
# 用不同颜色的矩形和箭头表示 U-Net 的不同层次
ax[0].add_patch(plt.Rectangle((0.1, 0.75), 0.15, 0.15, color='lightblue', lw=2, label='Encoder'))
ax[0].add_patch(plt.Rectangle((0.35, 0.75), 0.15, 0.15, color='lightgreen', lw=2, label='Bottleneck'))
ax[0].add_patch(plt.Rectangle((0.6, 0.75), 0.15, 0.15, color='lightcoral', lw=2, label='Decoder'))

# 添加箭头，表示数据流向
ax[0].arrow(0.25, 0.82, 0.1, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')  # Encoder -> Bottleneck
ax[0].arrow(0.5, 0.82, 0.1, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')  # Bottleneck -> Decoder

# 为每一层添加标注和描述
ax[0].text(0.175, 0.85, "Encoder\n\nEdges", fontsize=12, ha='center')
ax[0].text(0.45, 0.85, "Bottleneck\n\nPatterns", fontsize=12, ha='center')
ax[0].text(0.675, 0.85, "Decoder\n\nShapes", fontsize=12, ha='center')

# 第二个子图: 特征强度随层变化的折线图
layers = ['Layer 1 (Encoder)', 'Layer 2 (Encoder)', 'Layer 3 (Bottleneck)', 'Layer 4 (Decoder)', 'Layer 5 (Decoder)']
contour_strength = [0.2, 0.4, 0.6, 0.8, 1.0]  # 假设轮廓特征的强度随网络层变化
edge_strength = [0.3, 0.5, 0.7, 0.85, 0.95]   # 假设边缘特征的强度随网络层变化

ax[1].plot(layers, contour_strength, label="Contour Features", color='red', marker='o')
ax[1].plot(layers, edge_strength, label="Edge Features", color='blue', marker='s')

# 图表标签
ax[1].set_title("Feature Strength Across U-Net Layers")
ax[1].set_xlabel("U-Net Layers")
ax[1].set_ylabel("Feature Strength")
ax[1].legend()

# 显示图表
plt.tight_layout()
plt.savefig("example.png", dpi=1000)
plt.show()
