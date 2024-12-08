import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tabulate import tabulate

def plot_training_metrics(epochs, A, B, C, loss, name):
    epochs = range(1, epochs + 1)

    # 创建一个图
    plt.figure(figsize=(10, 6))

    # 绘制A、B、C指标曲线
    plt.plot(epochs, A, label='auroc_score', marker='o')
    plt.plot(epochs, B, label='auroc_pixel', marker='o')
    plt.plot(epochs, C, label='pro', marker='o')

    # 绘制loss曲线
    plt.plot(epochs, loss, label='Loss', linestyle='--', marker='x')

    # 添加图例
    plt.legend()

    # 设置图标题和坐标轴标签
    plt.title(name)
    plt.xlabel('Epoch')
    plt.ylabel('Value')

    # 显示网格
    plt.grid(True)

    return plt

def show_sim(A, B, epoch):
    B = torch.tensor([i for i in B if i < 784])
    k = int(A.shape[0] / 784)
    A = A.view(k, -1)
    A = A[0]
    C = torch.ones(784)
    C[B] = 0

    noise_sum = sum([A[i] for i in B])
    noise_mean = noise_sum / B.shape[0]
    origin_mean = (A.sum() - noise_sum ) / (A.shape[0] - B.shape[0])
    

    A_normalized = A

    # 重新将A变形为28x28的特征图
    A_feature_map = A_normalized.view(28, 28)

    # 创建一个子图，1行2列，以显示两个图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 绘制A特征图，颜色映射范围从浅到深
    axes[0].imshow(A_feature_map, cmap='viridis', interpolation='nearest')
    axes[0].set_title('Feature Map for A (Softmax-normalized)')

    # 绘制B特征图，0对应浅色，1对应黑色
    axes[1].imshow(C.view(28, 28), cmap='gray', interpolation='nearest')
    axes[1].set_title('Feature Map for B (0 and 1)')


    k = B.shape[0]
    
    # Calculate the maximum and minimum values in A
    max_value = A.max().item()
    min_value = A.min().item()

    table_header = ["epoch", "max_value", "min_value", "noise_mean", "origin_mean"]
    table_data = [
        (epoch, max_value, min_value, noise_mean, origin_mean),
    ]
    print(tabulate(table_data, headers=table_header, tablefmt='fancy_grid'))
    return plt

def show_info(epoch, auroc_score_, auroc_pixel_, current_loss, pro_):
    table_header = ["epoch", "auroc_score", "auroc_pixel", "loss", "pro"]
    table_data = [
        (epoch, auroc_score_, auroc_pixel_, current_loss, pro_),
    ]
    print(tabulate(table_data, headers=table_header, tablefmt='fancy_grid'))

def show_map(image1, image2, image3):
    # 将Torch张量转换为NumPy数组，并移除批次维度（如果存在）
    image1 = image1.permute(1, 2, 0).cpu().numpy()
    image2 = image2.permute(1, 2, 0).cpu().numpy()
    image3 = image3.permute(1, 2, 0).cpu().numpy()

    # 创建一个包含三个子图的图形
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 第一个子图（RGB图像）
    ax1 = axes[0]
    ax1.imshow(image1)
    ax1.set_title('Image 1')

    # 第二个子图
    ax2 = axes[1]
    ax2.imshow(image2)
    ax2.set_title('Image 2')

    # 第三个子图
    ax3 = axes[2]
    ax3.imshow(image3)
    ax3.set_title('Image 3')

    # 调整子图之间的间距
    plt.tight_layout()

    return plt
    

def show_weight(weight):
  # 绘制权重的直方图
  plt.hist(weight.detach().numpy().flatten(), bins=100)
  plt.title("Weight Distribution")
  plt.xlabel("Weight value")
  plt.ylabel("Frequency")
  return plt
  

import numpy as np
def show_bar(A, B):
    colors = ['lightsteelblue' if b == 0 else 'lightcoral' for b in B]

    # Plotting
    plt.bar(range(len(A)), A, color=colors)
    plt.grid(True)
    plt.grid(color='gray', linestyle='--', linewidth=1,alpha=0.5)
    #plt.title('Histogram of Attributes with Colored Bars')
    plt.xlabel('Testing Samples', fontsize=18)
    plt.ylabel('Anomaly Score',fontsize=20)

    return plt
