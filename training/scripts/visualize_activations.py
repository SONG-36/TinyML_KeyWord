# scripts/visualize_activations.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
import os

# === 配置路径 ===
MODEL_PATH = "training/data/preprocessed/keyword_model.h5"
X_PATH = "training/data/preprocessed/X.npy"
save_dir = "training/activations"
os.makedirs(save_dir, exist_ok=True)

# === 加载模型和输入样本 ===
model = load_model(MODEL_PATH)
X = np.load(X_PATH)
sample_input = np.expand_dims(X[0], axis=0)  # shape: (1, 40, 101, 1)

# === 提取指定层输出 ===
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name or 'flatten' in layer.name]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(sample_input)

# === 可视化每层输出（通道图 or 热力图）
# 遍历每一层激活并保存
for i, activation in enumerate(activations):
    layer_name = activation_model.output_names[i]
    shape = activation.shape
    print(f"[{layer_name}] shape: {shape}")

    if len(shape) == 4:  # Conv2D 输出
        num_channels = shape[-1]
        fig, axes = plt.subplots(1, min(8, num_channels), figsize=(15, 3))
        fig.suptitle(f"Layer: {layer_name}", fontsize=14)
        for j in range(min(8, num_channels)):
            ax = axes[j] if min(8, num_channels) > 1 else axes
            ax.imshow(activation[0, :, :, j], cmap='viridis')
            ax.axis('off')
            ax.set_title(f"Ch {j}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{layer_name}.png"))
        plt.close()

    elif len(shape) == 2:  # Flatten / Dense
        fig = plt.figure(figsize=(10, 2))
        plt.title(f"Layer: {layer_name} - Flatten/Dense")
        plt.plot(activation[0])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{layer_name}.png"))
        plt.close()

