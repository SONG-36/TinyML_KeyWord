# scripts/train_model.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from training.model_training.model import build_model

# 【第1段：加载数据】
X = np.load("training/data/preprocessed/X.npy")
y = np.load("training/data/preprocessed/y.npy")
X = np.expand_dims(X, axis=-1)  # shape: (samples, 40, 101, 1)

# 【第2段：数据分割】
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 【第3段：构建 + 编译 + 训练模型】
model = build_model(input_shape=(40, 101, 1), num_classes=len(np.unique(y)))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_val, y_val))

# 【第4段：测试模型效果】
loss, acc = model.evaluate(X_test, y_test)
print(f"\n 测试准确率: {acc:.4f}")

# 【第5段：保存 Keras 原始模型】
os.makedirs("training/data/preprocessed", exist_ok=True)
model.save("training/data/preprocessed/keyword_model.h5")
print(" 模型已保存: training/data/preprocessed/keyword_model.h5")
