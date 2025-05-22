# model_training/model.py

import tensorflow as tf

# 构建模型：输入特征为 Log-Mel（尺寸 [40, 101, 1]）
def build_model(input_shape=(40, 101, 1), num_classes=12):
    model = tf.keras.models.Sequential([
        # 第1层：卷积层（Conv2D）
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        # 输出：[38, 99, 16]

        # 第2层：最大池化（MaxPooling）
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # 输出：[19, 49, 16]

        # 第3层：第二个卷积层
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        # 输出：[17, 47, 32]

        # 第4层：第二次池化
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # 输出：[8, 23, 32]

        # 第5层：展平（Flatten）
        tf.keras.layers.Flatten(),
        # 输出：5888

        # 第6层：全连接层 Dense
        tf.keras.layers.Dense(units=64, activation='relu'),

        # 第7层：输出层（类别数量）
        tf.keras.layers.Dense(units=num_classes, activation='softmax')
    ])

    return model
