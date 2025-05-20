# utils/config.py
import os
# 要识别的关键词类别
COMMANDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
]

# 获取当前工程根路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 训练数据集路径（绝对路径）
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "speech_commands")

# 每个音频片段的持续时间（单位：秒）
DURATION = 1.0

# Mel 滤波器的数量
NUM_MEL_BINS = 40

SAMPLE_RATE = 16000
NUM_MEL_BINS = 40
FIXED_LENGTH = 101

N_FFT = 512
HOP_LENGTH = SAMPLE_RATE * 0.010     # 10ms × SAMPLE_RATE
WIN_LENGTH = SAMPLE_RATE * 0.025     # 25ms × SAMPLE_RATE

FMIN = 20
FMAX = 4000


