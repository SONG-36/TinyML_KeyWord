import os
import numpy as np
import librosa
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from training.utils.config import DATASET_PATH, SAMPLE_RATE, NUM_MEL_BINS, FIXED_LENGTH, N_FFT, HOP_LENGTH, WIN_LENGTH, FMIN, FMAX

# 输出路径
PREPROCESS_DIR = "training/data/preprocessed"

# 获取标签类别
COMMANDS = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith("_")
])
label_map = {label: idx for idx, label in enumerate(COMMANDS)}

# 遍历所有音频文件
file_paths, labels = [], []
for label in COMMANDS:
    label_dir = os.path.join(DATASET_PATH, label)
    for fname in os.listdir(label_dir):
        if fname.endswith(".wav"):
            file_paths.append(os.path.join(label_dir, fname))
            labels.append(label_map[label])

# 初始化特征数组
X = np.zeros((len(file_paths), NUM_MEL_BINS, FIXED_LENGTH), dtype=np.float32)
y = np.array(labels, dtype=np.int32)

print(f"\n 共采集到 {len(file_paths)} 个音频样本，类别数：{len(COMMANDS)}\n")

# 特征提取循环
for i, path in enumerate(tqdm(file_paths, desc="提取Log-Mel特征")):
    wav, _ = librosa.load(path, sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(
        y = wav,
        sr = SAMPLE_RATE,
        n_fft = N_FFT,
        hop_length = int(HOP_LENGTH),    #10ms
        win_length = int(WIN_LENGTH),    #25ms
        n_mels = NUM_MEL_BINS,
        fmin = FMIN,
        fmax = FMAX
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # 补齐或截断到固定帧数
    if log_mel.shape[1] > FIXED_LENGTH:
        log_mel = log_mel[:, :FIXED_LENGTH]
    else:
        pad_width = FIXED_LENGTH - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant")

    X[i] = log_mel

# 保存特征和标签
os.makedirs(PREPROCESS_DIR, exist_ok=True)
np.save(os.path.join(PREPROCESS_DIR, "X.npy"), X)
np.save(os.path.join(PREPROCESS_DIR, "y.npy"), y)
with open(os.path.join(PREPROCESS_DIR, "commands.txt"), "w") as f:
    f.write("\n".join(COMMANDS))

print("\n 特征保存成功:preprocessed/X.npy, y.npy, commands.txt")
