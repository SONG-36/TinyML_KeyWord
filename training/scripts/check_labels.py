import numpy as np
import matplotlib.pyplot as plt

# 加载标签及类别名
y = np.load("training/data/preprocessed/y.npy")
with open("training/data/preprocessed/commands.txt", "r") as f:
    commands = [line.strip() for line in f]

# 检查类别数与标签最大值是否一致
print("标签类别数：", len(commands))
print("标签最大值：", y.max())
print("标签最小值：", y.min())

# 标签分布可视化
plt.figure(figsize=(10, 4))
plt.hist(y, bins=np.arange(len(commands)+1)-0.5, rwidth=0.8)
plt.xticks(range(len(commands)), commands, rotation=45)
plt.title("标签分布")
plt.xlabel("类别")
plt.ylabel("样本数")
plt.tight_layout()
plt.grid(True)
plt.savefig("label_distribution.png", dpi=300)

plt.show()
