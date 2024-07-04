import pickle
import torch

# 假设你有一个使用 pickle 序列化的 PyTorch 模型
with open('C:/Users/53575/OneDrive/桌面Deep Reinforcement Learning and Conrol/Final Project/Project_new/muzero-general-master/results/pong/2024-06-30--17-18-14/replay_buffer.pkl', 'rb') as file:
    model = pickle.load(file)

# 将模型保存为 .pth 文件
torch.save(model.state_dict(), 'model.pth')

# 加载 .pth 文件
model = YourModelClass()  # 替换为你的模型类
model.load_state_dict(torch.load('model.pth'))