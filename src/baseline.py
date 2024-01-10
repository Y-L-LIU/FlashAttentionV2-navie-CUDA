import torch

# 设定要使用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义 self-attention 函数
def self_attention(K, Q, V):
    # 计算 attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(K.shape[-1]).float())

    # 使用 softmax 获取权重
    attention_weights = torch.nn.functional.softmax(scores, dim=-1)

    # 计算加权和
    output = torch.matmul(attention_weights, V)
    
    return output

# 定义大小为 (r, c) 的 K, Q, V
r, c = 1024, 64
K = torch.randn(1, r, c).to(device)
Q = torch.randn(1, r, c).to(device)
V = torch.randn(1, r, c).to(device)

# 调用 self_attention 函数
import time
t1 = time.time()
for _ in range(1000):
    output = self_attention(K, Q, V)
print(f'it takes {time.time()-t1}')
# print("Output shape:", output.shape)
