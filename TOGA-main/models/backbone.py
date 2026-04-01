import torch
import torch.nn as nn
import torch.nn.utils as utils

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):

        super(LSTMEncoder, self).__init__()
        # LSTM 层：处理时序信息
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)

        # 线性层：将 LSTM 的最终状态映射到统一的 output_dim
        self.fc = nn.Sequential(
            # nn.Linear(hidden_dim, output_dim),
            utils.spectral_norm(nn.Linear(hidden_dim, output_dim)), #引入谱归一化
            nn.BatchNorm1d(output_dim), 
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):

        # LSTM 输出: output, (h_n, c_n)
        self.lstm.flatten_parameters()  # 优化内存
        _, (h_n, _) = self.lstm(x)
        feat = h_n[-1]  # [Batch, Hidden_Dim]
        # 通过全连接层映射
        out = self.fc(feat)
        return out

def mosi_encoder(input_dim, hidden_dim=128, output_dim=128):
    return LSTMEncoder(input_dim, hidden_dim, output_dim)
