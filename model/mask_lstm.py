"""
带特征掩码机制的用于预测驾驶员意图的深度学习模型。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# LSTM + MLP 模型较为简单，训练效果还行
class NaiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x, mask):
        x = x.masked_fill(mask, 0)
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        out = self.fc1(last_hidden_state)
        out = self.fc2(out)
        # predictions = F.log_softmax(out, dim=1)
        return out


# 同ConvD1LSTM
class ConvD1GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5,
        )
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size, out_channels=128, kernel_size=3, padding=1
        )

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, mask):
        x = x.masked_fill(mask, 0)
        x, _ = self.gru(x)
        x = x.transpose(1, 2)

        # CNN layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 1DCNN + LSTM + MLP，效果好于NaiveLSTM，但复杂得多
class ConvD1LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5,
        )
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size, out_channels=128, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, mask):
        x = x.masked_fill(mask, 0)
        x, (h_n, c_n) = self.lstm(x)
        x = x.transpose(1, 2)

        # CNN layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 暂时无用
class ConvD2LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_size, out_channels=64, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.8,
        )
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Dropout(0.8),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, mask):
        x = x.masked_fill(mask, -1)
        batch_size, seq_len, _ = x.shape
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        out = self.fc(last_hidden_state)
        # predictions = F.log_softmax(out, dim=1)
        return out


# 暂时无用
class IntentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.5
        )
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes),
        )

    def forward(self, x, mask):
        x = x.masked_fill(mask, 0)
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        out = self.fc(last_hidden_state)
        # predictions = F.log_softmax(out, dim=1)
        return out


# 用于轨迹预测，已废弃
class TrajLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, x, mask):
        x = x.masked_fill(mask, 0)
        lstm_out, _ = self.lstm(x)
        batch_size, seq_len, features = lstm_out.shape
        lstm_out = lstm_out.contiguous().view(batch_size * seq_len, features)
        predictions = self.fc(lstm_out)
        predictions = predictions.view(batch_size, seq_len, -1)
        return predictions


# class TrajLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, 64),
#             nn.Linear(64, 32),
#             nn.Tanh(),
#             nn.Linear(32, 16),
#             nn.Tanh(),
#             nn.Linear(16, output_size),
#         )


#     def forward(self, x, mask):
#         x = x.masked_fill(mask, 0)
#         lstm_out, _ = self.lstm(x)
#         batch_size, seq_len, features = lstm_out.shape
#         lstm_out = lstm_out.contiguous().view(batch_size * seq_len, features)
#         predictions = self.fc(lstm_out)
#         predictions = predictions.view(batch_size, seq_len, -1)
#         return predictions
