import torch
import torch.nn as nn
# Define the LSTM-based model
class NextWordPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(NextWordPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])
        return output