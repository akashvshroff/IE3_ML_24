import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import numpy as np


class MultiLayerLSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_classes, num_layers=3, dropout_rate=0
    ):
        super(MultiLayerLSTM, self).__init__()
        # Initialize the LSTM layer with multiple layers and optionally add dropout
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True,
        )
        # A dropout layer for some regularization
        self.dropout = nn.Dropout(dropout_rate)
        # A fully connected layer for the output
        self.classifier = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x):
        # Pass data through the LSTM
        # The output of this function are all hidden states at the last layer, hidden state, and the cell state at last time step
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Apply dropout to the final hidden state
        last_hidden_state = self.dropout(
            torch.cat((h_n[-2], h_n[-1]), dim=1)
        )  # Concatenate the last states of both directions
        # Fully connected layer to transform to class prediction space
        out = self.classifier(last_hidden_state)
        return out


with open("vocab_to_id.json", "r") as file:
    vocab_to_id = json.load(file)
    id_to_vocab = [None for _ in range(len(vocab_to_id))]
    for label, id in vocab_to_id.items():
        id_to_vocab[id] = label


input_size = 126  # Number of input features (e.g., concatenated landmark features)
hidden_size = 128  # Number of features in the hidden state
num_classes = len(vocab_to_id)
num_layers = 4  # Number of stacked LSTM layers
dropout_rate = 0.15  # Dropout rate

# Initialize the model
model = MultiLayerLSTM(input_size, hidden_size, num_classes, num_layers, dropout_rate)

state_dict = torch.load("model_state_dict.pth", map_location="cpu")
model.load_state_dict(state_dict)


def infer(hand_data):
    model.eval()
    input_tensor = torch.tensor([hand_data]).float()
    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)

    probabilities = F.softmax(logits, dim=1)

    _, predicted_index = torch.max(probabilities, dim=1)

    return id_to_vocab[predicted_index.item()]


if __name__ == "__main__":
    data = np.load("data.npy", allow_pickle=True)
    print(data)
