import torch
import torch.nn as nn


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
