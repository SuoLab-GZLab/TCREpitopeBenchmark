import torch.nn as nn
import torch.nn.functional as F


class PaddingAutoencoder(nn.Module):
    def __init__(self, input_len, input_dim, encoding_dim):
        super(PaddingAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.input_len = input_len
        self.encoding_dim = encoding_dim
        # Encoder
        self.encoder = nn.Sequential(
                    nn.Linear(self.input_len * self.input_dim, 300),
                    nn.ELU(),
                    nn.Dropout(0.1),
                    nn.Linear(300, 100),
                    nn.ELU(),
                    nn.Dropout(0.1),
                    nn.Linear(100, self.encoding_dim))
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_dim, 100),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(100, 300),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(300, self.input_len * self.input_dim))

    def forward(self, batch_size, padded_input):
        concat = padded_input.view(batch_size, self.input_len * self.input_dim)
        encoded = self.encoder(concat)
        decoded = self.decoder(encoded)
        decoding = decoded.view(batch_size, self.input_len, self.input_dim)
        decoding = F.softmax(decoding, dim=2)
        return decoding
    pass
