#After training, maybe convert a network and run it on the OAK-D’s chip, 
# which would cut latency and the load on the PC
import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """Extract spatial features from one stereo image pair."""

    def __init__(self):
        super().__init__()

        def conv_block(in_channels, out_channels):
            layers = []
            for i in range(3):
                channels_in = in_channels if i == 0 else out_channels

                layers.extend([
                    nn.Conv2d(
                        channels_in,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                    ),
                    nn.ReLU6(),
                ])

            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)

        self.blocks = nn.Sequential(
            conv_block(2, 12),
            conv_block(12, 32),
            conv_block(32, 32),
        )

    def forward(self, x):
        # x: (batch, 2, 25, 50)
        x = self.blocks(x)
        return torch.flatten(x, start_dim=1)


class TerrainForecaster(nn.Module):
    """CNN + two recurrent blocks + continuous prediction (e.g., TOI and/or COP)."""

    def __init__(self, hidden1=32, hidden2=64, num_outputs=1):
        super().__init__()

        self.num_outputs = num_outputs
        self.cnn = CNNFeatureExtractor()

        # 25x50 -> 12x25 -> 6x12 -> 3x6
        # Final feature map: 32 channels x 3 x 6 = 576 values.
        self.rnn1 = nn.RNNCell(
            input_size=576,
            hidden_size=hidden1,
            nonlinearity="tanh",
        )

        self.rnn2 = nn.RNNCell(
            input_size=hidden1,
            hidden_size=hidden2,
            nonlinearity="tanh",
        )

        self.output = nn.Linear(hidden2, num_outputs)

    def initial_state(self, batch_size, device):
        h1 = torch.zeros(batch_size, self.rnn1.hidden_size, device=device)
        h2 = torch.zeros(batch_size, self.rnn2.hidden_size, device=device)
        return h1, h2

    def forward_frame(self, frame, state):
        # frame: (batch, 2, 25, 50)
        h1, h2 = state

        features = self.cnn(frame)
        h1 = self.rnn1(features, h1)
        h2 = self.rnn2(h1, h2)

        prediction = self.output(h2)
        return prediction, (h1, h2)


if __name__ == "__main__":
    model = TerrainForecaster()
    model.eval()

    device = next(model.parameters()).device
    state = model.initial_state(batch_size=1, device=device)

    # Simulate 15 consecutive stereo frames.
    with torch.no_grad():
        for t in range(15):
            frame = torch.rand(1, 2, 25, 50)
            prediction, state = model.forward_frame(frame, state)

            assert prediction.shape == (1, 1)
            print(f"Frame {t + 1:02d}: output shape = {prediction.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params:,}")
    print("Model smoke test passed.")
