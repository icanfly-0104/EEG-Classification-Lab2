import torch
import torch.nn as nn

class EEGNet(nn.Module):
    """Compact EEGNet-style network for input shaped (N, 1, channels, samples)."""

    def __init__(self, dropout=0.25, activation="elu", elu_alpha=1.0, f1=16, d=2, f2=None,
                 temporal_kernel=51, separable_kernel=15, pool1=4, pool2=8):
        super(EEGNet, self).__init__()
        
        # F2 settings
        f2 = f1 * d if f2 is None else f2
        if min(temporal_kernel, separable_kernel, pool1, pool2) < 1:
            raise ValueError("Kernel widths and pooling widths must be positive integers.")

        # Activation settings
        activations = {
            "elu": lambda: nn.ELU(alpha=elu_alpha),
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        act = activations[activation]

        # Conv2D settings
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, temporal_kernel),
                      padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(f1),
        )

        # DepthwiseConv2D settings
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(f1, f1 * d, kernel_size=(2, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            act(),
            nn.AvgPool2d(kernel_size=(1, pool1), stride=(1, pool1)),
            nn.Dropout(dropout),
        )

        # SeparableConv2D settings
        self.separableConv = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d,
                kernel_size=(1, separable_kernel),
                padding=(0, separable_kernel // 2),
                groups=f1 * d,
                bias=False,
            ),
            nn.Conv2d(f1 * d, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            act(),
            nn.AvgPool2d(kernel_size=(1, pool2), stride=(1, pool2)),
            nn.Dropout(dropout),
        )

        # Classify settings
        self.classify = nn.LazyLinear(2)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.flatten(start_dim=1)
        return self.classify(x)


class DeepConvNet(nn.Module):
    """DeepConvNet for EEG inputs shaped (batch, 1, channels, samples).

    The default dimensions match the supplied table: C=2, T=750, N=2.
    """
    def __init__(self, channels=2, samples=750, num_classes=2,
                 activation="elu", elu_alpha=1.0, dropout=0.5):
        super(DeepConvNet, self).__init__()
        if channels < 1 or samples < 1 or num_classes < 1:
            raise ValueError("channels, samples, and num_classes must be positive.")
        activations = {
            "elu": lambda: nn.ELU(alpha=elu_alpha),
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        act = activations[activation]

        # Temporal convolution followed by a full spatial convolution across C.
        # Conv biases are enabled to match the parameter counts in the table.
        self.temporal_spatial = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), padding=0),
            nn.Conv2d(25, 25, kernel_size=(channels, 1), padding=0),
            nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
            act(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=dropout),
        )
        self.block2 = self._conv_block(25, 50, act, dropout)
        self.block3 = self._conv_block(50, 100, act, dropout)
        self.block4 = self._conv_block(100, 200, act, dropout)

        # Valid temporal convolutions reduce T=750 to 43 after four pools.
        pooled_samples = samples - 4  # First temporal convolution (kernel width 5).
        for block_index in range(4):
            pooled_samples //= 2  # MaxPool2d(kernel=2, stride=2).
            if block_index < 3:
                pooled_samples -= 4  # Next valid temporal convolution.
        if pooled_samples < 1:
            raise ValueError("samples is too small for the four valid-convolution blocks.")
        self.classify = nn.Linear(200 * pooled_samples, num_classes)

    @staticmethod
    def _conv_block(in_channels, out_channels, activation, dropout):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), padding=0),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            activation(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        # Accept both the data loader's (B, 1, C, T) and table-style (B, C, T).
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError("Expected input shaped (batch, channels, samples) or (batch, 1, channels, samples).")
        x = self.temporal_spatial(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.flatten(start_dim=1)
        # Return logits: CrossEntropyLoss applies log-softmax internally.
        return self.classify(x)
