import torch
from torch import nn

from src.utils.torch_utilities import resolve_device


def _conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


class Conv_VAE(nn.Module):
    """Original 256-dimensional stochastic convolutional autoencoder.

    Architecture is preserved; the training objective currently omits KL loss.
    Tensors are N x C x H x W: inputs are 3 x 128 x 960 in [0, 1] (see
    `src.utils.torch_utilities.to_model_input`). Weights are Glorot-uniform,
    biases zero.
    """

    latent_dim = 256

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            _conv(3, 4),
            _conv(4, 4),
            nn.MaxPool2d(2),
            _conv(4, 8),
            _conv(8, 8),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8 * 32 * 240, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 8 * 32 * 240),
            nn.ReLU(),
            nn.Unflatten(1, (8, 32, 240)),
            nn.Upsample(scale_factor=2),
            _conv(8, 8),
            _conv(8, 8),
            nn.Upsample(scale_factor=2),
            _conv(8, 8),
            _conv(8, 8),
            _conv(8, 3),
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def sample(self, eps=None):
        if eps is None:
            eps = torch.randn(100, self.latent_dim, device=self.encoder[0].weight.device)
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = self.encoder(x).chunk(2, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean)
        return eps * torch.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits

    def forward(self, x):
        mean, logvar = self.encode(x)
        return self.decoder(self.reparameterize(mean, logvar))


def load_trained_model(weights_path, device="auto"):
    """Rebuild the architecture and load weights saved by the training script."""
    device = resolve_device(device) if isinstance(device, str) else device
    model = Conv_VAE()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return model.to(device).eval()
