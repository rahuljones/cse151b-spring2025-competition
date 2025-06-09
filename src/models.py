import torch.nn as nn
from omegaconf import DictConfig
import torch
from lightning.pytorch import LightningDataModule
from src.utils import create_climate_data_array, create_comparison_plots
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

def visualize_high_loss_samples(
    model: torch.nn.Module,
    datamodule: LightningDataModule,
    k: int = 3,
):
    model.eval()
    dm = datamodule
    norm = dm.normalizer
    loader = dm.train_dataloader()

    records = []
    with torch.no_grad():
        for batch_idx, (x, y_true_norm) in enumerate(loader):
            y_pred_norm = model(x)                         # (B, C)
            loss_per_sample = ((y_pred_norm - y_true_norm)
                               .pow(2)
                               .mean(dim=1))               # (B,)
            for i, loss in enumerate(loss_per_sample):
                global_idx = batch_idx * loader.batch_size + i
                records.append((loss.item(),
                                global_idx,
                                y_pred_norm[i].cpu(),
                                y_true_norm[i].cpu()))

    topk = sorted(records, key=lambda r: r[0], reverse=True)[:k]

    for rank, (loss_val, idx, pred_n, true_n) in enumerate(topk, 1):
        # un‐normalize & reshape back to (1, C, H, W)
        pred_grid = norm.inverse_transform_output(pred_n.unsqueeze(0))
        true_grid = norm.inverse_transform_output(true_n.unsqueeze(0))

        lat, lon = dm.get_coords()
        time = [0]
        pred_xr = create_climate_data_array(pred_grid.numpy(), time, lat, lon,
                                            var_name="pred", var_unit="n/a")
        true_xr = create_climate_data_array(true_grid.numpy(), time, lat, lon,
                                            var_name="true", var_unit="n/a")

        fig = create_comparison_plots(
            true_xr.isel(time=0),
            pred_xr.isel(time=0),
            title_prefix=f"Train sample {idx}, loss={loss_val:.4f}"
        )
        fig.savefig(f"high_loss_{rank}.png")
        plt.close(fig)
def get_model(cfg: DictConfig):
    # Create model based on configuration
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}
    model_kwargs["n_input_channels"] = len(cfg.data.input_vars)
    model_kwargs["n_output_channels"] = len(cfg.data.output_vars)
    if cfg.model.type == "simple_cnn":
        model = SimpleCNN(**model_kwargs)
    elif cfg.model.type == "rnn":
        model = RNN(**model_kwargs)
    elif cfg.model.type == "cnn_rnn":
        return CNNRNN(**model_kwargs)
    elif cfg.model.type == "cnn_convlstm12":
        return CNN_ConvLSTM12(**model_kwargs)
    elif cfg.model.type == "resunet":
       # rename to match our ResUNet signature
        model_kwargs["in_channels"]  = model_kwargs.pop("n_input_channels")
        model_kwargs["out_channels"] = model_kwargs.pop("n_output_channels")
        # init_dim is already in model_kwargs
        return ResUNet(**model_kwargs)
    elif cfg.model.type == "attentionresunet":
       # rename to match our ResUNet signature
        model_kwargs["in_channels"]  = model_kwargs.pop("n_input_channels")
        model_kwargs["out_channels"] = model_kwargs.pop("n_output_channels")
        # init_dim is already in model_kwargs
        return AttentionResUNet(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return model


# --- Model Architectures ---


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.skip(identity)
        out = self.relu(out)

        return out


class SimpleCNN(nn.Module):
    def __init__(
        self,
        n_input_channels,
        n_output_channels,
        kernel_size=3,
        init_dim=64,
        depth=4,
        dropout_rate=0.2,
    ):
        super().__init__()

        # Initial convolution to expand channels
        self.initial = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),
        )

        # Residual blocks with increasing feature dimensions
        self.res_blocks = nn.ModuleList()
        current_dim = init_dim

        for i in range(depth):
            out_dim = current_dim * 2 if i < depth - 1 else current_dim
            self.res_blocks.append(ResidualBlock(current_dim, out_dim))
            if i < depth - 1:  # Don't double the final layer
                current_dim *= 2

        # Final prediction layers
        self.dropout = nn.Dropout2d(dropout_rate)
        self.final = nn.Sequential(
            nn.Conv2d(current_dim, current_dim // 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(current_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(current_dim // 2, n_output_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.initial(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.dropout(x)
        x = self.final(x)

        return x
    # after your training loop, e.g. in your script right before or after trainer.fit(...)
class CNNRNN(nn.Module):
    def __init__(
        self,
        n_input_channels: int,
        n_output_channels: int,
        kernel_size: int = 3,
        init_dim: int = 64,
        depth: int = 4,
        dropout_rate: float = 0.2,
        rnn_hidden_dim: int = 128,
    ):
        super().__init__()
        # Initial conv
        self.initial = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),
        )
        # Residual blocks + skip convs
        self.res_blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        curr_dim = init_dim
        for i in range(depth):
            out_dim = curr_dim * 2 if i < depth - 1 else curr_dim
            # main conv path
            block = nn.Sequential(
                nn.Conv2d(curr_dim, out_dim, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_dim, out_dim, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_dim),
            )
            self.res_blocks.append(block)
            # skip projection if needed
            if out_dim != curr_dim:
                self.skip_convs.append(nn.Conv2d(curr_dim, out_dim, kernel_size=1))
            else:
                self.skip_convs.append(nn.Identity())
            curr_dim = out_dim
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        # GRU over flattened grid
        self.gru = nn.GRU(input_size=curr_dim, hidden_size=rnn_hidden_dim, batch_first=True)
        # decode
        self.decoder = nn.Conv2d(rnn_hidden_dim, n_output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv backbone
        x = self.initial(x)
        for block, skip in zip(self.res_blocks, self.skip_convs):
            identity = skip(x)
            out = block(x)
            x = self.relu(out + identity)
        x = self.dropout(x)
        B, C_feat, H, W = x.shape
        # flatten to (B, H*W, C_feat)
        seq = x.view(B, C_feat, H * W).permute(0, 2, 1)
        # run GRU
        gru_out, _ = self.gru(seq)
        # reshape back (B, rnn_hidden_dim, H, W)
        feat = gru_out.permute(0, 2, 1).view(B, -1, H, W)
        # final 1x1 conv
        y = self.decoder(feat)
        return y

# ------------------------------------
# ConvLSTM Cell and Wrapper
# ------------------------------------
class ConvLSTMCell2d(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        # one conv produces input, forget, output, and candidate gates
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        i_f_o_g = self.conv(combined)
        i, f, o, g = i_f_o_g.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

class ConvLSTM2d(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell2d(in_channels, hidden_channels, kernel_size)

    def forward(self, seq):
        B, T, C, H, W = seq.shape
        device = seq.device
        h = torch.zeros(B, self.cell.hidden_channels, H, W, device=device)
        c = torch.zeros_like(h)
        outputs = []
        for t in range(T):
            h, c = self.cell(seq[:, t], h, c)
            outputs.append(h)
        return torch.stack(outputs, dim=1), (h, c)

# ------------------------------------
# CNN + ConvLSTM2d Model (T=12)
# ------------------------------------
class CNN_ConvLSTM12(nn.Module):
    """
    A model that:
      - Encodes each of 12 past months with a 2D CNN
      - Runs a ConvLSTM2d over that 12-step sequence
      - Decodes the last hidden state to 2 output channels on the 48×72 grid

    Input:  x of shape (B, 12, 5, 48, 72)
    Output: y of shape (B, 2, 48, 72)
    """
    def __init__(self,
                 n_input_channels: int,
                 n_output_channels: int,
                 lstm_hidden: int = 288
                 ):
        super().__init__()
        # per-frame CNN: n_input_channels→64→128→256→512
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(n_input_channels,  64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),             nn.ReLU(inplace=True),
            nn.Conv2d(128,256, 3, padding=1),             nn.ReLU(inplace=True),
            nn.Conv2d(256,512, 3, padding=1),             nn.ReLU(inplace=True),
        )
        # ConvLSTM over 12 frames
        self.convlstm = ConvLSTM2d(in_channels=512,
                                   hidden_channels=lstm_hidden,
                                   kernel_size=3)
        # final 1×1 decoder to n_output_channels
        self.decoder = nn.Conv2d(lstm_hidden, n_output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        assert T == 12, f"Expected 12 time-steps, got {{T}}"
        feats = [self.frame_encoder(x[:, t]) for t in range(T)]
        seq = torch.stack(feats, dim=1)
        _, (h_last, _) = self.convlstm(seq)
        y = self.decoder(h_last)
        return y


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = F.pad(x, [0, skip.size(3) - x.size(3), 0, skip.size(2) - x.size(2)])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class ResUNet(nn.Module):
    """
    A 4‐level ResUNet that takes in ‘in_channels’ and outputs ‘out_channels.’
    init_dim sets the number of filters in the very first conv; each subsequent
    level doubles the filters. We add two ResidualBlocks at each scale
    (instead of just one), and a deeper bottleneck.
    """
    def __init__(self, in_channels=5, out_channels=2, init_dim=32):
        super().__init__()
        # ─── Encoder ───────────────────────────────────────────────────────────────
        # initial conv: in_channels → init_dim
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),
            ResidualBlock(init_dim),     # first Residual at input
            ResidualBlock(init_dim)      # second Residual at input
        )
        # Down1: init_dim → init_dim*2
        self.down1 = Down(init_dim,     init_dim * 2)   #  32→64
        # Down2: init_dim*2 → init_dim*4
        self.down2 = Down(init_dim * 2, init_dim * 4)   #  64→128
        # Down3: init_dim*4 → init_dim*8
        self.down3 = Down(init_dim * 4, init_dim * 8)   # 128→256
        # Down4: init_dim*8 → init_dim*16
        self.down4 = Down(init_dim * 8, init_dim * 16)  # 256→512



        

        # ─── Bottleneck ────────────────────────────────────────────────────────────
        # After down4 pooling, we have (batch, init_dim*16, H/16, W/16)
        # Let’s go to init_dim*32 channels in the bottleneck:
        self.bottleneck = nn.Sequential(
            nn.Conv2d(init_dim * 16, init_dim * 32, kernel_size=3, padding=1),  # 512→1024
            nn.BatchNorm2d(init_dim * 32),
            nn.ReLU(inplace=True),
            ResidualBlock(init_dim * 32),
            ResidualBlock(init_dim * 32)
        )

        # ─── Decoder ───────────────────────────────────────────────────────────────
        # Up4: bottleneck (init_dim*32) → up to init_dim*16
        self.up4 = Up(init_dim * 32, init_dim * 16)    # 1024→512
        # Up3: init_dim*16 → init_dim*8
        self.up3 = Up(init_dim * 16, init_dim * 8)     # 512→256
        # Up2: init_dim*8 → init_dim*4
        self.up2 = Up(init_dim * 8, init_dim * 4)      # 256→128
        # Up1: init_dim*4 → init_dim*2
        self.up1 = Up(init_dim * 4, init_dim * 2)      # 128→64

        # ─── Final output conv ────────────────────────────────────────────────────
        # After up1 we have (batch, init_dim*2, H, W), so reduce → init_dim, then → out_channels
        self.out_conv = nn.Sequential(
            nn.Conv2d(init_dim * 2, init_dim, kernel_size=3, padding=1),  # 64→32
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_dim, out_channels, kernel_size=1)              # 32→2
        )

    def forward(self, x):
        # Encoder
        x0 = self.in_conv(x)             # → (batch, init_dim, H, W)
        s1, x1 = self.down1(x0)          # s1=(batch, init_dim*2, H/2, W/2), x1=(pooled)
        s2, x2 = self.down2(x1)          # s2=(batch, init_dim*4, H/4, W/4)
        s3, x3 = self.down3(x2)          # s3=(batch, init_dim*8, H/8, W/8)
        s4, x4 = self.down4(x3)          # s4=(batch, init_dim*16, H/16, W/16)

        # Bottleneck
        x5 = self.bottleneck(x4)         # → (batch, init_dim*32, H/16, W/16)

        # Decoder (upsample + skip connections)
        x6 = self.up4(x5, s4)            # → (batch, init_dim*16, H/8, W/8)
        x7 = self.up3(x6, s3)            # → (batch, init_dim*8, H/4, W/4)
        x8 = self.up2(x7, s2)            # → (batch, init_dim*4, H/2, W/2)
        x9 = self.up1(x8, s1)            # → (batch, init_dim*2, H, W)

        # Final output
        return self.out_conv(x9)         # → (batch, out_channels, H, W)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block:
      - Global average pool to get (B, C)
      - Two FC layers: C → C//r → C, followed by sigmoid
      - Multiply back onto the original feature map
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)      # → (B, C, 1, 1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze: global average pooling
        y = self.avg_pool(x).view(b, c)           # → (B, C)
        y = self.relu(self.fc1(y))                # → (B, C//r)
        y = self.sigmoid(self.fc2(y))             # → (B, C)
        y = y.view(b, c, 1, 1)                    # → (B, C, 1, 1)
        return x * y                              # Scale original input


# ─── 2. Spatial Attention ───────────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    """
    Spatial Attention block:
      - Concatenate max-pool and avg-pool along channel axis → (B, 2, H, W)
      - 7×7 conv → (B, 1, H, W), sigmoid → spatial mask
      - Multiply mask onto input
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # Compute max & avg along channel axis
        max_pool, _ = torch.max(x, dim=1, keepdim=True)    # → (B, 1, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)      # → (B, 1, H, W)
        concat = torch.cat([max_pool, avg_pool], dim=1)    # → (B, 2, H, W)
        mask = self.sigmoid(self.conv(concat))             # → (B, 1, H, W)
        return x * mask                                    # scale input by mask


# ─── 3. Attention-Enhanced ResidualBlock ────────────────────────────────────────
class ResidualBlock(nn.Module):
    """
    A standard 2-conv residual block, but with both SEBlock (channel attention)
    and SpatialAttention applied after the two conv layers.
    """
    def __init__(self, in_channels: int, reduction: int = 16, sa_kernel: int = 7):
        super().__init__()
        # First conv
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.relu  = nn.ReLU(inplace=True)
        # Second conv
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(in_channels)

        # Attention blocks
        self.se   = SEBlock(in_channels, reduction=reduction)
        self.sa   = SpatialAttention(kernel_size=sa_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Apply channel attention (SE) first, then spatial attention
        out = self.se(out)    # (B, C, H, W) scaled by channel‐wise weights
        out = self.sa(out)    # (B, C, H, W) scaled by spatial mask

        return self.relu(out + residual)


# ─── 4. Plug into Down/Up blocks as before ──────────────────────────────────────
class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Two conv layers + one ResidualBlock (with attention) at this scale
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels),   # now includes SE + spatial attention
            ResidualBlock(out_channels)    # stack two for more capacity
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if needed (in case of odd dimensions)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ─── 5. Full ResUNet with Attention ─────────────────────────────────────────────
class AttentionResUNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=2, init_dim=32):
        super().__init__()
        # ─── Encoder ───────────────────────────────────────────────────────────────
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),
            ResidualBlock(init_dim),     # two attention‐enhanced ResBlocks
            ResidualBlock(init_dim),
        )

        self.down1 = Down(init_dim,     init_dim * 2)   # 32→64
        self.down2 = Down(init_dim * 2, init_dim * 4)   # 64→128
        self.down3 = Down(init_dim * 4, init_dim * 8)   # 128→256
        self.down4 = Down(init_dim * 8, init_dim * 16)  # 256→512

        # ─── Bottleneck ────────────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv2d(init_dim * 16, init_dim * 32, kernel_size=3, padding=1),  # 512→1024
            nn.BatchNorm2d(init_dim * 32),
            nn.ReLU(inplace=True),
            ResidualBlock(init_dim * 32),  # this also has SE + spatial attention
            ResidualBlock(init_dim * 32),
        )

        # ─── Decoder ───────────────────────────────────────────────────────────────
        self.up4 = Up(init_dim * 32, init_dim * 16)    # 1024→512
        self.up3 = Up(init_dim * 16, init_dim * 8)     # 512→256
        self.up2 = Up(init_dim * 8,  init_dim * 4)     # 256→128
        self.up1 = Up(init_dim * 4,  init_dim * 2)     # 128→64

        # ─── Final Output Conv ────────────────────────────────────────────────────
        self.out_conv = nn.Sequential(
            nn.Conv2d(init_dim * 2, init_dim, kernel_size=3, padding=1),  # 64→32
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_dim, out_channels, kernel_size=1),             # 32→2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.in_conv(x)              # (B, init_dim,   H,   W)
        s1, x1 = self.down1(x0)           # s1=(B, init_dim*2, H/2, W/2)
        s2, x2 = self.down2(x1)           # s2=(B, init_dim*4, H/4, W/4)
        s3, x3 = self.down3(x2)           # s3=(B, init_dim*8, H/8, W/8)
        s4, x4 = self.down4(x3)           # s4=(B, init_dim*16, H/16, W/16)

        # Bottleneck
        x5 = self.bottleneck(x4)          # (B, init_dim*32, H/16, W/16)

        # Decoder (upsample + skip‐connections)
        x6 = self.up4(x5, s4)             # (B, init_dim*16, H/8,  W/8)
        x7 = self.up3(x6, s3)             # (B, init_dim*8,  H/4,  W/4)
        x8 = self.up2(x7, s2)             # (B, init_dim*4,  H/2,  W/2)
        x9 = self.up1(x8, s1)             # (B, init_dim*2,  H,    W)

        return self.out_conv(x9)          # (B, out_channels, H, W)
'''
class RNN(nn.Module):
    def __init__(
        self,
        n_input_channels: int,
        n_output_channels: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        height: int = 48,      # ← add these
        width: int = 72,       # ← and these
    ):
        super().__init__()
        self.height = height
        self.width = width

        # LSTM encoder (same as before)
        self.lstm = nn.LSTM(
            input_size=n_input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        # Now our head must predict every pixel:
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            # output hidden → (n_output_channels * H * W)
            nn.Linear(hidden_size // 2, n_output_channels * height * width),
        )

    def forward(self, x):
        # If you get a 4D input (B,C,H,W), turn it into a sequence of length H*W:
        if x.dim() == 4:
            B, C, H, W = x.shape
            # → (B, seq_len=H*W, features=C)
            x = x.view(B, C, H * W).permute(0, 2, 1)
    
        # x is now (B, seq_len, n_input_channels)
        outputs, _ = self.lstm(x)
        last = outputs[:, -1, :]                  # (B, hidden_size)
    
        out = self.regressor(last)                # (B, n_output_channels*H*W)
        B = out.shape[0]  # Get the batch size
        # reshape back to a full grid:
        return out.view(B,
                        -1,                      # n_output_channels
                        self.height,
                        self.width)
class CNNRNN(nn.Module):
    def __init__(
        self,
        n_input_channels: int,       # this is your sequence length
        n_output_channels: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        height: int = 48,
        width: int = 72,
    ):
        super().__init__()
        self.height, self.width = height, width
        self.seq_len = n_input_channels

        # LSTM processes one scalar per time‐step, per cell
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        # map final hidden state → n_output_channels
        self.head = nn.Linear(hidden_size, n_output_channels)

    def forward(self, x):
        # x: (B, T=n_input_channels, H, W)
        B, T, H, W = x.shape

        # fold spatial dims into batch
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, T)
        x = x.view(-1, T)                       # (B*H*W, T)
        x = x.unsqueeze(-1)                     # (B*H*W, T, 1)

        # run through LSTM
        out_seq, _ = self.lstm(x)               # (B*H*W, T, hidden_size)
        h_last = out_seq[:, -1, :]              # (B*H*W, hidden_size)

        # project & unflatten
        y = self.head(h_last)                   # (B*H*W, n_output_channels)
        y = y.view(B, H, W, -1)                 # (B, H, W, n_output_channels)
        return y.permute(0, 3, 1, 2)            # (B, n_output_channels, H, W)
'''