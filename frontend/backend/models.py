"""
KSL Model Architectures
Contains all model definitions for inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Constants
# ============================================================================

NUMBER_CLASSES = sorted(["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"])
WORD_CLASSES = sorted(["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                       "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"])
ALL_CLASSES = NUMBER_CLASSES + WORD_CLASSES

# Graph edges for ST-GCN models (v10+)
HAND_EDGES = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17),
]

POSE_EDGES = [
    (42, 43), (42, 44), (43, 45), (44, 46), (45, 47), (46, 0), (47, 21),
]

POSE_INDICES = [11, 12, 13, 14, 15, 16]


# ============================================================================
# v14 ST-GCN Model (Graph-based)
# ============================================================================

def build_adjacency_matrix(n: int = 48) -> torch.Tensor:
    """Build normalized adjacency matrix for ST-GCN."""
    adj = np.zeros((n, n))

    # Left hand edges
    for i, j in HAND_EDGES:
        adj[i, j] = adj[j, i] = 1

    # Right hand edges (offset by 21)
    for i, j in HAND_EDGES:
        adj[i + 21, j + 21] = adj[j + 21, i + 21] = 1

    # Pose edges
    for i, j in POSE_EDGES:
        adj[i, j] = adj[j, i] = 1

    # Cross-hand connection (weak)
    adj[0, 21] = adj[21, 0] = 0.3

    # Self-loops
    adj += np.eye(n)

    # Normalize: D^(-0.5) * A * D^(-0.5)
    d = np.sum(adj, axis=1)
    d_inv = np.power(d, -0.5)
    d_inv[np.isinf(d_inv)] = 0

    return torch.FloatTensor(np.diag(d_inv) @ adj @ np.diag(d_inv))


class GConv(nn.Module):
    """Graph Convolution Layer."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.matmul(adj, x))


class STGCNBlock(nn.Module):
    """Spatial-Temporal Graph Convolution Block."""
    def __init__(self, in_channels: int, out_channels: int, adj: torch.Tensor,
                 kernel_size: int = 9, stride: int = 1, dropout: float = 0.3):
        super().__init__()
        self.register_buffer('adj', adj)
        self.gcn = GConv(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (kernel_size, 1),
                     padding=(kernel_size // 2, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        b, c, t, n = x.shape

        # Graph convolution
        x = x.permute(0, 2, 3, 1).reshape(b * t, n, c)
        x = self.gcn(x, self.adj)
        x = x.reshape(b, t, n, -1).permute(0, 3, 1, 2)

        # Batch norm and temporal convolution
        x = self.dropout(self.tcn(F.relu(self.bn1(x))))

        return F.relu(x + residual)


class STGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network for v10-v14 models."""
    def __init__(self, num_classes: int, num_nodes: int = 48, in_channels: int = 3,
                 hidden_dim: int = 128, num_layers: int = 6, temporal_kernel: int = 9,
                 dropout: float = 0.3, adj: torch.Tensor = None):
        super().__init__()

        if adj is None:
            adj = build_adjacency_matrix(num_nodes)
        self.register_buffer('adj', adj)

        self.data_bn = nn.BatchNorm1d(num_nodes * in_channels)

        # Channel progression: [3, 128, 128, 256, 256, 512, 512]
        channels = [in_channels] + [hidden_dim] * 2 + [hidden_dim * 2] * 2 + [hidden_dim * 4] * 2
        channels = channels[:num_layers + 1]

        # Build layers with stride=2 at layers 2 and 4 for temporal downsampling
        self.layers = nn.ModuleList([
            STGCNBlock(channels[i], channels[i + 1], adj, temporal_kernel,
                      stride=2 if i in [2, 4] else 1, dropout=dropout)
            for i in range(num_layers)
        ])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, n = x.shape

        # Data batch normalization
        x = self.data_bn(x.permute(0, 1, 3, 2).reshape(b, c * n, t))
        x = x.reshape(b, c, n, t).permute(0, 1, 3, 2)

        # ST-GCN layers
        for layer in self.layers:
            x = layer(x)

        # Classification
        x = self.pool(x).view(b, -1)
        return self.classifier(x)


# ============================================================================
# v8/v9 TemporalPyramid Model (Feature-based)
# ============================================================================

class TemporalPyramid(nn.Module):
    """Temporal Pyramid model for v8/v9."""
    def __init__(self, num_classes: int, feature_dim: int = 649,
                 hidden_dim: int = 320, dropout: float = 0.35):
        super().__init__()

        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # Multi-scale temporal convolutions
        self.temporal_scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=7, padding=3),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=15, padding=7),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.GELU()
            ),
        ])

        self.temporal_fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2 * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=3,
                           batch_first=True, bidirectional=True, dropout=dropout)

        # Multi-head attention
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(4)
        ])

        self.pre_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 * 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2 * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = self.input_proj(self.input_norm(x))

        # Multi-scale temporal features
        x_conv = x.transpose(1, 2)  # (B, H, T)
        multi_scale = torch.cat([scale(x_conv) for scale in self.temporal_scales], dim=1)
        multi_scale = multi_scale.transpose(1, 2)  # (B, T, H*3/2)

        x = self.temporal_fusion(multi_scale)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Multi-head attention
        contexts = []
        for head in self.attention_heads:
            attn_weights = F.softmax(head(lstm_out), dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)
            contexts.append(context)

        combined = torch.cat(contexts, dim=-1)

        return self.classifier(self.pre_classifier(combined))


# ============================================================================
# Model Factory
# ============================================================================

def create_model(version: str, num_classes: int, device: torch.device = None) -> nn.Module:
    """Create model based on version string."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    version = version.lower()

    if version in ["v10", "v11", "v12", "v13", "v14"]:
        adj = build_adjacency_matrix(48).to(device)
        model = STGCN(
            num_classes=num_classes,
            num_nodes=48,
            in_channels=3,
            hidden_dim=128,
            num_layers=6,
            temporal_kernel=9,
            dropout=0.5 if version == "v14" else 0.3,
            adj=adj
        )
    elif version in ["v8", "v9"]:
        feature_dim = 657 if version == "v9" else 649
        model = TemporalPyramid(
            num_classes=num_classes,
            feature_dim=feature_dim,
            hidden_dim=320,
            dropout=0.35
        )
    else:
        # Default to v8 for older versions
        model = TemporalPyramid(
            num_classes=num_classes,
            feature_dim=649,
            hidden_dim=320,
            dropout=0.35
        )

    return model.to(device)
