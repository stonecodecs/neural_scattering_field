## network modules
# import tinycudann as tcnn   # only works for T4 GPU in colab
import torch
import torch.nn as nn
import torch.nn.functional as F

# Traditional PE used in NeRF.
# If used with TensoRF, we won't use this. 
# Also, Fourier Feature Transforms may be better.
class PositionalEncoder(nn.Module):
    def __init__(self, input_dim, num_freqs):
        super().__init__()
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.out_dim = input_dim * 2 * num_freqs  # Sine and cosine for each frequency

    def forward(self, x):
        encoded = []
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)


# Intermediate Feature Network
class FeatureMLP(nn.Module):
    # input dimension dim_x should be post-PE
    def __init__(self, dim_x=3, dim_z=256, num_freqs=8):
        super().__init__()
        self.encoder = PositionalEncoder(dim_x, num_freqs)
        self.mlp = nn.Sequential(
            *[nn.Sequential(nn.Linear(256 if i > 0 else self.encoder.out_dim, 256), nn.ReLU())
              for i in range(8)]  # 8 layers
        )
        # self.mlp = tcnn.Network(
        #     n_input_dims=dim_x,
        #     n_output_dims=dim_z,
        #     network_config={
        #         "otype": "FullyFusedMLP",  # Fully fused MLP for speed
        #         "activation": "ReLU",  # ReLU for hidden layers
        #         "output_activation": "None",  # No activation at output
        #         "n_neurons": 256,  # Number of neurons per hidden layer
        #         "n_hidden_layers": 8,  # Four hidden layers
        #     }
        # )

    def forward(self, x):
        x_encoded = self.encoder(x)
        return self.mlp(x_encoded)
    

# Scatter Network gets sigma_t, sigma_s, 'g'
class ScatterMLP(nn.Module):
    # outputs [sigma_s (1), sigma_s (3), Henyey-Greenstein parameter 'g' (1)]
    def __init__(self, input_dim=256):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        # self.mlp = tcnn.Network(
        #     n_input_dims=input_dim,
        #     n_output_dims=5,
        #     network_config={
        #         "otype": "FullyFusedMLP",  # Fully fused MLP for speed
        #         "activation": "ReLU",  # ReLU for hidden layers
        #         "output_activation": "None",  # No activation at output
        #         "n_neurons": 128,  # Number of neurons per hidden layer
        #         "n_hidden_layers": 1,  # Four hidden layers
        #     }
        # )

    def forward(self, features):
        output = self.layer(features)
        sigma_t = torch.relu(output[:, 0])  # Extinction coefficient (σ ≥ 0)
        sigma_s = torch.sigmoid(output[:, 1:4])  # RGB albedo ∈ [0, 1]
        g = torch.tanh(output[:, 4])  # Asymmetry parameter ∈ [-1, 1]
        return sigma_t, sigma_s, g


# SH coefficient predictor MLP
class SphericalHarmonicsMLP(nn.Module):
    def __init__(self, input_dim=256, l_max=5):
        super().__init__()
        self.l_max = l_max
        self.num_coef = (l_max + 1) ** 2  # num of SH coefficients
        self.mlp = nn.Sequential(
            *[nn.Sequential(nn.Linear(128 if i > 0 else input_dim, 128), nn.ReLU())
              for i in range(8)],  # 8 layers
            nn.Linear(128, 3 * self.num_coef)  # RGB coefficients
        )
        # self.mlp = tcnn.Network(
        #     n_input_dims=input_dim,
        #     n_output_dims=3 * self.num_coef,
        #     network_config={
        #         "otype": "FullyFusedMLP",  # Fully fused MLP for speed
        #         "activation": "ReLU",  # ReLU for hidden layers
        #         "output_activation": "None",  # No activation at output
        #         "n_neurons": 128,  # Number of neurons per hidden layer
        #         "n_hidden_layers": 8,  # Four hidden layers
        #     }
        # )

    def forward(self, features):
        sh_coeffs = self.mlp(features)
        return sh_coeffs.view(-1, 3, self.num_sh_coeffs)  # Shape: [B, 3, (l_max+1)^2]
    

# Predicts transmittance T
class VisibilityMLP(nn.Module):
    def __init__(self, pos_dim=3, dir_dim=3, pos_freqs=8, dir_freqs=1):
        super().__init__()
        self.pos_encoder = PositionalEncoder(pos_dim, pos_freqs)
        self.dir_encoder = PositionalEncoder(dir_dim, dir_freqs)
        input_dim = self.pos_encoder.out_dim + self.dir_encoder.out_dim
        self.mlp = nn.Sequential(
            *[nn.Sequential(nn.Linear(256 if i > 0 else input_dim, 256), nn.ReLU())
              for i in range(4)],  # 4 layers
            nn.Linear(256, 1),
            nn.Sigmoid()  # Transmittance τ ∈ [0, 1]
        )
        # self.mlp = tcnn.Network(
        #     n_input_dims=input_dim,
        #     n_output_dims=1,
        #     network_config={
        #         "otype": "FullyFusedMLP",  # Fully fused MLP for speed
        #         "activation": "ReLU",  # ReLU for hidden layers
        #         "output_activation": "None",  # No activation at output
        #         "n_neurons": 256,  # Number of neurons per hidden layer
        #         "n_hidden_layers": 4,  # Four hidden layers
        #     }
        # )

    def forward(self, x, d):
        pos_enc = self.pos_encoder(x)
        dir_enc = self.dir_encoder(d)
        x = torch.cat([pos_enc, dir_enc], dim=-1)
        return self.mlp(x).squeeze(-1)

# multiple scattering NeRF
class NeuralScatteringField(nn.Module):
    def __init__(self,
        enc_pos=PositionalEncoder(3, 8),
        enc_dir=PositionalEncoder(2, 1),
        visibility_network=VisibilityMLP(),
        feature_network=FeatureMLP(),
        sh_network=SphericalHarmonicsMLP(),
        main_network=ScatterMLP()
    ):
        super().__init__()
        self.enc_pos = enc_pos
        self.enc_dir = enc_dir
        self.V = visibility_network
        self.F = feature_network
        self.SH = sh_network
        self.M = main_network

    def forward(self, x, d):
        x_encoded = self.enc_pos(x)
        d_encoded = self.enc_dir(d)
        T_pred = self.V(x_encoded, d_encoded)
        feats = self.F(x_encoded)
        sh_coefs = self.SH(feats)
        # multiple scatter function(sh_coefs, g, sigma_s)
        sigma_t, sigma_s, g = self.M(feats) # extinction, scatter, HG coefficients
        # single scatter function(all output from self.M)
        # return single + multi (image prediction)

