## network modules
# import tinycudann as tcnn   # only works for T4 GPU in colab 
import torch
import torch.nn as nn
import torch.nn.functional as F

# if using tinycudann, uncomment the related region of code per class

def create_mlp(input_dim, latent_dim, output_dim, num_layers,
               output_activation=None, activation=nn.ReLU):
    layers = [nn.Sequential(nn.Linear(latent_dim if i > 0 else input_dim, latent_dim), activation())
              for i in range(num_layers - 1)]
    if num_layers == 1: # hacky
        layers = [nn.Linear(input_dim, latent_dim), activation()]
    layers.append(nn.Linear(latent_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)

# def create_tinycudann_mlp(input_dim, latent_dim, output_dim, num_layers,
#                           output_activation="None", activation="ReLU"):
#     return tcnn.Network(
#         n_input_dims=input_dim,
#         n_output_dims=output_dim,
#         network_config={
#             "otype": "FullyFusedMLP",  # Fully fused MLP for speed
#             "activation": activation,  # ReLU for hidden layers
#             "output_activation": output_activation,  # No activation at output
#             "n_neurons": latent_dim,  # Number of neurons per hidden layer
#             "n_hidden_layers": num_layers,  # Four hidden layers
#         }
#     )

# Traditional PE used in NeRF.
# If using TensoRF, we won't use this. 
# Also, Fourier Feature Transforms may be better.
class PositionalEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, log_space=False, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.log_space = log_space
        self.funcs = [lambda x: x] if kwargs.get("include_input", True) else []
        self.output_dim = input_dim * (len(self.funcs) + 2 * encoding_dim)
        self.device = kwargs.get("device", "cuda")

        if self.log_space:
            freqs = 2.0 ** torch.linspace(0.0, self.encoding_dim - 1, encoding_dim, device=self.device)
        else: # linear space
            freqs = torch.linspace(2.0 ** 0, 2.0 ** (self.encoding_dim - 1), self.encoding_dim, device=self.device)

        for freq in freqs:
            # alternate between cos/sin
            self.funcs.append(lambda x, freq=freq: torch.sin(x * freq))
            self.funcs.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, v):
        # 'v' is (HWN, 3), or flattened 3D positions of samples
        # resulting dim should be (2 * encoding_dim + 1)
        return torch.concat([func(v) for func in self.funcs], dim=-1)


# Intermediate Feature Network
class FeatureMLP(nn.Module):
    # input dimension dim_x should be post-PE
    def __init__(self, dim_x, dim_z=256, dim_out=256, num_layers=8):
        super().__init__()
        self.mlp = create_mlp(dim_x, dim_z, dim_out, num_layers=num_layers)
        self.output_dim = dim_out
        # self.mlp = create_tinycudann_mlp(dim_x, dim_z, dim_out, num_layers)

    def forward(self, x_enc):
        return self.mlp(x_enc)
    

# Scatter Network gets sigma_t, sigma_s, 'g'
class ScatterMLP(nn.Module):
    # outputs [sigma_s, sigma_s, HG parameter 'g']
    def __init__(self, dim_x=256, dim_z=128, dim_out=5, num_layers=1):
        super().__init__()
        self.mlp = create_mlp(dim_x, dim_z, dim_out, num_layers=num_layers)
        # self.mlp = create_tinycudann_mlp(dim_x, dim_z, dim_out, num_layers)

    def forward(self, features):
        output = self.mlp(features)
        sigma_t = torch.relu(output[:, 0:1])  # extinction
        sigma_s = torch.relu(output[:, 1:4])  # scattering (per RGB)
        g = torch.tanh(output[:, 4:5])  # g [-1, 1]
        return torch.concat([sigma_t, sigma_s, g], dim=1)


# SH coefficient predictor MLP
class SphericalHarmonicsMLP(nn.Module):
    # outputs SH coefficients
    def __init__(self, dim_x=256, dim_z=128, num_layers=8, lmax=2):
        super().__init__()
        self.lmax = lmax
        self.num_coefs = (lmax + 1) ** 2  # num of SH coefficients (* 3 for RGB later)
        self.mlp = create_mlp(dim_x, dim_z, self.num_coefs * 3, num_layers=num_layers)
        # self.mlp = create_tinycudann_mlp(dim_x, dim_z, self.num_coef * 3, num_layers)

    def forward(self, features):
        sh_coeffs = self.mlp(features)
        return sh_coeffs.view(-1, 3, self.num_coefs)  # Shape: [B, 3, (l_max+1)^2]
    

# Predicts final transmittance T
# example: (----->|||-->|-> then -> is T)
class VisibilityMLP(nn.Module):
    # Expects PE encoded position and dimensional vectors; takes in as input their concatenated vector
    def __init__(self, posenc_dim, direnc_dim, dim_z=256, dim_out=1, num_layers=4):
        super().__init__()
        input_dim = posenc_dim + direnc_dim
        self.mlp = create_mlp(input_dim, dim_z, dim_out, num_layers=num_layers, output_activation=nn.Sigmoid)
        # self.mlp = create_tinycudann_mlp(dim_x, dim_z, dim_out, num_layers, output_activation="Sigmoid")

    def forward(self, x_enc, d_enc):
        return self.mlp(torch.concat([x_enc, d_enc], dim=1))


# DEPRECATED -- decoupling is best
# class NeuralScatteringField(nn.Module):
#     def __init__(self,
#         env_map, # requires lighting
#         enc_pos=PositionalEncoder(3, 8),
#         enc_dir=PositionalEncoder(3, 1),
#         visibility_network=VisibilityMLP(),
#         feature_network=FeatureMLP(),
#         sh_network=SphericalHarmonicsMLP(),
#         main_network=ScatterMLP(),
#         num_samples=64
#     ):
#         super().__init__()
#         self.env_map = env_map
#         self.enc_pos = enc_pos
#         self.enc_dir = enc_dir
#         self.F = feature_network
#         self.S = sh_network
#         self.M = main_network
#         self.num_samples = num_samples

#     def single_scatter(self, x, d, sigma_s, sigma_t, g):
#         # single scattering samples
#         pass

#     def forward(self, x, d):
#         # this should only output: sigma_s, sigma_t, g, sh_coefs

#         # SH coefficinets from S

#         # get rgb from ray marching (volume rendering)

#         # return rgb, s_t, s_s, g, sh_coeffs
#         # should output: rgb (for loss), s_t, s_s, g, coefs, That
#         x_encoded = self.enc_pos(x)
#         d_encoded = self.enc_dir(d)
#         T_pred = self.V(x_encoded, d_encoded)
#         feats = self.F(x_encoded)
#         sh_coefs = self.SH(feats)

#         # visibility branch
#         T = self.V(x_encoded, d_encoded) # "transmittance given a direction @ x"
        

#         # main branch
#         # multiple scatter function(sh_coefs, g, sigma_s)
#         sigma_t, sigma_s, g = self.M(feats) # extinction, scatter, HG coefficients


#         # single scatter function(all output from self.M)


#         # return single + multi (image prediction)

