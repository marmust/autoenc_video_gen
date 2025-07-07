# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2LMHeadModel
import imageio, os
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

run_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# video settings
RESOLUTION_WIDTH = 128
RESOLUTION_HEIGHT = 128
CHANNELS = 3
CONVERTED_FRAMERATE = 24

# model settings
WINDOW_SIZE = 48
ENCODED_DIM = 768

# %%
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
    def forward(self, x):
        return x + self.block(x)

class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=CHANNELS, latent_dim=ENCODED_DIM, input_resolution=(RESOLUTION_WIDTH, RESOLUTION_HEIGHT)):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),           # 32x32
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 128, 4, 2, 1),          # 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),         # 8x8
            nn.ReLU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_resolution)
            enc_out = self.encoder(dummy)
            self.flattened_size = enc_out.view(1, -1).shape[1]

        self.encoder_fc = nn.Linear(self.flattened_size, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, self.flattened_size)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, enc_out.shape[1:]),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32x32
            nn.ReLU(),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),  # 128x128
            nn.Tanh()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = F.gelu(self.encoder_fc(x))
        
        return x

    def decode(self, z):
        z = F.gelu(self.decoder_fc(z))
        z = self.decoder(z)
        
        return z

# %%
class ImageProcessor:
    def tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        """
        Convert a tensor to a PIL Image.
        
        Args:
            image_tensor (torch.Tensor): A tensor of shape (C, H, W) with pixel values in the range [0, 1].
        
        Returns:
            Image.Image: A PIL Image object.
        """
        # Clamp to [-1, 1], convert to [0, 255] and uint8
        image_np = (image_tensor.clamp(-1, 1).mul(255).byte().cpu().permute(1, 2, 0).numpy())
        return Image.fromarray(image_np)
    
    def pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """
        Convert a PIL image to a PyTorch tensor of shape (C, H, W) with values in [0, 1].
        
        Args:
            image (Image.Image): A PIL Image object.
        
        Returns:
            torch.Tensor: A tensor of shape (C, H, W) with pixel values in the range [0, 1].
        """
        return transforms.ToTensor()(image) * 2 - 1  # Already returns (C, H, W)

# %%
# Load the autoencoder
autoencoder = ConvAutoencoder()
autoenced_state_dict = torch.load("checkpoints/run1/autoenc.pth", map_location=run_device)
autoencoder.load_state_dict(autoenced_state_dict)
autoencoder = autoencoder.to(run_device).eval()

# load the transformer
transformer = GPT2LMHeadModel.from_pretrained("checkpoints/run1/gpt2_decap").transformer
transformer = transformer.to(run_device).eval()

# Load image processor
proc = ImageProcessor()

# %%
def generate_frames(num_frames, context_length=WINDOW_SIZE, autoencoder=autoencoder, transformer=transformer):
    autoencoder.eval()
    transformer.eval()
    
    total_seq = torch.zeros(num_frames + context_length, CHANNELS, RESOLUTION_HEIGHT, RESOLUTION_WIDTH, device=run_device)

    with torch.no_grad():
        for i in tqdm(range(num_frames)):
            current_slice = total_seq[i:i + context_length]
            
            slice_latents = autoencoder.encode(current_slice)
            
            print(torch.max(slice_latents))
            
            slice_latents += torch.rand(slice_latents.shape, device=run_device) * 1000
            
            prediction_latents = transformer(inputs_embeds=slice_latents.unsqueeze(0)).last_hidden_state
            
            prediction_frame = autoencoder.decode(prediction_latents.squeeze(0))  # shape: (context_length, C, H, W)
            
            total_seq[context_length + i] = prediction_frame[-1]
    
    return total_seq

# %%
def save_video(frames, output='output.mp4', fps=CONVERTED_FRAMERATE):
    writer = imageio.get_writer(output, fps=fps)
    for frame in tqdm(frames):
        img = frame.permute(1,2,0).numpy()
        img = ((img + 1)/2 * 255).astype('uint8')
        writer.append_data(img)
    writer.close()
    print(f'Saved {output}')

# %%
init_img = proc.pil_to_tensor(Image.open("test.png").convert('RGB').resize((RESOLUTION_WIDTH, RESOLUTION_HEIGHT))).to(run_device)

frames = generate_frames(512).detach().cpu()
torch.cuda.empty_cache()
save_video(frames)

# %%
1 / 0

# %%
frames.shape

# %%
img = Image.open("test.png").convert('RGB').resize((RESOLUTION_WIDTH, RESOLUTION_HEIGHT))

# %%
enc_img = proc.pil_to_tensor(img).to(run_device).unsqueeze(0)

# %%
latent = autoenc.encode(enc_img)

# %%
prediction = transformer(inputs_embeds=latent).last_hidden_state

# %%
decoded = autoenc.decode(prediction)

# %%
proc.tensor_to_pil(decoded.squeeze(0))

# %%
init_img

# %%



