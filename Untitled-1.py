# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2LMHeadModel
import imageio, os
from torchvision import transforms
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from PIL import Image
import numpy as np
from tqdm import tqdm

run_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# model settings
RESOLUTION_WIDTH = 128
RESOLUTION_HEIGHT = 128
CHANNELS = 3
BOTTLENECK_DIM = 768

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
        # Clamp to [0, 1], convert to [0, 255] and uint8
        image_np = (image_tensor.clamp(0, 1).mul(255).byte().cpu().permute(1, 2, 0).numpy())
        return Image.fromarray(image_np)
    
    def pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """
        Convert a PIL image to a PyTorch tensor of shape (C, H, W) with values in [0, 1].
        
        Args:
            image (Image.Image): A PIL Image object.
        
        Returns:
            torch.Tensor: A tensor of shape (C, H, W) with pixel values in the range [0, 1].
        """
        return transforms.ToTensor()(image)  # Already returns (C, H, W)

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
    def __init__(self, in_channels=CHANNELS, latent_dim=BOTTLENECK_DIM, input_resolution=(RESOLUTION_WIDTH, RESOLUTION_HEIGHT)):
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
        return self.encoder_fc(x)

    def decode(self, z):
        z = self.decoder_fc(z)
        return self.decoder(z)

# %%
# Load the autoencoder
autoenc = ConvAutoencoder()
state_dict = torch.load("checkpoints/run9/autoenc.pth", map_location=run_device)
autoenc.load_state_dict(state_dict)
autoenc = autoenc.to(run_device).eval()

# Load the base model first
transformer = GPT2Model.from_pretrained("checkpoints/run2/gpt2_decap")
transformer = transformer.to(run_device).eval()

# Load image processor
proc = ImageProcessor()

# %%
def generate_frames(num_frames, context_length=32, initial_image_tensor=None, initial_image_count=16, autoenc=autoenc, transformer=transformer):
    with torch.no_grad():
        # holder dim: sequence / time / frames, channels, x, y
        total_seq = torch.zeros(1, CHANNELS, RESOLUTION_HEIGHT, RESOLUTION_WIDTH, device=run_device)
        
        if initial_image_tensor is not None:
            initial_image_tensor = initial_image_tensor.unsqueeze(0).repeat((initial_image_count, 1, 1, 1))
            total_seq = torch.cat((total_seq, initial_image_tensor))
        
        for _ in tqdm(range(num_frames)):
            # the slicing is on the seq dimentsion only
            current_slice = total_seq[-context_length:, :, :, :].clone()
            
            # encode the slice
            slice_latents = autoenc.encode(current_slice) # out dim: ( <= context_length, embed/bottleneck_dim)
            
            # feed latents thru transformer, add fake batch dim since unlike autoenc transformer needs a time dim
            slice_prediction = transformer(inputs_embeds=slice_latents.unsqueeze(0)).last_hidden_state # out dim: (1, <= context_length, embed/bottleneck_dim)
            
            # decode prediction back thru the autoenc to get images (again remove fake batch dim cuz autoenc doesnt respect time dim)
            slice_prediction = autoenc.decode(slice_prediction.squeeze(0)) # out dim: (1,  <= context_length, CHANNELS 3, RESOLUTION_HEIGHT 128, RESOLUTION_WIDTH 128)
            
            # append the last (actually future frame) onto the total_seq
            # the unsqueezes are to turn the (3, 128, 128) image from the local slice into a (1, 3, 128, 128) shape fit for the total seq tensor
            total_seq = torch.cat((total_seq, slice_prediction[-1].unsqueeze(0)), dim=0)
        
        return total_seq

# %%
def save_video(frames, output='output.mp4', fps=60):
    writer = imageio.get_writer(output, fps=fps)
    for frame in tqdm(frames):
        img = frame.permute(1,2,0).numpy()
        img = ((img + 1)/2 * 255).astype('uint8')
        writer.append_data(img)
    writer.close()
    print(f'Saved {output}')

# %%
init_img = proc.pil_to_tensor(Image.open("test.png").convert('RGB').resize((RESOLUTION_WIDTH, RESOLUTION_HEIGHT))).to(run_device)

frames = generate_frames(1200, initial_image_tensor=init_img).detach().cpu()
torch.cuda.empty_cache()
save_video(frames)

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



