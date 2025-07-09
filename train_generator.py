# %%
import torch
import torch.nn as nn
from torchvision.models import vgg16
#from torchvision.io import VideoReader
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from decord import VideoReader, cpu
import torch.nn.functional as F
from math import floor
from torch.utils.tensorboard import SummaryWriter
from peft import get_peft_model, LoraConfig, TaskType
import os
from transformers import GPT2LMHeadModel, GPT2Model
import gc
import json
from typing import List, Tuple

# %%
# training hyperparams
BATCH_SIZE = 1
EPOCHS = 24
LR = 0.001

# video settings
RESOLUTION_WIDTH = 128
RESOLUTION_HEIGHT = 128
CHANNELS = 3
CONVERTED_FRAMERATE = 16

# model settings
WINDOW_SIZE = 46
ENCODED_DIM = 768

# misc pytorch settings
run_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TENSORBOARD_LOG_DIR = "runs/exp7"

# %%
class PerceptualLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        vgg = vgg16(pretrained=True).features.eval()

        for layer in vgg:
            if isinstance(layer, nn.ReLU):
                layer.inplace = False

        self.vgg = vgg
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Selected layers: low-level to mid-level features
        self.layers = {
            "0": "relu1_1",
            "3": "relu1_2",
            "8": "relu2_2",
            "15": "relu3_3"
        }

        # Prioritize early edges more explicitly
        self.layer_weights = weights or {
            "relu1_1": 2.0,
            "relu1_2": 1.5,
            "relu2_2": 0.7,
            "relu3_3": 0.2,
        }

    def forward(self, x, y):
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            name = self.layers.get(str(i))
            if name:
                weight = self.layer_weights[name]
                loss += weight * F.mse_loss(x, y)
            if i > max(map(int, self.layers.keys())):
                break
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, perceptual_weight=0.7, mse_weight=1.4):
        super().__init__()
        self.perceptual_loss = PerceptualLoss()
        self.mse_loss = nn.MSELoss()
        self.perceptual_weight = perceptual_weight
        self.mse_weight = mse_weight

    def forward(self, reconstructed_images, target_images):
        return (
            self.perceptual_weight * self.perceptual_loss(reconstructed_images, target_images)
            + self.mse_weight * self.mse_loss(reconstructed_images, target_images)
        )

loss_fn = CombinedLoss()
loss_fn = loss_fn.to(run_device)

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
        x = torch.sigmoid(self.encoder(x))
        x = torch.flatten(x, 1)
        x = torch.tanh(self.encoder_fc(x))
        
        return x

    def decode(self, z):
        z = torch.tanh(self.decoder_fc(z))
        z = torch.sigmoid(self.decoder(z))
        
        return z

# %%
class PreprocessingFrameDataset(Dataset):
    def __init__(self, folder_path, window_size=WINDOW_SIZE,
                 resize=(RESOLUTION_WIDTH, RESOLUTION_HEIGHT),
                 framerate=CONVERTED_FRAMERATE,
                 cache_dir='preprocessed_frames'):
        self.folder_path = folder_path
        self.window_size = window_size
        self.resize = resize
        self.framerate = framerate
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.resize_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(resize),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2 - 1)  # normalize [0,1] → [-1,1]
        ])

        self.frame_files = []
        self.index = []
        self._prepare_frames()

    def _prepare_frames(self):
        video_files = [f for f in os.listdir(self.folder_path) if f.endswith('.mp4')]
        
        for i, fname in enumerate(video_files):
            base = os.path.splitext(fname)[0]
            cache_path = os.path.join(self.cache_dir, base + '.pt')
            
            if not os.path.exists(cache_path):
                print(f'Preprocessing {fname} -> {cache_path}')
                vr = VideoReader(os.path.join(self.folder_path, fname), ctx=cpu())
                original_fps = vr.get_avg_fps()
                step = max(int(original_fps // self.framerate), 1)

                frame_indices = list(range(0, len(vr), step))
                n_frames = len(frame_indices)

                # Preallocate tensor for all resized frames (C, H, W)
                sample_frame = self.resize_transform(vr[0].asnumpy())  # to get shape
                C, H, W = sample_frame.shape
                frame_tensor = torch.empty((n_frames, C, H, W), dtype=sample_frame.dtype)

                for idx, frame_idx in enumerate(frame_indices):
                    frame_tensor[idx] = self.resize_transform(vr[frame_idx].asnumpy())

                torch.save(frame_tensor, cache_path)
                del frame_tensor, vr
                gc.collect()
            
            self.frame_files.append(cache_path)
            frame_len = torch.load(cache_path, map_location='cpu').shape[0]
            n_clips = floor(frame_len / self.window_size)
            for j in range(n_clips):
                self.index.append((i, j * self.window_size))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, start = self.index[idx]
        frames = torch.load(self.frame_files[file_idx], mmap=True, map_location='cpu')
        return frames[start:start + self.window_size]

# %%
class Trainer:
    def __init__(self, autoencoder, transformer, dataloader, RESOLUTION_HEIGHT=RESOLUTION_HEIGHT, RESOLUTION_WIDTH=RESOLUTION_WIDTH, BOTTLENECK_DIM=ENCODED_DIM, epochs=EPOCHS, lr=LR, device=run_device, loss=loss_fn, writer: SummaryWriter = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)):
        self.autoencoder = autoencoder
        self.transformer = transformer
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device
        self.writer = writer
        params = list(autoencoder.parameters()) + list(transformer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.loss_fn = loss
        
        self.RESOLUTION_HEIGHT = RESOLUTION_HEIGHT
        self.RESOLUTION_WIDTH = RESOLUTION_WIDTH
        self.BOTTLENECK_DIM = BOTTLENECK_DIM

    def train(self):
        self.autoencoder.train()
        self.transformer.train()
        
        for epoch in range(self.epochs):
            total_loss = 0.0
            
            for batch in self.dataloader:
                # !!! WARNING !!! the following segment was revealed to me in a dream !!! DO NOT MODIFY !!!
                
                batch = batch.to(self.device)
                B, T, C, H, W = batch.shape
                
                # split the frames into inputs and outputs (shifted by 1 futureward)
                input_frames = batch[:, :-1, :, :, :].clone()                    # (B, T-1, C, H, W)     [1 TOWARDS THE PAST]
                output_frames = batch[:, 1:, :, :, :].clone()                    # (B, T-1, C, H, W)     [1 TOWARDS THE FUTURE]
                
                # encode the WHOLE sequence even across batches
                #                  in:   (B, T-1, C, H, W)     ------>     out:   (B, T-1, BOTTLENECK_DIM)
                input_latents = self.autoencoder.encode(input_frames.view(B * (T - 1), C, H, W)).view(B, T - 1, self.BOTTLENECK_DIM)
                
                # run the latents thru the transformer
                #                          [1 TOWARDS THE PAST]                           [1 TOWARDS THE FUTURE]
                #                  in:   (B, T-1, BOTTLENECK_DIM)     ------>     out:   (B, T-1, BOTTLENECK_DIM)
                predicted_latents = self.transformer(inputs_embeds=input_latents).last_hidden_state
                
                # decode the predicted future back to frames
                #                  in:   (B, T-1, BOTTLENECK_DIM)     ------>     out:   (B, T-1, C, H, W)
                predicted_frames = self.autoencoder.decode(predicted_latents.reshape(-1, self.BOTTLENECK_DIM)).view(B, T - 1, C, H, W)
                
                # calculate the loss between the predicted frames and the target frames
                self.optimizer.zero_grad()
                
                # VGG loss CANNOT handler a time dim, so we combine the sequences with the batches tto trick VGG into thinking that its only batches
                # NOTE: this doesnt cause cross batch contamination since view works the EXACT same way twice, aligning each target frame with its corresponding prediction
                predicted_frames = predicted_frames.view(-1, C, H, W)  # (B * (T-1), C, H, W)
                output_frames = output_frames.view(-1, C, H, W)        # (B * (T-1), C, H, W)
                
                loss = self.loss_fn(predicted_frames, output_frames)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # just for the video logging, reshape back to a sequence format (with batches)
            output_frames_video = output_frames.view(B, T - 1, C, H, W) / 2 + 0.5 # normalize [-1,1] → [0,1]
            predicted_frames_video = predicted_frames.view(B, T - 1, C, H, W) / 2 + 0.5 # normalize [-1,1] → [0,1]
            
            # logging
            avg_loss = total_loss / len(self.dataloader)
            lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f} - LR: {lr:.6f}')
            self.writer.add_scalar("Loss/train", avg_loss, epoch)
            self.writer.add_scalar("LearningRate", lr, epoch)
            self.scheduler.step()
            
            self.writer.add_video("expected_output", output_frames_video, global_step=epoch, fps=CONVERTED_FRAMERATE)
            self.writer.add_video("transformer_output", predicted_frames_video, global_step=epoch, fps=CONVERTED_FRAMERATE)
        
        self.writer.close()

# %%
# load GPT2 and decapitate
gpt2_full = GPT2LMHeadModel.from_pretrained("gpt2", device_map=run_device)
decap_gpt2 = gpt2_full.transformer

# %%
# standart autoenc initialization here
autoencoder = ConvAutoencoder().to(run_device)

# %%
dataset = PreprocessingFrameDataset('video_dataset')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

trainer = Trainer(autoencoder, decap_gpt2, dataloader)

# %%
trainer.train()

# %%
torch.save(autoencoder.state_dict(), "checkpoints/run7/autoenc.pth")
gpt2_full.save_pretrained("checkpoints/run7/gpt2_decap")

# %%



