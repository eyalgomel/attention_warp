from pathlib import Path
from typing import Dict
import numpy as np
import torch

from attn_warp.render_utils import warp_image
from attn_warp.visualization_utils import visualize_attention_output


def dict_to_device(data: Dict, device: torch.device):
    for key in data:
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)
    return data


def image2latent(image, pipe):
    """Encode images to latents"""
    image = image * 2 - 1
    latents = pipe.vae.encode(image)['latent_dist'].mean
    latents = latents * 0.18215
    return latents


def depth2disparity(depth):
    """
    Args: depth torch tensor
    Return: disparity
    """
    disparity = 1 / (depth + 1e-5)
    disparity_map = disparity / torch.max(disparity) # 0.00233~1
    disparity_map = disparity_map[None].repeat(3, 1, 1)[None]
    return disparity_map


def run_prompts_and_pick_higher_score(image, prompts, model):
    best_prompt = None
    best_score = float('-inf')

    for prompt in prompts:
        _, _, _, logits = model.predict(image, prompt)
        if len(logits) == 0: 
            continue
        score = logits[0].item()
        
        if score > best_score:
            best_score = score
            best_prompt = prompt
    
    return best_prompt, best_score


class AttentionMapCapture(torch.nn.Module):
    def __init__(self, processor):
        super().__init__()
        self._processor = processor
        self.attention_maps = {}
        self._timestep = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        attention_map = self._processor(attn, hidden_states, encoder_hidden_states, attention_mask)
        self.attention_maps[self._timestep] = attention_map.detach()
        self._timestep += 1
        return attention_map


class AttentionMapWarpedInject(torch.nn.Module):
    def __init__(self, processor, num_heads: int = 8, diffusion_steps: int = 20,
                 attention_maps=None, pixel_mapping=None, name=None, alpha=0.9,
                 channels=(0,1), timestep_condition=None,  timestep_decay=None,
                 res_to_use=(32, 64), step_to_visulize=(), debug_folder=None):
        super().__init__()
        self._processor = processor
        self._timestep = 0
        self.num_heads = num_heads
        self.diffusion_steps = diffusion_steps
        self.attention_maps = attention_maps if attention_maps is not None else {}
        self.to_inject = attention_maps is not None
        self.pixel_mapping = pixel_mapping # (H, W, 2), 2D pixel coordinates in the target view for each pixel in the source view
        self.name = name
        self.alpha = alpha
        self.channels = np.array(channels)
        self.timestep_condition = timestep_condition
        self.timestep_decay = timestep_decay
        self.step_to_visulize = step_to_visulize
        self.res_to_use = res_to_use
        self.debug_folder = debug_folder

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        attention_map = self._processor(attn, hidden_states, encoder_hidden_states, attention_mask)
        assert self._timestep < self.diffusion_steps, f"timestep {self._timestep} | {self._idx}"
        
        res = int(attention_map.shape[1] ** 0.5)
        if res not in self.res_to_use:
            self._timestep += 1
            return attention_map

        if self.to_inject and self.timestep_condition(self._timestep):

            original_shape = attention_map.shape
            h = w = int(attention_map.shape[1] ** 0.5)
            out = self.attention_maps[self._timestep][self.channels].reshape(len(self.channels), h, w, -1).permute(0, 3, 1, 2) # (1, C, H, W) 
            res = out.shape[2]
            alpha = self.alpha if not callable(self.timestep_decay) else self.timestep_decay(self._timestep, self.diffusion_steps)

            if res == 64 and self._timestep in self.step_to_visulize and "attentions.0" in self.name and "attn1" in self.name:
                timestep = self._timestep
                name = f"{self.name} | T {timestep:<3d} | res {res:<3d}"
                save_path = self.debug_folder / f"{self.name}_{timestep:03d}_before.png"
                
                if not hasattr(AttentionMapWarpedInject, '_visualized'):
                    visualize_attention_output(self.attention_maps[self._timestep][self.channels][:1].cpu().numpy(), method='mean', num_heads=self.num_heads, h=h, w=w, name=f"{name}_before", save_path=save_path)
                    AttentionMapWarpedInject._visualized = True

                save_path = self.debug_folder / f"{self.name}_{timestep:03d}_original.png"
                visualize_attention_output(attention_map[self.channels][:1].cpu().numpy(), method='mean', num_heads=self.num_heads, h=h, w=w, name=f"{name}_original", save_path=save_path)
            
            attention_map = attention_map.reshape(attention_map.shape[0], h, w, -1)

            warped_image, _ = warp_image(out.to(torch.float), self.pixel_mapping[res])
            warped_image = warped_image.to(torch.half)
            depth_mask = self.pixel_mapping["depth_masks"][res].to(torch.half).to(warped_image.device)
            
            warped_image = warped_image * depth_mask.unsqueeze(0).unsqueeze(0)
            warped_image = warped_image.permute(0, 2, 3, 1)
            non_seen_mask = 1 - depth_mask.unsqueeze(0).unsqueeze(-1).expand_as(warped_image)
            out = warped_image * (1 - non_seen_mask) + attention_map[self.channels] * (non_seen_mask)

            # blend the two maps
            out = out * alpha + attention_map[self.channels] * (1 - alpha)

            attention_map[self.channels,:,:,:] = out
            attention_map = attention_map.reshape(original_shape)

            if res == 64 and self._timestep in self.step_to_visulize and "attentions.0" in self.name and "attn1" in self.name:
                save_path = self.debug_folder / f"{self.name}_{timestep:03d}_after.png"
                visualize_attention_output(attention_map[self.channels][:1].cpu().numpy(), method='mean', num_heads=self.num_heads, h=h, w=w, name=f"{name}_after", save_path=save_path)

        self._timestep += 1
        return attention_map
