import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms
import numpy as np

# based on https://github.com/ayaanzhaque/instruct-nerf2nerf/blob/main/metrics/clip_metrics.py
class ClipSimilarity(nn.Module):
    def __init__(self, name: str = "ViT-L/14@336px"):
        super().__init__()
        assert name in ("RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px")  # fmt: skip
        self.size = {"RN50x4": 288, "RN50x16": 384, "RN50x64": 448, "ViT-L/14@336px": 336}.get(name, 224)

        self.model, _ = clip.load(name, device="cpu", download_root="./")
        self.model.eval().requires_grad_(False)

        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def encode_text(self, text):
        text = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def encode_image(self, image):  # Input images in range [0, 1].
        image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
        image = image - rearrange(self.mean, "c -> 1 c 1 1").expand_as(image)
        image = image / rearrange(self.std, "c -> 1 c 1 1").expand_as(image)
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    @torch.no_grad()
    def forward(self, image_0, image_1, text_0, text_1):
        image_features_0 = self.encode_image(image_0)
        image_features_1 = self.encode_image(image_1)
        text_features_0 = self.encode_text(text_0)
        text_features_1 = self.encode_text(text_1)
        sim_0 = F.cosine_similarity(image_features_0, text_features_0)
        sim_1 = F.cosine_similarity(image_features_1, text_features_1)
        sim_direction = F.cosine_similarity(image_features_1 - image_features_0, text_features_1 - text_features_0)
        sim_image = F.cosine_similarity(image_features_0, image_features_1)
        return sim_0, sim_1, sim_direction, sim_image
    
    @torch.no_grad()
    def clip_similarity(
        self, images, text, batch_size=16
    ):    
        sim = 0
        text_features = self.encode_text(text)
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            image_features = self.encode_image(batch_images)
            sim += F.cosine_similarity(image_features, text_features).sum()

        sim /= len(images)
        return sim
    
    @torch.no_grad()
    def clip_directional_similarity(self, image_0, image_1, text_0, text_1, batch_size=16):    
        sim_direction = 0
        text_features_0 = self.encode_text(text_0)
        text_features_1 = self.encode_text(text_1)
        text_direction = text_features_1 - text_features_0

        for i in range(0, len(image_0), batch_size):
            batch_image_0 = image_0[i:i+batch_size]
            batch_image_1 = image_1[i:i+batch_size]
            image_features_0 = self.encode_image(batch_image_0)
            image_features_1 = self.encode_image(batch_image_1)

            sim_direction += F.cosine_similarity(image_features_1 - image_features_0, text_direction).sum()

        sim_direction /= len(image_0)
        return sim_direction
    
    @torch.no_grad()
    def clip_consistency(self, image_0_i, image_1_i, image_0_j, image_1_j, batch_size=16):    
        sim_image = 0
        for i in range(0, len(image_0_i), batch_size):
            batch_image_0_i = image_0_i[i:i+batch_size]
            batch_image_1_i = image_1_i[i:i+batch_size]
            image_features_0_i = self.encode_image(batch_image_0_i)
            image_features_1_i = self.encode_image(batch_image_1_i)

            batch_image_0_j = image_0_j[i:i+batch_size]
            batch_image_1_j = image_1_j[i:i+batch_size]
            image_features_0_j = self.encode_image(batch_image_0_j)
            image_features_1_j = self.encode_image(batch_image_1_j)

            sim_image += F.cosine_similarity(image_features_0_i - image_features_1_i, image_features_0_j - image_features_1_j).sum()

        sim_image /= len(image_0_i)
        return sim_image
    
    @torch.no_grad()
    def single_image_similarity(
        self, src_image, tgt_images, batch_size=16
    ):
        src_image_features = self.encode_image(src_image)
        
        sim = 0
        for i in range(0, len(tgt_images), batch_size):
            ref_images = tgt_images[i:i+batch_size]
            target_features = self.encode_image(ref_images)
            sim += F.cosine_similarity(src_image_features, target_features).sum()

        sim /= len(tgt_images)
        return sim
    

class DinoV2Similarity(nn.Module):
    normalize_transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    
    def __init__(self, name: str = "dinov2_vits14_reg"):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        self.model.eval()
        self.model.requires_grad_(False)
        self.patch_size = self.model.patch_size

    def __repr__(self):
        return f"DinoV2Similarity"

    def preprocess(self, image):
        # Normalize the image to the model's requirements
        image = image.float()
        image = self.normalize_transform(image)
        # find the image size that is divisible by the patch size
        closeest_size = int(np.ceil(image.shape[-1] / self.patch_size) * self.patch_size)
        image = F.interpolate(image, size=(closeest_size, closeest_size), mode='bilinear')
        return image

    @torch.no_grad()
    def forward(self, src_image, tgt_images, batch_size=16):
        src_image = self.preprocess(src_image)
        tgt_images = self.preprocess(tgt_images)

        src_feat = self.model(src_image)
        dist = 0
        for i in range(0, len(tgt_images), batch_size):
            target_images_batch = tgt_images[i:i+batch_size]
            dst_feat = self.model(target_images_batch)
            dist += F.cosine_similarity(src_feat, dst_feat).sum()

        dist /= len(tgt_images)
        return dist