from dataclasses import dataclass, field
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import CenterCrop, Resize, GaussianBlur
from torchvision.transforms.functional import to_pil_image
from typing import List, Literal, Optional, Type
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils import writer
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from igs2gs.igs2gs_pipeline import InstructGS2GSPipelineConfig, InstructGS2GSPipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.schedulers import DDIMScheduler, DDIMInverseScheduler

from ip2p.warp_attention import InstructPix2PixWarpAttention
from attn_warp.metrics import ClipSimilarity, DinoV2Similarity
from attn_warp.render_utils import adjust_camera_intrinsics, compute_mask_and_difference, warp_to_new_view
from attn_warp.utils import AttentionMapCapture, AttentionMapWarpedInject, depth2disparity, dict_to_device, image2latent, run_prompts_and_pick_higher_score
from splatfacto_2dgs.splatfacto_2dgs import get_viewmat
from lang_sam import LangSAM

try:
    from gsplat.rendering import rasterization_2dgs
except ImportError:
    print("Please install gsplat>=1.0.0")

DEFAULT_SEED = 42
def timestemp_condition_func(x):
    return x < 20

def timestep_decay_func(x, steps):
    return 0.9 - 0.9 * (x / steps)

class InstructGS2GSWarpAttentionPipeline(InstructGS2GSPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        del self.ip2p # make sure to delete the ip2p attribute before reassigning it

        self._load_model()
        self.original_processors = self.fetch_original_processors()
        self.config.num_heads = self.diffusion_model.unet.config.attention_head_dim
        self.edit_src_image = True
        self.first_step = None
        self.stage = -1
        if self.config.mask_prompt is not None:
            self.config.use_mask = True

    def _load_model(self):
        assert self.config.model_type in ["ip2p", "controlnet-depth"], f"Model must be either 'ip2p' or 'controlnet-depth', got {self.config.model_type}"
        if self.config.model_type == "ip2p":
            self.diffusion_model = InstructPix2PixWarpAttention(torch.device(self.config.model_device),
                                                                ip2p_use_full_precision=self.config.ip2p_use_full_precision,
                                                                **{"seed": self.config.seed})
            CONSOLE.log("Loaded IP2P model", style="green")
        elif self.config.model_type == "controlnet-depth":
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")
            pipe = StableDiffusionControlNetPipeline.from_pretrained(self.config.default_sd_ckpt, controlnet=controlnet).to(self.config.model_device).to(torch.float16)
            pipe.to(self.config.model_device)
            self.diffusion_model = pipe
            self.ddim_inverser = DDIMInverseScheduler.from_pretrained(self.config.default_sd_ckpt, subfolder="scheduler")
            self.ddim_scheduler = DDIMScheduler.from_pretrained(self.config.default_sd_ckpt, subfolder="scheduler")
            self.generator = torch.Generator(self.device).manual_seed(self.config.seed)
            CONSOLE.log("Loaded ControlNet model", style="green")

            added_prompt = 'best quality, extremely detailed'
            self.positive_reverse_prompt = self.config.reverse_prompt + ', ' + added_prompt
            self.positive_prompt = self.config.prompt + ', ' + added_prompt
            self.negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

        self.langsam = LangSAM()

    def edit_image(self, image: torch.Tensor, image_cond: torch.Tensor, action: Literal["capture", "inject"],
                    **kwargs):
        
        self.reset_attention_processors()

        if self.config.model_type == "ip2p":
            if action == "capture":
                self.capture_attention_maps()
            elif action == "inject":
                self.inject_attention_hook(kwargs["pixel_mapping"], str(self.step))
                pixel_mapping = kwargs.pop("pixel_mapping")
            
            if self.config.use_mask and not self.edit_src_image:
                front_prompts = ["face", "looking directly at the camera"]
                back_prompt = 'behind'
                best_prompt, best_score = run_prompts_and_pick_higher_score(to_pil_image(image[0]), [*front_prompts, back_prompt], self.langsam)
                if best_prompt == back_prompt:
                    prompt = self.config.prompt + f", viewed from behind, highlighting the subject's rear details"
                else:
                    prompt = self.config.prompt # do not add front view prompt

                self.text_embedding = self.diffusion_model.pipe._encode_prompt(
                        prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
                    )
            
            edited_image = self.diffusion_model.edit_image(self.text_embedding.to(self.config.model_device),
                                                           image.to(self.config.model_device),
                                                           image_cond.to(self.config.model_device),
                                                           image_guidance_scale=self.config.image_guidance_scale,
                                                           diffusion_steps=self.config.diffusion_steps,
                                                           lower_bound=self.config.lower_bound,
                                                           upper_bound=self.config.upper_bound,
                                                           **kwargs)
        elif self.config.model_type == "controlnet-depth":
            init_latent = image2latent(image.half(), self.diffusion_model)
            disparity = depth2disparity(image_cond[:,:,0])
            self.diffusion_model.scheduler = self.ddim_inverser
            latent, _ = self.diffusion_model(prompt=self.positive_reverse_prompt,
                                             num_inference_steps=self.config.diffusion_steps, 
                                             latents=init_latent, 
                                             image=disparity, return_dict=False, guidance_scale=0, output_type='latent')

            if action == "capture":
                self.capture_attention_maps()
            elif action == "inject":
                self.inject_attention_hook(kwargs["pixel_mapping"], str(self.step))
            self.diffusion_model.scheduler = self.ddim_scheduler

            if self.config.use_mask and not self.edit_src_image:
                front_prompts = ["face", "looking directly at the camera"]
                back_prompt = 'behind'
                best_prompt, best_score = run_prompts_and_pick_higher_score(to_pil_image(image[0]), [*front_prompts, back_prompt], self.langsam)
                if best_prompt == back_prompt:
                    positive_prompt = self.positive_prompt + f", viewed from behind, highlighting the subject's rear details"
                else:
                    positive_prompt = self.positive_prompt # do not add front view prompt
            else:
                positive_prompt = self.positive_prompt

            controlnet_input = {
                "generator": self.generator if self.config.use_generator and self.stage == 0 else None,
                "latents": latent if not (self.config.use_generator and self.stage == 0) else None,
            }
            edited_image = self.diffusion_model(prompt=[positive_prompt],
                                          negative_prompt=[self.negative_prompts],
                                        #   latents=latent,
                                          image=disparity,
                                          num_inference_steps=self.config.diffusion_steps,
                                          controlnet_conditioning_scale=1.,
                                          eta=0.,
                                          output_type='pt',
                                        #   generator=self.generator,
                                          **kwargs,
                                          **controlnet_input).images
        if action == "capture":
            self.collect_attention_maps()

        return edited_image

    def src_image_pipeline(self, step: int):
        idx = self.config.edit_idx
        camera, data = self.datamanager.next_train_idx(idx)

        if self.training:
            optimized_camera_to_world = self.model.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds
        self.src_viewmat = get_viewmat(optimized_camera_to_world)
        self.src_intrinsics = {}
        self.src_intrinsics[self.config.crop_size] = camera.get_intrinsics_matrices().to(self.device).detach()

        model_outputs = self.model(camera)
        original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
        self.src_image = original_image.detach().clone()
        self.model.src_normals = self.model.normals.detach().clone()
        rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

        if self.config.model_type == "ip2p":
            model_input = {"image": original_image.to(self.config.model_device),
                           "image_cond": original_image.to(self.config.model_device),
                           "guidance_scale": self.config.guidance_scale,
                           "action": "capture"}
        elif self.config.model_type == "controlnet-depth":
            model_input = {"image": original_image.to(self.config.model_device), 
                           "image_cond": model_outputs["depth"].to(self.config.model_device),
                           "action": "capture"}
        edited_image = self.edit_image(**model_input)

        concat_image = torch.cat([original_image.detach().cpu(), rendered_image.detach().cpu(), edited_image.detach().cpu()], dim=3)
        writer.put_image(f"source_image/{idx}", concat_image.squeeze().permute(1,2,0), step)

        edit_mask = torch.ones_like(edited_image.squeeze().permute(1,2,0))
        if self.config.use_mask:
            pil_image = to_pil_image(original_image[0])
            sam_masks, _, _, _ = self.langsam.predict(pil_image, self.config.mask_prompt)
            sam_masks = GaussianBlur(kernel_size=(5,5), sigma=1.5)(sam_masks[0].unsqueeze(0).to(torch.float32)).clamp(0, 1)
            writer.put_image(f"sam_mask/src", sam_masks[0][:,:,None].to(torch.float32), step)
            edit_mask = sam_masks[0, ..., None].to(self.device).to(torch.float32)

        # write edited image to dataloader
        edited_image = edited_image.to(original_image.dtype)
        self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0) * edit_mask + \
            original_image.squeeze().permute(1,2,0).to(edited_image.dtype).to(self.device) * (1 - edit_mask)
        data["image"] = edited_image.squeeze().permute(1, 2, 0)

        edited_image_np = (self.datamanager.cached_train[idx]["image"] * 255).cpu().numpy().astype(np.uint8)
        self.src_edited_image = edited_image_np

        metrics_dict = self.model.get_metrics_dict(model_outputs, data)
        loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)
        loss_dict = {k: v * 0 for k, v in loss_dict.items()}

        for res in [8, 16, 32, 64]:
            camera, data = self.datamanager.next_train_idx(idx)
            adjust_camera_intrinsics(camera, new_w=res, new_h=res, crop_w=self.config.center_crop_image, crop_h=self.config.center_crop_image)
            self.src_intrinsics[res] = camera.get_intrinsics_matrices().to(self.device).detach()

        return model_outputs, loss_dict, metrics_dict

    def ddpm_inversion_src_image_pipeline(self, step: int):
        """ support only in ip2p model """
        self.reset_attention_processors()
        idx = self.config.edit_idx
        camera, data = self.datamanager.next_train_idx(idx)
        adjust_camera_intrinsics(camera, new_w=self.config.crop_size, new_h=self.config.crop_size, crop_w=self.config.center_crop_image, crop_h=self.config.center_crop_image)

        if self.training:
            optimized_camera_to_world = self.model.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds
        self.src_viewmat = get_viewmat(optimized_camera_to_world)
        self.src_intrinsics = {}
        self.src_intrinsics[self.config.crop_size] = camera.get_intrinsics_matrices().to(self.device).detach()

        model_outputs = self.model(camera)
        original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
        original_image = CenterCrop((self.config.center_crop_image, self.config.center_crop_image))(original_image)
        original_image = Resize((self.config.crop_size, self.config.crop_size), antialias=True)(original_image)
    
        reference_image = cv2.imread(self.config.ddpm_reference_image)
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB) / 255.
        reference_image = torch.from_numpy(reference_image).float().permute(2, 0, 1).unsqueeze(0)

        self.src_image = original_image.detach().clone()
        rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

        guidance_scale = self.config.guidance_scale
        image_guidance_scale = self.config.image_guidance_scale

        edited_image = self.model.ddpm_inversion_forward(self.text_embedding.to(self.config.model_device),
                                            reference_image.to(self.config.model_device),
                                            original_image.to(self.config.model_device),
                                            guidance_scale=guidance_scale,
                                            image_guidance_scale=image_guidance_scale,
                                            diffusion_steps=self.config.diffusion_steps,
                                            lower_bound=self.config.lower_bound,
                                            upper_bound=self.config.upper_bound)

        self.capture_attention_maps()
        edited_image = self.model.ddpm_inversion_backward(self.text_embedding.to(self.config.model_device),
                                                         original_image.to(self.config.model_device),
                                                         guidance_scale=guidance_scale,
                                                         image_guidance_scale=image_guidance_scale,
                                                         diffusion_steps=self.config.diffusion_steps,
                                                         skip=0)

        if (edited_image.size() != rendered_image.size()):
            edited_image = F.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

        concat_image = torch.cat([original_image.detach().cpu(), rendered_image.detach().cpu(), edited_image.detach().cpu()], dim=3)
        writer.put_image(f"source_image/{idx}", concat_image.squeeze().permute(1,2,0), step)

        # write edited image to dataloader
        edited_image = edited_image.to(original_image.dtype)
        self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
        data["image"] = edited_image.squeeze().permute(1, 2, 0)

        metrics_dict = self.model.get_metrics_dict(model_outputs, data)
        loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

        for res in self.config.res_to_use:
            camera, data = self.datamanager.next_train_idx(idx)
            adjust_camera_intrinsics(camera, new_w=res, new_h=res, crop_w=self.config.center_crop_image, crop_h=self.config.center_crop_image)
            self.src_intrinsics[res] = camera.get_intrinsics_matrices().to(self.device).detach()

        self.collect_attention_maps()

        return model_outputs, loss_dict, metrics_dict

    def capture_attention_maps(self):
        self.attention_map_captures = {}
        for name, processor in self.diffusion_model.unet.attn_processors.items():
            self.attention_map_captures[name] = AttentionMapCapture(processor)

        self.diffusion_model.unet.set_attn_processor(self.attention_map_captures)
        CONSOLE.log("Register Hook: Capture Attention Maps", style="blue")

    def fetch_original_processors(self):
        original_processors = {}
        for name, processor in self.diffusion_model.unet.attn_processors.items():
            original_processors[name] = processor
        
        return original_processors

    def collect_attention_maps(self):
        attention_maps = {}
        for name, processor in self.diffusion_model.unet.attn_processors.items():
            if processor.attention_maps is not None:
                attn_maps = processor.attention_maps
                attention_maps[name] = dict_to_device(attn_maps, self.config.model_device)
        self.attention_maps = attention_maps
        CONSOLE.log("Collect Attention Maps", style="green")

    def before_train(self, trainer, step: int):
        # do once before training
        self.step = step
        if self.first_step is None:
            self.attention_debug_folder = (trainer.base_dir / "edited_images/attentions")
            self.first_step = step
            for idx in range(len(self.datamanager.cached_train)):
                data = self.datamanager.cached_train[idx]
                curr_h, curr_w = data["image"].shape[:2]
                image = data["image"]
                image = image.unsqueeze(0).permute(0, 3, 1, 2)
                image = F.interpolate(image, size=(self.config.crop_size, self.config.crop_size), mode='bilinear')
                data["image"] = image.squeeze().permute(1, 2, 0)

                image = self.datamanager.original_cached_train[idx]["image"]
                image = image.unsqueeze(0).permute(0, 3, 1, 2)
                image = F.interpolate(image, size=(self.config.crop_size, self.config.crop_size), mode='bilinear')
                self.datamanager.original_cached_train[idx]["image"] = image.squeeze().permute(1, 2, 0)

                camera = self.datamanager.train_dataset.cameras[idx : idx + 1]
                adjust_camera_intrinsics(camera, new_w=self.config.crop_size, new_h=self.config.crop_size, crop_w=curr_w, crop_h=curr_h)
                for attr in camera.__dict__.keys():
                    if isinstance(camera.__dict__[attr], torch.Tensor):
                        self.datamanager.train_dataset.cameras.__dict__[attr].data[idx : idx + 1] = camera.__dict__[attr]

        stage = (step - self.first_step) // self.config.gs_steps
        if stage > self.stage:
            self.stage = min(stage, 2)
            # generate indices for training
            if stage == 0:
                self.indices_subset = np.random.choice(len(self.datamanager.cached_train), size=self.config.subset_size, replace=False)
                self.all_edited_indices = set(self.indices_subset)

            else: # choose from the rest of the dataset
                self.all_edited_indices = self.all_edited_indices.union(set(self.indices_subset))
                not_edited = set(range(len(self.datamanager.cached_train))) - self.all_edited_indices
                if not not_edited:
                    self.indices_subset = np.random.choice(len(self.datamanager.cached_train), size=self.config.subset_size, replace=False)
                else:
                    if len(not_edited) < self.config.subset_size:
                        remaining_needed = self.config.subset_size - len(not_edited)
                        additional_indices = np.random.choice(range(len(self.datamanager.cached_train)), size=remaining_needed, replace=False)
                        self.indices_subset = np.concatenate((list(not_edited), additional_indices))
                    else:
                        self.indices_subset = np.random.choice(list(not_edited), size=self.config.subset_size, replace=False)


    def save_artifacts(self, trainer, step: int):
        (trainer.base_dir / "edited_images").mkdir(exist_ok=True)
        out_path = trainer.base_dir / "edited_images" / f"{self.config.edit_idx:04d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(self.src_edited_image, cv2.COLOR_BGR2RGB))

    @torch.no_grad()
    def evaluation(self, trainer, step: int):
        if self.config.evaluate:
            from nerfstudio.scripts.render import RenderCameraPath
            import tempfile

            clip_eval = ClipSimilarity().to(self.device)
            dino_eval = DinoV2Similarity().to(self.device)

            train_rendered_images = []
            gt_edited_images = []
            original_images = []
            for idx in range(len(self.datamanager.cached_train)):
                camera, _ = self.datamanager.next_train_idx(idx)
                model_outputs = self.model(camera)
                rendered_image = model_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)
                gt_edit_image = self.datamanager.cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
                original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)

                train_rendered_images.append(rendered_image.to(self.device))
                gt_edited_images.append(gt_edit_image.to(self.device))
                original_images.append(original_image.to(self.device))

            train_rendered_images = torch.cat(train_rendered_images, dim=0).to(self.device)
            gt_edited_images = torch.cat(gt_edited_images, dim=0).to(self.device)
            original_images = torch.cat(original_images, dim=0).to(self.device)

            src_rendered_image = torch.from_numpy(self.src_edited_image).permute(2,0,1).unsqueeze(dim=0).to(self.device) / 255.

            # 2. evaluate metrics: edit psnr, clip similarity, clip directional similarity, dino single image similarity, clip single image similarity
            edit_psnr = self.model.psnr(train_rendered_images[list(self.all_edited_indices)], gt_edited_images[list(self.all_edited_indices)]).mean()
            clip_sim = clip_eval.clip_similarity(train_rendered_images, self.config.target_prompt).mean()
            clip_dir_sim = clip_eval.clip_directional_similarity(original_images, train_rendered_images,
                                                                    self.config.src_prompt, self.config.target_prompt).mean()
            dino_single_image_similarity = dino_eval(src_rendered_image, train_rendered_images)
            clip_single_image_similarity = clip_eval.single_image_similarity(src_rendered_image, train_rendered_images)

            # 3. render novel view path
            if self.config.novel_view_path is not None:
                yaml_file = trainer.base_dir / "config.yml"
                if not yaml_file.exists():
                    return
                # Render novel view path using edited model
                renderer = RenderCameraPath(camera_path_filename=self.config.novel_view_path, load_config=yaml_file,
                                            output_path=trainer.base_dir / "render.mp4")
                renderer.main()

            # 5. log metrics
            writer.put_scalar("Evaluation/Edit PSNR", edit_psnr, step)
            writer.put_scalar("Evaluation/Clip Similarity", clip_sim, step)
            writer.put_scalar("Evaluation/Clip Directional Similarity", clip_dir_sim, step)
            writer.put_scalar("Evaluation/Dino Single Image Similarity", dino_single_image_similarity, step)
            writer.put_scalar("Evaluation/Clip Single Image Similarity", clip_single_image_similarity, step)

            CONSOLE.log(f"Edit PSNR: {edit_psnr:.4f}", style="green")
            CONSOLE.log(f"Clip Similarity: {clip_sim:.4f}", style="green")
            CONSOLE.log(f"Clip Directional Similarity: {clip_dir_sim:.4f}", style="green")
            CONSOLE.log(f"Dino Single Image Similarity: {dino_single_image_similarity:.4f}", style="green")
            CONSOLE.log(f"Clip Single Image Similarity: {clip_single_image_similarity:.4f}", style="green")

            # save as json
            metrics = {"Edit PSNR": edit_psnr.item(),
                       "Clip Similarity": clip_sim.item(),
                       "Clip Directional Similarity": clip_dir_sim.item(),
                       "Dino Single Image Similarity": dino_single_image_similarity.item(),
                       "Clip Single Image Similarity": clip_single_image_similarity.item(),
                       "edit_idx": self.config.edit_idx,
                       "src_prompt": self.config.src_prompt,
                       "target_prompt": self.config.target_prompt,
                       "edit_prompt": self.config.prompt,
                       "mask_prompt": self.config.mask_prompt,
                       }

            with open(trainer.base_dir / "results.json", "w") as f:
                json.dump(metrics, f)

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        cbs = super().get_training_callbacks(training_callback_attributes)
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.before_train, args=[training_callback_attributes.trainer]))
        cbs.append(TrainingCallback([TrainingCallbackLocation.AFTER_TRAIN], self.evaluation, args=[training_callback_attributes.trainer]))
        cbs.append(TrainingCallback([TrainingCallbackLocation.AFTER_TRAIN], self.save_artifacts, args=[training_callback_attributes.trainer]))
        return cbs
    
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
      
        self.during_edit = False
        if (step - self.first_step) % self.config.gs_steps == 0:
            self.makeSquentialEdits = True

        if self.edit_src_image:
            if not self.config.ddpm_inversion:
                model_outputs, loss_dict, metrics_dict = self.src_image_pipeline(step)
            else:
                model_outputs, loss_dict, metrics_dict = self.ddpm_inversion_src_image_pipeline(step)
            self.edit_src_image = False
            return model_outputs, loss_dict, metrics_dict

        if (not self.makeSquentialEdits):
            if step % 10 == 0:
                camera, data = self.datamanager.next_train_idx(self.config.edit_idx)
            else:
                camera, data = self.datamanager.next_train_idx(self.indices_subset[step % len(self.indices_subset)])
            model_outputs = self.model(camera)

            if data["image_idx"] in [1, 20, 40]:
                rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
                original_image = data["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
                concat_image = torch.cat([rendered_image.detach().cpu(), original_image.detach().cpu()], dim=3)
                writer.put_image(f"rendered_image/{data['image_idx']}", concat_image.squeeze().permute(1,2,0), step)

        else:
            self.during_edit = True
            idx = self.indices_subset[self.curr_edit_idx]
            camera, data = self.datamanager.next_train_idx(idx)

            if self.training:
                optimized_camera_to_world = self.model.camera_optimizer.apply_to_camera(camera)
            else:
                optimized_camera_to_world = camera.camera_to_worlds
            dst_viewmat = get_viewmat(optimized_camera_to_world).to(self.device)
            dst_intrinsics = camera.get_intrinsics_matrices().to(self.device)

            model_outputs = self.model(camera)

            original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
            rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

            ## Warp and inject attention maps
            # 1. generate mask
            dst_normals = self.model.normals.detach()
            with torch.no_grad():
                src_normals = F.normalize(self.model.src_normals, dim=1)
                dst_normals = F.normalize(dst_normals, dim=1)

                angle = torch.linalg.vecdot(src_normals, dst_normals).acos() * 180 / np.pi
                normal_weight = angle < 60 

                opacities_crop = self.model.opacities[normal_weight]
                means_crop = self.model.means[normal_weight]
                features_dc_crop = self.model.features_dc[normal_weight]
                features_rest_crop = self.model.features_rest[normal_weight]
                scales_crop = self.model.scales[normal_weight]
                quats_crop = self.model.quats[normal_weight]
                colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

                optimized_camera_to_world = self.model.camera_optimizer.apply_to_camera(camera)
                viewmat = get_viewmat(optimized_camera_to_world)
                K = camera.get_intrinsics_matrices().cuda()

                (
                render,
                alpha,
                _,
                _,
                _,
                _,
                _,
                ) = rasterization_2dgs(means=means_crop,
                                    quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                                    scales=torch.exp(scales_crop),
                                    opacities=torch.sigmoid(opacities_crop).squeeze(-1),
                                    colors=colors_crop,
                                    sh_degree=self.model.config.sh_degree,
                                    viewmats=viewmat,
                                    Ks=K,
                                    width=self.config.crop_size,
                                    height=self.config.crop_size,
                                    near_plane=0.2,
                                    far_plane=200,
                                    render_mode="RGB+ED",
                                    distloss=self.model.config.dist_loss,
                                    packed=False,
                                    absgrad=True,
                                    sparse_grad=False)
                
                alpha = alpha[:, ...]
                background = self.model._get_background_color()
                rgb = render[:, ..., :3] + (1 - alpha) * background
                rgb = torch.clamp(rgb, 0.0, 1.0)
                r = rgb[0].permute(2,0,1)
                depth_im = render[:, ..., 3:4]
                depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
                depth_map = depth_im[:,:,0].detach()

            mask = compute_mask_and_difference(dst_intrinsics, dst_viewmat, depth_map, self.src_intrinsics[self.config.crop_size], self.src_viewmat,
                                               self.src_image.to(self.device), original_image.to(self.device))

            edit_mask = torch.ones_like(original_image.squeeze().permute(1,2,0)).to(self.device)
            if self.config.use_mask:
                edited_pil = to_pil_image(original_image.squeeze(0))
                sam_masks, _, _, _ = self.langsam.predict(edited_pil, self.config.mask_prompt)
                sam_masks = GaussianBlur(kernel_size=(5,5), sigma=1.5)(sam_masks[0].unsqueeze(0).to(torch.float32)).clamp(0, 1)
                edit_mask = sam_masks[0, ..., None].to(self.device).to(torch.float32)
                writer.put_image(f"sam_mask/{idx}", sam_masks[0][:,:,None].to(torch.float32), step)
                mask = mask * sam_masks[0].to(mask.device)

            # 2. compute warp
            warped_pixel_coords_by_res = {}
            warped_pixel_coords_by_res["depth_masks"] = {}
            for res in self.config.res_to_use:
                depth_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(res, res), mode='bilinear').squeeze()
                depth_mask = (depth_mask > 0.5).float()
                warped_pixel_coords_by_res["depth_masks"][res] = depth_mask

                depth_map_res = F.interpolate(depth_map.unsqueeze(0).unsqueeze(0), size=(res, res), mode='bilinear').squeeze()
                camera, _ = self.datamanager.next_train_idx(idx)
                adjust_camera_intrinsics(camera, new_w=res, new_h=res, crop_w=self.config.center_crop_image, crop_h=self.config.center_crop_image)
                res_dst_intrinsics = camera.get_intrinsics_matrices().to(self.device)
                res_src_intrinsics = self.src_intrinsics[res]
                warped_pixel_coords = warp_to_new_view(res_dst_intrinsics[0], dst_viewmat[0], depth_map_res, res_src_intrinsics[0], self.src_viewmat[0])
                warped_pixel_coords_by_res[res] = warped_pixel_coords.to(self.config.model_device)

            # 3. edit image
            if self.config.model_type == "ip2p":
                model_input = {"image": rendered_image.to(self.config.model_device) if self.stage != 0 else original_image.to(self.config.model_device),
                               "image_cond": original_image.to(self.config.model_device),
                               "action": "inject",
                               "pixel_mapping": warped_pixel_coords_by_res,
                               "guidance_scale": self.config.guidance_scale - self.stage * 1.5,
                               "use_last_timesteps": True}
            elif self.config.model_type == "controlnet-depth":
                model_input = {"image": rendered_image.to(self.config.model_device) if self.stage != 0 else original_image.to(self.config.model_device),
                               "image_cond": model_outputs["depth"].to(self.config.model_device),
                               "action": "inject",
                               "pixel_mapping": warped_pixel_coords_by_res,
                               "guidance_scale": self.config.guidance_scale}

            edited_image = self.edit_image(**model_input)

            if (edited_image.size() != rendered_image.size()):
                edited_image = F.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

            # Visualization
            normal_mask = self.compute_soft_mask_from_normals(dst_viewmat[0, :3, :3].T, model_outputs["normals"])
            top_row = torch.cat([original_image.detach().cpu(),
                                 rendered_image.detach().cpu(),
                                 edited_image.detach().cpu()], dim=3)
            bottom_row = torch.cat([mask[None].expand_as(edited_image).detach().cpu(), 
                                    model_outputs["normals_for_viewer"].permute(2,0,1)[None].detach().cpu(),
                                    normal_mask[:,:,None].permute(2,0,1)[None].expand_as(edited_image).detach().cpu()**4
                                    ], dim=3)
            concat_image = torch.cat([top_row, bottom_row], dim=2)
            writer.put_image(f"edited_image/{idx}", concat_image.squeeze().permute(1,2,0), step)

            # write edited image to dataloader
            edited_image = edited_image.to(original_image.dtype)
            self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0) * edit_mask + \
                                                          original_image.squeeze().permute(1,2,0).to(edited_image.dtype).to(self.device) * (1 - edit_mask)
            data["image"] = self.datamanager.cached_train[idx]["image"]

            #increment curr edit idx
            self.curr_edit_idx += 1
            if (self.curr_edit_idx >= len(self.indices_subset)):
                self.curr_edit_idx = 0
                self.makeSquentialEdits = False

        metrics_dict = self.model.get_metrics_dict(model_outputs, data)
        loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

        # zero loss during edit
        if self.during_edit:
            loss_dict = {k: v * 0 for k, v in loss_dict.items()}
        
        return model_outputs, loss_dict, metrics_dict

    def inject_attention_hook(self, warped_pixel_coords_by_res, step: str):
        attention_map_inject = {}
        for name, processor in self.diffusion_model.unet.attn_processors.items():
            if self.config.only_cross_attention:
                maps = self.attention_maps[name] if "attn2" in name else None
            elif self.config.only_self_attention:
                maps = self.attention_maps[name] if "attn1" in name else None
            else:
                maps = self.attention_maps[name]
            if self.config.use_only_up_blocks:
                maps = maps if "down_blocks" not in name else None

            attention_map_inject[name] = AttentionMapWarpedInject(processor, num_heads=self.config.num_heads, diffusion_steps=self.config.diffusion_steps,
                                                                  attention_maps=maps, pixel_mapping=warped_pixel_coords_by_res, name=f"{self.indices_subset[self.curr_edit_idx]}_{name}",
                                                                  alpha=self.config.alpha, channels=self.config.channels, timestep_condition=self.config.timestep_condition,
                                                                  timestep_decay=self.config.timestep_decay, res_to_use=self.config.res_to_use, 
                                                                  step_to_visulize=[],
                                                                  debug_folder=self.attention_debug_folder)
                                                                  

        self.diffusion_model.unet.set_attn_processor(attention_map_inject)
        CONSOLE.log("Register Hook: Inject Attention Maps", style="yellow")

    def reset_attention_processors(self):
        attention_map_none = {}
        for name in self.diffusion_model.unet.attn_processors.keys():
            attention_map_none[name] = self.original_processors[name]

        self.diffusion_model.unet.set_attn_processor(attention_map_none)
        CONSOLE.log("Reset Attention Processors", style="red")

    def compute_soft_mask_from_normals(self, R, normals):
        # Step 1: Compute camera direction and normalize
        camera_dir = -R[:, 2]  # Extract the camera's forward direction
        camera_dir = F.normalize(camera_dir, dim=-1)  # Normalize the direction
        
        # Step 2: Normalize the normal vectors
        normals_norm = F.normalize(normals, dim=-1)  # Shape: (H, W, 3)
        
        # Step 3: Compute the dot product between normals and camera direction
        normal_weight = torch.linalg.vecdot(normals_norm, camera_dir.unsqueeze(0), dim=-1)  # Shape: (H, W)
        
        # Step 4: Create a soft mask (clamp negative values to 0)
        soft_mask = normal_weight.clamp(0, 1)  # Clamp negative values to 0, and keep the range [0, 1]
        
        return soft_mask

@dataclass
class WarpAttentionPipelineConfig(InstructGS2GSPipelineConfig):
    _target: Type = field(default_factory=lambda: InstructGS2GSWarpAttentionPipeline)

    # igs2gs original parameters
    gs_steps: int = 1000
    model_type: Literal["ip2p", "controlnet-depth"] = "ip2p"
    default_sd_ckpt: str = "CompVis/stable-diffusion-v1-4"
    model_device: Optional[str] = "cuda:0"
    ddpm_inversion: bool = False # relevant for ip2p only
    ddpm_reference_image: Optional[str] = None # relevant for ddpm inversion only
    use_generator: bool = False

    reverse_prompt: Optional[str] = None # relevant for controlnet-depth only
    use_mask: Optional[bool] = False
    mask_prompt: Optional[str] = None

    seed: int = DEFAULT_SEED
    edit_idx: int = 32
    subset_size: int = 40
    crop_size: int = 512
    center_crop_image: int = 512

    only_cross_attention: bool = False
    only_self_attention: bool = False
    use_only_up_blocks: bool = True

    alpha: float = 0.9
    channels: tuple = (0, 1)
    res_to_use: tuple = (32, 64)
    num_heads: int = None # assign dynamically
    timestep_condition: callable = timestemp_condition_func
    timestep_decay: callable = timestep_decay_func

    # Evaluation
    evaluate: bool = False # evaluate the model
    src_prompt: Optional[str] = None # relevant for clip directional similarity
    target_prompt: Optional[str] = None # relevant for clip directional similarity
    novel_view_path: Optional[str] = None # relevant for clip consistency
