from typing import Dict
import torch
from torch import Tensor
from jaxtyping import Float
from igs2gs.ip2p import InstructPix2Pix

import sys
sys.path.append(".")
from DDPM_inversion.ddm_inversion.inversion_utils import ip2p_inversion_forward_process, ip2p_inversion_reverse_process, reverse_step


class InstructPix2PixWarpAttention(InstructPix2Pix):
    def __init__(self, *args, **kwargs):
        seed = kwargs.pop("seed", 42)
        super().__init__(*args, **kwargs)
        self.last_parameters = {
            "seed": seed,
            "noise": None,
            "latents": {}
        }

    @torch.no_grad()
    def edit_image(
        self,
        text_embeddings: Float[Tensor, "N max_length embed_dim"],
        image: Float[Tensor, "BS 3 H W"],
        image_cond: Float[Tensor, "BS 3 H W"],
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 20,
        lower_bound: float = 0.70,
        upper_bound: float = 0.98,
        seed: int = None,
        use_last_timesteps: bool = False
    ) -> torch.Tensor:
        """Edit an image for Instruct-GS2GS using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
            seed: random seed
            use_last_timesteps: whether to use the last timesteps
        Returns:
            edited image
        """
        if seed is None:
            seed = self.last_parameters["seed"]
        torch.manual_seed(seed)

        if not use_last_timesteps:
            min_step = int(self.num_train_timesteps * lower_bound)
            max_step = int(self.num_train_timesteps * upper_bound)

            # select t, set multi-step diffusion
            T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
            
            self.scheduler.config.num_train_timesteps = T.item()
            self.scheduler.set_timesteps(diffusion_steps)

        with torch.no_grad():
            # prepare image and image_cond latents
            latents = self.imgs_to_latent(image)
            image_cond_latents = self.prepare_image_latents(image_cond)

        # add noise
        if use_last_timesteps:
            noise = self.last_parameters["noise"]
            assert noise is not None, "Noise must be provided if use_last_timesteps is True"
        else:
            noise = torch.randn_like(latents)
            self.last_parameters["noise"] = noise

        latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])  # type: ignore

        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        for i, t in enumerate(self.scheduler.timesteps):

            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # pred noise
                latent_model_input = torch.cat([latents] * 3)
                latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_image)
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )

            # get previous sample, continue loop
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents to get edited image
        with torch.no_grad():
            decoded_img = self.latents_to_img(latents)

        return decoded_img

    @torch.no_grad()
    def edit_image_ddpm(
        self,
        text_embeddings: Float[Tensor, "N max_length embed_dim"],
        image: Float[Tensor, "BS 3 H W"],
        image_cond: Float[Tensor, "BS 3 H W"],
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 20,
        lower_bound: float = 0.70,
        upper_bound: float = 0.98,
        seed: int = None,
        use_last_timesteps: bool = False,
        eta: float = 1
    ) -> torch.Tensor:
        """Edit an image for Instruct-GS2GS using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
            seed: random seed
            use_last_timesteps: whether to use the last timesteps
        Returns:
            edited image
        """
        if seed is None:
            seed = self.last_parameters["seed"]
        torch.manual_seed(seed)

        if not use_last_timesteps:
            min_step = int(self.num_train_timesteps * lower_bound)
            max_step = int(self.num_train_timesteps * upper_bound)

            # select t, set multi-step diffusion
            T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
            
            self.scheduler.config.num_train_timesteps = T.item()
            self.scheduler.set_timesteps(diffusion_steps)

        with torch.no_grad():
            # prepare image and image_cond latents
            latents = self.imgs_to_latent(image)
            image_cond_latents = self.prepare_image_latents(image_cond)

        # add noise
        if use_last_timesteps:
            noise = self.last_parameters["noise"]
            assert noise is not None, "Noise must be provided if use_last_timesteps is True"
        else:
            noise = torch.randn_like(latents)
            self.last_parameters["noise"] = noise

        latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])  # type: ignore
        zs = self.inversion_params["zs"].to(self.device).half()

        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        for i, t in enumerate(self.scheduler.timesteps):

            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # pred noise
                latent_model_input = torch.cat([latents] * 3)
                latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_image)
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )

            # get previous sample, continue loop
            z = zs[len(self.scheduler.timesteps) - i - 1]
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            # latents_reverse = reverse_step(self.pipe, noise_pred, t, latents, eta=eta, variance_noise=z).half()
            # latents = (latents + latents_reverse) / 2
            # latents = latents_reverse

        # decode latents to get edited image
        with torch.no_grad():
            decoded_img = self.latents_to_img(latents)

        return decoded_img

    @torch.no_grad()
    def ddpm_inversion_forward(self,
                               text_embeddings: Float[Tensor, "N max_length embed_dim"],
                               image: Float[Tensor, "BS 3 H W"],
                               image_cond: Float[Tensor, "BS 3 H W"],
                               guidance_scale: float = 7.5,
                               image_guidance_scale: float = 1.5,
                               diffusion_steps: int = 100,
                               lower_bound: float = 0.70,
                               upper_bound: float = 0.98,
                               eta: float = 1
    ) -> None:
        """
        Perform DDPM inversion for InstructPix2PixWarpAttention
        code is based on https://github.com/inbarhub/DDPM_inversion
        """

        # forward process
        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        
        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)

        latents = self.imgs_to_latent(image)
        image_cond_latents = self.prepare_image_latents(image_cond)

        _, zs, wts, noises = ip2p_inversion_forward_process(self.pipe, latents, image_cond_latents, text_embeddings,
                                                     etas=eta, guidance_scale=guidance_scale,
                                                     image_guidance_scale=image_guidance_scale, 
                                                     num_inference_steps=diffusion_steps)

        xt_noise = noises[self.pipe.scheduler.timesteps[0].item()]
        self.inversion_params = {
            "zs": zs.detach().cpu(),
            "wts": wts.detach().cpu(),
        }
        self.last_parameters["noise"] = xt_noise.detach().to(self.device).half()

    @torch.no_grad()
    def ddpm_inversion_backward(self,
                               text_embeddings: Float[Tensor, "N max_length embed_dim"],
                               image_cond: Float[Tensor, "BS 3 H W"],
                               guidance_scale: float = 7.5,
                               image_guidance_scale: float = 1.5,
                               diffusion_steps: int = 100,
                               skip: int = 36,
                               eta: float = 1
    ) -> torch.Tensor:
        """
        Perform backward DDPM inversion for InstructPix2PixWarpAttention
        """
        # backward process
        wts = self.inversion_params["wts"].to(self.device)
        zs = self.inversion_params["zs"].to(self.device)

        image_cond_latents = self.prepare_image_latents(image_cond)

        w0, _ = ip2p_inversion_reverse_process(self.pipe, xt=wts[diffusion_steps-skip][None], zs=zs[:(diffusion_steps-skip)],
                                               image_cond_x0=image_cond_latents, text_embeddings=text_embeddings, etas=eta,
                                               guidance_scale=guidance_scale, image_guidance_scale=image_guidance_scale)

        with torch.no_grad():
            decoded_img = self.latents_to_img(w0)
        
        return decoded_img
