import torch
import logging
from tqdm import tqdm
from ovi.modules.fusion import FusionModel
from ovi.modules.mmaudio.features_utils import FeaturesUtils
from ovi.modules.t5 import T5EncoderModel
from ovi.modules.vae2_2 import Wan2_2_VAE
from ovi.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from ovi.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
import traceback
from omegaconf import OmegaConf
from ovi.utils.processing_utils import preprocess_image_tensor, snap_hw_to_multiple_of_32


DEFAULT_CONFIG = OmegaConf.load('ovi/configs/inference/inference_fusion.yaml')


class OviFusionEngine:
    def __init__(
        self, 
        backbone_model: FusionModel,
        vae_model_video: Wan2_2_VAE,
        vae_model_audio: FeaturesUtils,
        t5: T5EncoderModel,
        config=DEFAULT_CONFIG,
        device=0, 
        target_dtype=torch.bfloat16,
    ):
        # Load fusion model
        self.device = device
        self.target_dtype = target_dtype
        self.cpu_offload = config.get("cpu_offload", False)
        if self.cpu_offload:
            logging.info("CPU offloading is enabled. Initializing all models aside from VAEs on CPU")
        # Load VAEs
        self.vae_model_video = vae_model_video
        self.vae_model_audio = vae_model_audio

        # Load T5 text model
        self.text_model = t5
        if config.get("shard_text_model", False):
            raise NotImplementedError("Sharding text model is not implemented yet.")
        if self.cpu_offload:
            self.offload_to_cpu(self.text_model.model)
        self.model = backbone_model
        # Fixed attributes, non-configurable
        self.audio_latent_channel = backbone_model.audio_model.in_dim
        self.video_latent_channel = backbone_model.video_model.in_dim
        self.audio_latent_length = 157
        self.video_latent_length = 31

        logging.info(f"OVI Fusion Engine initialized, cpu_offload={self.cpu_offload}. GPU VRAM allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB, reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")

    @torch.inference_mode()
    def generate(self,
                    text_prompt, 
                    image=None,
                    video_frame_height_width=None,
                    seed=100,
                    solver_name="unipc",
                    sample_steps=50,
                    shift=5.0,
                    video_guidance_scale=5.0,
                    audio_guidance_scale=4.0,
                    slg_layer=9,
                    video_negative_prompt="",
                    audio_negative_prompt="",
                    **_
                ):

        params = {
            "Text Prompt": text_prompt,
            "Image Path": image if image else "None (T2V mode)",
            "Frame Height Width": video_frame_height_width,
            "Seed": seed,
            "Solver": solver_name,
            "Sample Steps": sample_steps,
            "Shift": shift,
            "Video Guidance Scale": video_guidance_scale,
            "Audio Guidance Scale": audio_guidance_scale,
            "SLG Layer": slg_layer,
            "Video Negative Prompt": video_negative_prompt,
            "Audio Negative Prompt": audio_negative_prompt,
        }

        pretty = "\n".join(f"{k:>24}: {v}" for k, v in params.items())
        logging.info("\n========== Generation Parameters ==========\n"
                    f"{pretty}\n"
                    "==========================================")
        scheduler_video, timesteps_video = self.get_scheduler_time_steps(
            sampling_steps=sample_steps,
            device=self.device,
            solver_name=solver_name,
            shift=shift
        )
        scheduler_audio, timesteps_audio = self.get_scheduler_time_steps(
            sampling_steps=sample_steps,
            device=self.device,
            solver_name=solver_name,
            shift=shift
        )

        is_t2v = image is None
        is_i2v = not is_t2v

        first_frame = None
        image = None
        if is_i2v:
            # Load first frame from path
            first_frame = preprocess_image_tensor(image, self.device, self.target_dtype)
        else:   
            assert video_frame_height_width is not None, f"If mode=t2v or t2i2v, video_frame_height_width must be provided."
            video_h, video_w = video_frame_height_width
            video_h, video_w = snap_hw_to_multiple_of_32(video_h, video_w, area = 720 * 720)
            video_latent_h, video_latent_w = video_h // 16, video_w // 16
            print(f"Pure T2V mode: calculated video latent size: {video_latent_h} x {video_latent_w}")

        
        if self.cpu_offload:
            self.text_model.model = self.text_model.model.to(self.device)
        text_embeddings = self.text_model([text_prompt, video_negative_prompt, audio_negative_prompt], self.text_model.device)
        text_embeddings = [emb.to(self.target_dtype).to(self.device) for emb in text_embeddings]

        if self.cpu_offload:
            self.offload_to_cpu(self.text_model.model)

        # Split embeddings
        text_embeddings_audio_pos = text_embeddings[0]
        text_embeddings_video_pos = text_embeddings[0] 

        text_embeddings_video_neg = text_embeddings[1]
        text_embeddings_audio_neg = text_embeddings[2]

        if is_i2v:
            if self.cpu_offload:
                self.vae_model_video.model = self.vae_model_video.model.to(
                    self.device
                )
            with torch.no_grad():
                latents_images = self.vae_model_video.wrapped_encode(first_frame[:, :, None]).to(self.target_dtype).squeeze(0) # c 1 h w 
            latents_images = latents_images.to(self.target_dtype)
            video_latent_h, video_latent_w = latents_images.shape[2], latents_images.shape[3]
            if self.cpu_offload:
                self.offload_to_cpu(self.vae_model_video.model)

        video_noise = torch.randn((self.video_latent_channel, self.video_latent_length, video_latent_h, video_latent_w), device=self.device, dtype=self.target_dtype, generator=torch.Generator(device=self.device).manual_seed(seed))  # c, f, h, w
        audio_noise = torch.randn((self.audio_latent_length, self.audio_latent_channel), device=self.device, dtype=self.target_dtype, generator=torch.Generator(device=self.device).manual_seed(seed))  # 1, l c -> l, c
        
        # Calculate sequence lengths from actual latents
        max_seq_len_audio = audio_noise.shape[0]  # L dimension from latents_audios shape [1, L, D]
        _patch_size_h, _patch_size_w = self.model.video_model.patch_size[1], self.model.video_model.patch_size[2]
        max_seq_len_video = video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3] // (_patch_size_h*_patch_size_w) # f * h * w from [1, c, f, h, w]
        
        # Sampling loop
        if self.cpu_offload:
            self.offload_to_cpu(self.vae_model_video.model)
            self.offload_to_cpu(self.vae_model_audio)
            self.model = self.model.to(self.device)
        with torch.amp.autocast('cuda', enabled=self.target_dtype != torch.float32, dtype=self.target_dtype):
            for i, (t_v, t_a) in tqdm(enumerate(zip(timesteps_video, timesteps_audio))):
                timestep_input = torch.full((1,), t_v, device=self.device)

                if is_i2v:
                    video_noise[:, :1] = latents_images

                # Positive (conditional) forward pass
                pos_forward_args = {
                    'audio_context': [text_embeddings_audio_pos],
                    'vid_context': [text_embeddings_video_pos],
                    'vid_seq_len': max_seq_len_video,
                    'audio_seq_len': max_seq_len_audio,
                    'first_frame_is_clean': is_i2v
                }

                pred_vid_pos, pred_audio_pos = self.model(
                    vid=[video_noise],
                    audio=[audio_noise],
                    t=timestep_input,
                    **pos_forward_args
                )
                
                # Negative (unconditional) forward pass  
                neg_forward_args = {
                    'audio_context': [text_embeddings_audio_neg],
                    'vid_context': [text_embeddings_video_neg],
                    'vid_seq_len': max_seq_len_video,
                    'audio_seq_len': max_seq_len_audio,
                    'first_frame_is_clean': is_i2v,
                    'slg_layer': slg_layer
                }
                
                pred_vid_neg, pred_audio_neg = self.model(
                    vid=[video_noise],
                    audio=[audio_noise],
                    t=timestep_input,
                    **neg_forward_args
                )

                # Apply classifier-free guidance
                pred_video_guided = pred_vid_neg[0] + video_guidance_scale * (pred_vid_pos[0] - pred_vid_neg[0])
                pred_audio_guided = pred_audio_neg[0] + audio_guidance_scale * (pred_audio_pos[0] - pred_audio_neg[0])

                # Update noise using scheduler
                video_noise = scheduler_video.step(
                    pred_video_guided.unsqueeze(0), t_v, video_noise.unsqueeze(0), return_dict=False
                )[0].squeeze(0)

                audio_noise = scheduler_audio.step(
                    pred_audio_guided.unsqueeze(0), t_a, audio_noise.unsqueeze(0), return_dict=False
                )[0].squeeze(0)

            if self.cpu_offload:
                self.offload_to_cpu(self.model)
                self.vae_model_video.model = self.vae_model_video.model.to(
                    self.device
                )
                self.vae_model_audio = self.vae_model_audio.to(self.device)

            if is_i2v:
                video_noise[:, :1] = latents_images

            # Decode audio
            audio_latents_for_vae = audio_noise.unsqueeze(0).transpose(1, 2)  # 1, c, l
            generated_audio = self.vae_model_audio.wrapped_decode(audio_latents_for_vae)
            generated_audio = generated_audio.squeeze().cpu().float().numpy()
            
            # Decode video  
            video_latents_for_vae = video_noise.unsqueeze(0)  # 1, c, f, h, w
            generated_video = self.vae_model_video.wrapped_decode(video_latents_for_vae)
            generated_video = generated_video.squeeze(0).cpu().float().numpy()  # c, f, h, w
            if self.cpu_offload:
                self.offload_to_cpu(self.vae_model_video.model)
                self.offload_to_cpu(self.vae_model_audio)
        return generated_video, generated_audio, image
            
    def offload_to_cpu(self, model):
        model = model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return model

    def get_scheduler_time_steps(self, sampling_steps, solver_name='unipc', device=0, shift=5.0):
        torch.manual_seed(4)

        if solver_name == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps

        elif solver_name == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift=shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=device,
                sigmas=sampling_sigmas)
            
        elif solver_name == 'euler':
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                shift=shift
            )
            timesteps, sampling_steps = retrieve_timesteps(
                sample_scheduler,
                sampling_steps,
                device=device,
            )
        
        else:
            raise NotImplementedError("Unsupported solver.")
        
        return sample_scheduler, timesteps
