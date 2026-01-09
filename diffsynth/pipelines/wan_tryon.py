import torch
from transformers import SiglipVisionModel, SiglipImageProcessor, AutoModel, AutoImageProcessor
from einops import rearrange
import torch.nn.functional as F
from ..diffusion import FlowMatchScheduler
from ..diffusion.base_pipeline import PipelineUnit
from ..models.wan_video_dit import WanModel
from ..diffusion.base_pipeline import BasePipeline
from .wan_video import (
                        WanVideoUnit_ShapeChecker, 
                        WanVideoUnit_NoiseInitializer, 
                        WanVideoUnit_PromptEmbedder, 
                        WanVideoUnit_InputVideoEmbedder,
                        WanVideoUnit_UnifiedSequenceParallel,
                        WanVideoUnit_TeaCache,
                        WanVideoUnit_CfgMerger,
                        TeaCache
                        )
from ..models.wan_video_text_encoder import HuggingfaceTokenizer

from typing import List, Optional, Tuple, Literal
from PIL import Image
from tqdm import tqdm

class WanVTONPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, 
            torch_dtype=torch_dtype,
            height_division_factor=16, 
            width_division_factor=16, 
            time_division_factor=4, 
            time_division_remainder=1
        )
        
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.dit = None 
        self.siglip_model = None
        self.siglip_processor = None
        self.dino_model = None
        self.dino_processor = None
        self.in_iteration_models = ("dit",)

        # 3. 【关键】重写 Units 列表
        # 只保留 VTON 训练真正需要的
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_VTONInputs(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
        ]
        
        self.model_fn = model_fn_wan_vton
        
        self.post_units = []

    @staticmethod
    def from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[],
        tokenizer_config=None,
        siglip_path="/data2/qinzijing/models/google/siglip2-large-patch16-384", 
        dinov3_path="/data2/qinzijing/models/facebook/dinov3-vith16plus-pretrain-lvd1689m",
        **kwargs
    ):
        pipe = WanVTONPipeline(device=device, torch_dtype=torch_dtype)
        
        
        model_pool = pipe.download_and_load_models(model_configs)
        
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        pipe.dit = model_pool.fetch_model("wan_tryon_dit") 
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        
        pipe.siglip_processor = SiglipImageProcessor.from_pretrained(siglip_path)
        pipe.siglip_model = SiglipVisionModel.from_pretrained(siglip_path).to(dtype=torch_dtype).eval()

        pipe.dino_processor = AutoImageProcessor.from_pretrained(dinov3_path)
        pipe.dino_processor.crop_size = dict(height=384, width=384)
        pipe.dino_processor.size = dict(height=384, width=384)
        pipe.dino_model = AutoModel.from_pretrained(dinov3_path).to(dtype=torch_dtype).eval()

        if tokenizer_config:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean='whitespace')
            
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        cloth: Image.Image,                  
        agnostic: List[Image.Image],         
        densepose: List[Image.Image],         
        mask: List[Image.Image],    
        
        prompt: str,
        negative_prompt: Optional[str] = "",
        
        input_video: Optional[List[Image.Image]] = None,          

        height: int = 480,
        width: int = 832,
        num_frames: int = 16, # 如果是单图生成，设为 1

        seed: Optional[int] = None,
        cfg_scale: float = 5.0,
        cfg_merge: Optional[bool] = False,
        num_inference_steps: int = 50,
        denoising_strength: float = 1.0, # 1.0 = 全量生成, <1.0 = 图生图/视频生视频
        sigma_shift: float = 5.0,
        tiled: bool = True,
        tile_size: Tuple[int, int] = (30, 52),
        tile_stride: Tuple[int, int] = (15, 26),
        tea_cache_l1_thresh: Optional[float] = None,
        progress_bar_cmd=tqdm,
        output_type: Literal["quantized", "floatpoint"] = "quantized", 
    ):
        # 1. 设置 Scheduler
        # shift 参数对于视频生成非常重要，Wan 默认通常是 5.0
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # 2. 准备输入字典
        # 这些字典会被传递给 Pipeline Units 进行处理
        
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, 
            "num_inference_steps": num_inference_steps,
        }
        
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, 
            "num_inference_steps": num_inference_steps,
        }
        
        # Inputs Shared: 包含所有 VTON 需要的图像/视频条件
        # 注意 key 的名字必须和 WanVideoUnit_VTONInputs 里的 input_params 对应
        inputs_shared = {
            # VTON 核心条件
            "cloth": cloth,
            "agnostic": agnostic,
            "densepose": densepose,
            "mask": mask,
            
            # 基础参数
            "height": height, 
            "width": width, 
            "num_frames": num_frames,
            "seed": seed, 
            "rand_device": self.device, # 通常是 cuda
            
            # 控制参数
            "cfg_scale": cfg_scale,
            "cfg_merge": cfg_merge, 
            "denoising_strength": denoising_strength,
            "sigma_shift": sigma_shift,
            
            # 显存优化参数
            "tiled": tiled, 
            "tile_size": tile_size, 
            "tile_stride": tile_stride,
            
            # 兼容性占位 (防止 Unit 报错)
            "input_video": input_video, # 推理时我们不需要 GT 视频，只需要生成
        }

        # 3. 运行 Pipeline Units (预处理 & 特征提取)
        # 这一步会调用 VTONInputs Unit，执行 SigLIP/DINO 提取特征，并对 Agnostic/Densepose 进行 VAE 编码
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        # 4. 加载 DiT 模型到显存
        self.load_models_to_device(self.in_iteration_models)
        
        # 准备模型字典传给 model_fn
        # 这里特别处理了我们手动加载的 SigLIP/DINO，虽然特征提取已经在 Unit 里做完了
        # 但为了接口一致性，还是通过 models 传进去 (尽管 model_fn 可能不直接用它们)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            noise_pred_posi = self.model_fn(
                **models, 
                **inputs_shared, 
                **inputs_posi, 
                timestep=timestep
            )
            
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            inputs_shared["latents"] = self.scheduler.step(
                noise_pred, 
                self.scheduler.timesteps[progress_id], 
                inputs_shared["latents"]
            )
            
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
            
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        elif output_type == "floatpoint":
            pass
        self.load_models_to_device([])
        return video
    
class WanVideoUnit_VTONInputs(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("cloth", "agnostic", "densepose", "mask", "tiled", "tile_size", "tile_stride", "height", "width"),
            output_params=("cloth_latents", "agnostic_latents", "densepose_latents", "mask_input", "ip_hidden_states"),
            onload_model_names=("vae")
        )

    def _prepare_vton_mask(self, mask_tensor, target_h, target_w):
        mask_padded = torch.cat(
            [
                torch.repeat_interleave(mask_tensor[:, :, 0:1], repeats=4, dim=2), 
                mask_tensor[:, :, 1:]
            ], dim=2
        )
        b, c, t, h, w = mask_padded.shape
        if t % 4 != 0:
            t_new = (t // 4) * 4
            mask_padded = mask_padded[:, :, :t_new, :, :]
            b, c, t, h, w = mask_padded.shape
        mask_view = mask_padded.view(b, c, t // 4, 4, h, w)
        mask_folded = mask_view.permute(0, 3, 1, 2, 4, 5).reshape(b, 4 * c, t // 4, h, w)
        mask_final = F.interpolate(
            mask_folded, 
            size=(mask_folded.shape[2], target_h, target_w), 
            mode="nearest" # Mask 建议用 nearest 保持二值边缘
        )        
        return mask_final

    def _get_visual_features(self, pipe, cloth_tensor):
        """
        完全复用 Reference Code 的逻辑进行特征提取
        """
        device = pipe.device
        dtype = pipe.torch_dtype
        
        # 将输入 tensor [-1, 1] 转换为 [0, 1]
        image_01 = cloth_tensor * 0.5 + 0.5

        # --- 内部辅助函数: 对应 encode_image_emb 中的 process_with_processor ---
        def process_with_processor(images, processor):
            # images: (B, C, H, W) range [0, 1]
            if isinstance(images, torch.Tensor):
                # Processor 通常期望 float32 输入进行归一化
                images = images.to(torch.float32)
            
            return processor(
                images=images,
                return_tensors="pt",
                do_resize=False,       # 我们自己做 resize (interpolate)
                do_rescale=False,      # 输入已经是 [0,1]，不需要 /255
                do_normalize=True,     # 只做 ImageNet Normalize
                data_format="channels_first", 
                input_data_format="channels_first"
            ).pixel_values.to(device=device, dtype=dtype)

        # --- 内部辅助函数: SigLIP 特征提取 ---
        def encode_siglip(pixel_values):
            # 将模型临时移到 device (如果之前在 CPU)
            pipe.siglip_model.to(device)
            with torch.no_grad():
                res = pipe.siglip_model(pixel_values, output_hidden_states=True)
                
                # Deep features: last_hidden_state
                embeds = res.last_hidden_state
                
                # Shallow features: layers [5, 11, 23]
                # SigLIP hidden_states 包含 embedding layer 输出，所以 index 要注意
                # Reference code 用的是 [5, 11, 23]，我们直接照搬
                shallow = torch.cat([res.hidden_states[i] for i in [5, 11, 23]], dim=1)
            return embeds, shallow

        # --- 内部辅助函数: DINOv3 特征提取 ---
        def encode_dino(pixel_values):
            pipe.dino_model.to(device)
            with torch.no_grad():
                res = pipe.dino_model(pixel_values, output_hidden_states=True)
                
                # DINOv3 通常有 register tokens 或 class token，Reference code 做了切片 [:, 5:]
                embeds = res.last_hidden_state[:, 5:]
                
                # Shallow features: layers [7, 15, 31], 切片 [:, 5:]
                shallow = torch.cat([res.hidden_states[i][:, 5:] for i in [7, 15, 31]], dim=1)
            return embeds, shallow

        # === Step B: Low Res (384x384) ===
        image_low_res = F.interpolate(image_01, size=(384, 384), mode='bicubic', align_corners=False, antialias=True)
        
        siglip_low_input = process_with_processor(image_low_res, pipe.siglip_processor)
        dino_low_input = process_with_processor(image_low_res, pipe.dino_processor)
        
        siglip_embeds_low, siglip_shallow_low = encode_siglip(siglip_low_input)
        dinov3_embeds_low, dinov3_shallow_low = encode_dino(dino_low_input)
        
        image_embeds_low_res_deep = torch.cat([siglip_embeds_low, dinov3_embeds_low], dim=2)
        image_embeds_low_res_shallow = torch.cat([siglip_shallow_low, dinov3_shallow_low], dim=2)

        # === Step C: High Res (768x768) & Crop ===
        image_high_res = F.interpolate(image_01, size=(768, 768), mode='bicubic', align_corners=False, antialias=True)
        
        # 切分成 4 个 384x384
        crops = [
            image_high_res[:, :, 0:384, 0:384],      # Top-Left
            image_high_res[:, :, 0:384, 384:768],    # Top-Right
            image_high_res[:, :, 384:768, 0:384],    # Bottom-Left
            image_high_res[:, :, 384:768, 384:768],  # Bottom-Right
        ]
        image_crops = torch.stack(crops, dim=1) # (B, 4, C, H, W)
        b, n, c, h, w = image_crops.shape
        image_crops_flat = rearrange(image_crops, 'b n c h w -> (b n) c h w')
        
        # === Step E: High Res Features ===
        siglip_input_high = process_with_processor(image_crops_flat, pipe.siglip_processor)
        dino_input_high = process_with_processor(image_crops_flat, pipe.dino_processor)
        
        siglip_embeds_high, _ = encode_siglip(siglip_input_high)
        dinov3_embeds_high, _ = encode_dino(dino_input_high)
        
        # Reshape back: (B*4, L, D) -> (B, 4*L, D)
        siglip_high_res_deep = rearrange(siglip_embeds_high, '(b n) l d -> b (n l) d', n=n)
        dinov3_high_res_deep = rearrange(dinov3_embeds_high, '(b n) l d -> b (n l) d', n=n)
        
        image_embeds_high_res_deep = torch.cat([siglip_high_res_deep, dinov3_high_res_deep], dim=2)

        return {
            "image_embeds_low_res_shallow": image_embeds_low_res_shallow,
            "image_embeds_low_res_deep": image_embeds_low_res_deep,
            "image_embeds_high_res_deep": image_embeds_high_res_deep
        }

    def process(self, pipe, cloth, agnostic, densepose, mask, tiled, tile_size, tile_stride, height, width):
        if any(x is None for x in [cloth, agnostic, densepose, mask]):
            return {}

        pipe.load_models_to_device(self.onload_model_names)

        def encode_condition(image_list_or_tensor):
            video_tensor = pipe.preprocess_video(image_list_or_tensor) # B C T H W
            latents = pipe.vae.encode(
                video_tensor, 
                device=pipe.device, 
                tiled=tiled, 
                tile_size=tile_size, 
                tile_stride=tile_stride
            )
            return latents.to(dtype=pipe.torch_dtype, device=pipe.device)

        cloth_latents = encode_condition(cloth)
        agnostic_latents = encode_condition(agnostic)
        densepose_latents = encode_condition(densepose)

        mask_tensor = pipe.preprocess_video(mask, min_value=0, max_value=1).to(device=pipe.device, dtype=pipe.torch_dtype)
        if mask_tensor.shape[1] == 3:
            mask_tensor = mask_tensor[:, 0:1, :, :, :]
        scale_factor = pipe.vae.upsampling_factor if hasattr(pipe.vae, 'upsampling_factor') else 8
        mask_input = self._prepare_vton_mask(mask_tensor, height // scale_factor, width // scale_factor)

        cloth_tensor = pipe.preprocess_video(cloth, min_value=-1, max_value=1)
        cloth_tensor = cloth_tensor[:, :, 0, :, :] 
        ip_hidden_states = self._get_visual_features(pipe, cloth_tensor)

        return {
            "cloth_latents": cloth_latents,
            "agnostic_latents": agnostic_latents,
            "densepose_latents": densepose_latents,
            "mask_input": mask_input,
            "ip_hidden_states": ip_hidden_states    
        }
    
def model_fn_wan_vton(
    dit,
    latents,          
    timestep,
    context,          
    
    # 来自 Unit 的输入
    cloth_latents,    
    agnostic_latents, 
    densepose_latents,
    mask_input,       
    ip_hidden_states,
    
    # 其他参数
    tea_cache=None,
    use_gradient_checkpointing=False,
    **kwargs
):

    
    x_input = torch.cat([
        latents, 
        mask_input, 
        agnostic_latents, 
    ], dim=1)

    output = dit(
        x=x_input,
        cloth_latents=cloth_latents,
        densepose_latents=densepose_latents, 
        ip_hidden_states=ip_hidden_states,
        timestep=timestep,
        context=context,
        seq_len=latents.shape[2], 
        use_gradient_checkpointing=use_gradient_checkpointing
    )

    return output