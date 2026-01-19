import torch, os, argparse, accelerate, warnings
from diffsynth.core import TryOnDataset
from diffsynth.diffusion import *
from diffsynth.pipelines.wan_tryon import WanVTONPipeline
from diffsynth.core.loader.config import  ModelConfig
from safetensors.torch import load_file
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class WanVTONTrainingModule(DiffusionTrainingModule):
    def __init__(self, device, pipeline_config, args):
        super().__init__()
        self.pipe = WanVTONPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[
                ModelConfig("/data2/qinzijing/models/alibaba-pai/Wan2.2-Fun-5B-InP/Wan2.2_VAE.safetensors"),
                ModelConfig("/data2/qinzijing/models/tryon/wan_vton_5b_init.safetensors"),
                ModelConfig("/data2/qinzijing/models/alibaba-pai/Wan2.2-Fun-5B-InP/models_t5_umt5-xxl-enc-bf16.safetensors"),
            ],
            siglip_path="/data2/qinzijing/models/google/siglip2-large-patch16-384",
            dinov3_path="/data2/qinzijing/models/facebook/dinov3-vith16plus-pretrain-lvd1689m",
            tokenizer_config=ModelConfig("/data2/qinzijing/models/alibaba-pai/Wan2.2-Fun-5B-InP/google/umt5-xxl")
        )

        self.args = args
        self.prepare_models_for_training()

        if args.resume_from_checkpoint is not None and os.path.exists(args.resume_from_checkpoint):
            print(f"\n>>> 🚀 Resuming from checkpoint: {args.resume_from_checkpoint}")
            self.load_finetuned_weights(args.resume_from_checkpoint)

        self.max_timestep_boundary = 1.0
        self.min_timestep_boundary = 0.0
        self.use_gradient_checkpointing = True 
        

    def prepare_models_for_training(self):
        self.pipe.scheduler.training = True
        self.pipe.siglip_model.requires_grad_(False).eval()
        self.pipe.dino_model.requires_grad_(False).eval()

        lora_target_str = "self_attn.q,self_attn.k,self_attn.v,self_attn.o,cross_attn.q,cross_attn.o"

        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models=None, 
            lora_base_model="dit",
            lora_target_modules=lora_target_str,
            lora_rank=self.args.lora_rank, 
        )
        
        target_keywords = [
            "subject_image_proj_model",  
            "condition_embedding",       
            "cross_attn.k_ip",          
            "cross_attn.v_ip",          
            "cross_attn.k_cloth",      
            "cross_attn.v_cloth",      
            "cross_attn.norm",         
            "adapter_cloth"            
        ]

        unfrozen_count = 0
        for name, param in self.pipe.dit.named_parameters():
            for kw in target_keywords:
                if kw in name:
                    param.requires_grad = True
                    unfrozen_count += 1
                    break 

        
        total_params = 0
        trainable_params = 0
        
        for _, param in self.pipe.dit.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
        
        print(f"   - Total DiT Params:     {total_params / 1e9:.3f} B")
        print(f"   - Trainable DiT Params: {trainable_params / 1e6:.3f} M")
        print(f"   - Trainable Ratio:      {trainable_params / total_params * 100:.3f}%")
        print("="*40 + "\n")

    def load_finetuned_weights(self, checkpoint_path):
        print(f">>> Loading clean Stage 1 weights from {checkpoint_path} ...")
        from safetensors.torch import load_file
        
        # 1. 读取原始文件
        state_dict = load_file(checkpoint_path)
        
        # 2. 手动清洗与映射 (替代有 Bug 的父类方法)
        new_state_dict = {}
        for k, v in state_dict.items():
            # A. 清洗前缀 (去除 module. 和 pipe.dit.)
            clean_k = k.replace("module.", "").replace("pipe.dit.", "")
            
            # B. 映射 LoRA 名字 (如果原本就是 default 则不变，如果不是则补上)
            # 这一步是为了兼容不同版本的 DiffSynth 保存格式
            if "lora_A.weight" in clean_k and "default" not in clean_k:
                clean_k = clean_k.replace("lora_A.weight", "lora_A.default.weight")
            elif "lora_B.weight" in clean_k and "default" not in clean_k:
                clean_k = clean_k.replace("lora_B.weight", "lora_B.default.weight")
            
            # C. 🔥 关键：无论是不是 LoRA，都要保留！
            new_state_dict[clean_k] = v
            
        # 3. 加载到模型
        missing, unexpected = self.pipe.dit.load_state_dict(new_state_dict, strict=False)
        
        # 4. 验证 VTON 参数 (adapter_cloth)
        vton_missing = [k for k in missing if "adapter_cloth" in k]
        
        if len(vton_missing) > 0:
            print(f"❌ [Error] Stage 1 Load FAILED!")
            print(f"   Missing VTON keys: {vton_missing[:3]}")
            # 打印几个 new_state_dict 里的 key 看看是不是名字还有问题
            print(f"   Available keys sample: {list(new_state_dict.keys())[:3]}")
            raise RuntimeError("Critical: Failed to load Stage 1 VTON weights!")
        else:
            print(f"✅ [Success] Stage 1 Adapter & LoRA loaded successfully.")
            print(f"    - Loaded keys count: {len(new_state_dict)}")

    def forward(self, data):
        inputs_posi = {"prompt": data["prompt"]} 
        inputs_nega = {"negative_prompt": ""}
        
        inputs_shared = {
            "input_video": data["input"], 
            "cloth": data["cloth"],
            "agnostic": data["agnostic"],
            "densepose": data["densepose"],
            "mask": data["mask"],
            "height": data["input"][0].size[1], 
            "width": data["input"][0].size[0],  
            "num_frames": len(data["input"]),
            "cfg_scale": 1.0,
            "cfg_merge": False, 
            "tiled": True,
            "tile_size": (30, 52),   
            "tile_stride": (15, 26), 
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
        }

        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )
            
        loss = FlowMatchSFTLoss(self.pipe, **inputs_shared, **inputs_posi)
        
        return loss

def parse_args():
    parser = argparse.ArgumentParser(description="Train WanVTON")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--stage", type=str, default="image", choices=["image", "video"], help="Training stage")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to stage 1 checkpoint (LoRA + Unfrozen weights) to resume from.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    accelerator = accelerate.Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
    )
    
    print(f"\n🚀 Starting Training Stage: {args.stage.upper()}")
    print(f"   Resolution: {args.height}x{args.width}")
    
    train_frames = 1 if args.stage == "image" else args.num_frames
    
    dataset = TryOnDataset(
        dataset_root=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        stage=args.stage,
        split="train",
        height=args.height,
        width=args.width,
        num_frames=train_frames,
        repeat=args.dataset_repeat if hasattr(args, 'dataset_repeat') else 1
    )

    model = WanVTONTrainingModule(accelerator.device, pipeline_config={"stage": args.stage}, args=args)
    
    model_logger = ModelLogger(
        output_path=args.output_path, 
        remove_prefix_in_ckpt="pipe.dit.", 
    )

    launch_training_task(
        accelerator, 
        dataset, 
        model, 
        model_logger,
        args=args  
    )