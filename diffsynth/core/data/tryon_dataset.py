import os
import json
import torch
from .operators import * 
class TryOnDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,          
        metadata_path,          
        stage="image",         
        split="train",          
        height=480,
        width=832,
        num_frames=49,          
        repeat=1,
        load_from_cache=False
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.stage = stage
        self.split = split     
        self.repeat = repeat
        
        self.num_frames = 1 if stage == "image" else num_frames
        self.height = height
        self.width = width
        self.load_from_cache = load_from_cache
        self.image_op = (
            LoadImage() 
            >> ImageCropAndResize(height, width) 
            >> ToList() 
        )

        self.video_op = LoadVideo(
            num_frames=self.num_frames,
            frame_processor=ImageCropAndResize(height, width) 
        )
        
        self.data = self._load_and_filter_metadata(metadata_path)
        print(f"[TryOnDataset] Stage: {stage.upper()} | Split: {split.upper()} | Loaded {len(self.data)} samples.")

    def _get_file_path(self, base_dir, folder_name, filename):
        """
        根据 Stage 和 Folder Name 强制修正文件后缀
        """
        file_root = os.path.splitext(filename)[0]
        
        if self.stage == "video":
            if folder_name == "cloth":
                ext = ".jpg"
            else:
                ext = ".mp4"
        else:
            ext = ".jpg"
        return os.path.join(base_dir, folder_name, file_root + ext)

    def _load_and_filter_metadata(self, metadata_path):
        processed_data = []
        
        with open(metadata_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                
                # --- A. 过滤 Stage ---
                data_type = item.get("type", "image")
                if self.stage == "image" and data_type != "image": continue
                if self.stage == "video" and data_type != "video": continue

                # --- B. 构建基础路径 ---
                # root / category / split (train/test)
                category = item["category"] # e.g., "dresscode"
                filename = item["x"]        # e.g., "048589_0.jpg"
                cloth_name = item["cloth"]  # e.g., "048589_0.jpg"
                
                base_dir = os.path.join(self.dataset_root, category, self.split)
                
                # --- C. 组装所有文件的绝对路径 ---
                # 1. 模特主图/视频 (GT)
                input_path = self._get_file_path(base_dir, "image", filename)
                
                # 2. 服装 (Cloth)
                cloth_path = self._get_file_path(base_dir, "cloth", cloth_name)
                
                # 3. 骨架 (DensePose)
                densepose_path = self._get_file_path(base_dir, "densepose", filename)
                
                # 4. 去衣底图 (Agnostic)
                agnostic_path = self._get_file_path(base_dir, "agnostic", filename)
                
                # 5. 遮罩 (Mask)
                mask_path = self._get_file_path(base_dir, "agnostic_mask", filename)
                processed_data.append({
                    "input_path": input_path,
                    "cloth_path": cloth_path,
                    "densepose_path": densepose_path,
                    "agnostic_path": agnostic_path,
                    "mask_path": mask_path,
                    "prompt": item.get("caption", ""),
                })
        
        return processed_data

    def __len__(self):
        return len(self.data) * self.repeat

    def __getitem__(self, index):
        item = self.data[index % len(self.data)]
        ret = {
            "prompt": item["prompt"]
        }

        try:
            # === 1. 加载 Reference Image (Cloth) ===
            # 无论什么阶段，衣服都是静态图，使用 image_op
            ret["cloth"] = self.image_op(item["cloth_path"])

            # === 2. 加载其余数据 (根据 Stage 切换 Operator) ===
            # 选择当前阶段对应的 Operator
            op = self.video_op if self.stage == "video" else self.image_op
            ret["input"] = op(item["input_path"])
            ret["densepose"] = op(item["densepose_path"])
            ret["agnostic"] = op(item["agnostic_path"])
            ret["mask"] = op(item["mask_path"])

        except Exception as e:
            print(f"[Dataset Error] Loading {item['input_path']} failed: {e}")
            return self.__getitem__(0)

        return ret