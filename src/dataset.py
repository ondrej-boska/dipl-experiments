import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class OMRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Recursively find all staves and their corresponding JSON files
        stave_dirs = list(self.root_dir.rglob("Staves"))
        self.json_files = []
        for stave_dir in stave_dirs:
            self.json_files.extend(stave_dir.rglob("coco-object-detection.json"))

    def __len__(self):
        return len(self.json_files)
        
    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        
        # Locate the corresponding PNG in the same folder
        img_path = next(json_path.parent.glob("*.jpg"))
        
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
            
        image = Image.open(img_path).convert("RGB")
        
        boxes = []
        labels = []
        
        # COCO bounding box format is [x_min, y_min, width, height]
        for ann in coco_data.get('annotations', []):
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])
            
        # Convert to PyTorch tensors
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        
        if self.transform:
            # Apply transforms (ensure your transforms handle bounding box updates)
            image, target = self.transform(image, target)
            
        return image, target


if __name__ == "__main__":
    # Example usage
    dataset = OMRDataset(root_dir="OmniOMR.Small")
    print(f"Number of samples in the dataset: {len(dataset)}")
    
    image, target = dataset[0]
    print(f"Image size: {image.size}, Target: {target}")