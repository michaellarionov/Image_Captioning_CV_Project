import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import GPT2Tokenizer

class COCOCaptionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, max_length=50):
        self.image_dir = image_dir
        self.max_length = max_length

        # Load GPT-2 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load COCO annotations
        with open(annotation_file) as f:
            data = json.load(f)

        # Map image_id → filename
        self.id_to_file = {img['id']: img['file_name'] for img in data['images']}

        # Store all (image_id, caption) pairs
        self.samples = [(ann['image_id'], ann['caption']) for ann in data['annotations']]

        # Image preprocessing for ResNet
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, caption = self.samples[idx]
        filename = self.id_to_file[image_id]

        # Load and transform image
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Tokenize caption
        tokens = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return (
            image,
            tokens['input_ids'].squeeze(),
            tokens['attention_mask'].squeeze()
        )