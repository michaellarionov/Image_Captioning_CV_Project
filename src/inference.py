import torch
from PIL import Image
from torchvision import transforms
from transformers import GPT2Tokenizer
from src.model import VisionScript

def load_model(checkpoint_path, device):
    model = VisionScript(freeze_encoder=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def prepare_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image)

def caption_image(image_path, model, tokenizer, device):
    image = prepare_image(image_path).to(device)
    caption = model.generate_caption(image, tokenizer, device=device)
    return caption

if __name__ == '__main__':
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Load trained model
    checkpoint_path = '/content/drive/MyDrive/visionscript/models/checkpoint_epoch_5.pt'
    model = load_model(checkpoint_path, device)
    print("✅ Model loaded")

    # Caption a test image
    test_image = '/content/val2017/000000000139.jpg'
    caption = caption_image(test_image, model, tokenizer, device)
    print(f"Generated caption: {caption}")