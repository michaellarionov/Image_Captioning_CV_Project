import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os
from src.dataset import COCOCaptionDataset
from src.model import VisionScript

def train(config):
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

    # Load dataset
    train_dataset = COCOCaptionDataset(
        image_dir=config['image_dir'],
        annotation_file=config['annotation_file']
    )

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=False
    )

    # Initialize model
    model = VisionScript(freeze_encoder=True).to(device)

    # Optimizer — only update parameters that require gradients
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate']
    )

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Training loop
    for epoch in range(config['epochs']):

        # Unfreeze encoder at Phase 2
        if epoch == 3:
            print("Unfreezing encoder for end-to-end training")
            for param in model.encoder.backbone.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config['learning_rate'] / 10
            )

        model.train()
        total_loss = 0

        for batch_idx, (images, input_ids, attention_mask) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass
            logits = model(images, input_ids[:, :-1], attention_mask[:, :-1])

            # Calculate loss
            loss = criterion(
                logits[:, 1:, :].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            # Log every 100 batches
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Average loss for epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")

        # Save checkpoint to Drive
        checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == '__main__':
    config = {
        'image_dir':        '/content/train2017',
        'annotation_file':  '/content/annotations/captions_train2017.json',
        'checkpoint_dir':   '/content/drive/MyDrive/visionscript/models',
        'batch_size':       8,
        'learning_rate':    3e-4,
        'epochs':           5
    }
    train(config)