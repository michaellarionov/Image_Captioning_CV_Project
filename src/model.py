import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from src.encoder import ResNetEncoder
from src.decoder import GPT2Decoder

class VisionScript(nn.Module):
    def __init__(self, freeze_encoder=True):
        super().__init__()
        self.encoder = ResNetEncoder(freeze=freeze_encoder)
        self.decoder = GPT2Decoder()

    def forward(self, images, input_ids, attention_mask):
        # Step 1 — encode image into visual features
        visual_features = self.encoder(images)

        # Step 2 — decode visual features into caption logits
        logits = self.decoder(visual_features, input_ids, attention_mask)

        return logits

    def generate_caption(self, image, tokenizer, max_length=50, device='cpu'):
        self.eval()
        with torch.no_grad():
            # Encode image
            visual_features = self.encoder(image.unsqueeze(0).to(device))

            # Start with BOS token
            input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)

            generated = []
            for _ in range(max_length):
                logits = self.decoder(
                    visual_features,
                    input_ids,
                    torch.ones_like(input_ids)
                )

                # Pick highest probability token
                next_token = logits[:, -1, :].argmax(dim=-1)

                # Stop if end of sequence token
                if next_token.item() == tokenizer.eos_token_id:
                    break

                generated.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        return tokenizer.decode(generated, skip_special_tokens=True)