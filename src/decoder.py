import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class GPT2Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pre-trained GPT-2 small (117M parameters)
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')

    def forward(self, visual_features, input_ids, attention_mask):
        # Get GPT-2 token embeddings
        token_embeddings = self.gpt2.transformer.wte(input_ids)  # [batch, seq_len, 768]

        # Prepend visual features as the first token
        inputs_embeds = torch.cat([visual_features, token_embeddings], dim=1)

        # Extend attention mask to cover the visual token
        visual_mask = torch.ones(attention_mask.shape[0], 1).to(attention_mask.device)
        extended_mask = torch.cat([visual_mask, attention_mask], dim=1)

        # Forward pass through GPT-2
        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask
        )

        return outputs.logits  # [batch, seq_len+1, 50257]