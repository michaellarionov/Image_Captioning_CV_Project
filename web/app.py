from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import sys
import os
from transformers import GPT2Tokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import caption_image, load_model

app = Flask(__name__)
CORS(app)

# Load model once at startup
device = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available()
                       else 'cpu')

print(f"Loading model on {device}...")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models', 'visionscript_final.pt'
)

if not os.path.exists(MODEL_PATH):
    print(f"\n❌ Model file not found at: {MODEL_PATH}")
    print("Please download the model from the link in the README and place it in the models/ directory.")
    print("Expected file: models/visionscript_final.pt\n")
    sys.exit(1)

model = load_model(MODEL_PATH, device)
print("✅ Model loaded and ready")

@app.route('/caption', methods=['POST'])
def caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Save to temp file
    temp_path = '/tmp/upload_image.jpg'
    image_file.save(temp_path)

    # Generate caption
    result = caption_image(temp_path, model, tokenizer, device)

    # Clean up temp file
    os.remove(temp_path)

    return jsonify({'caption': result})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': str(device)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)