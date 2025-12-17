# app.py
# Gr

import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gradio as gr

# ----------------------
# Model definition
# ----------------------
class CNN_LSTM_TUMOR(nn.Module):
    def __init__(self, params):
        super(CNN_LSTM_TUMOR, self).__init__()

        Cin, Hin, Win = params["shape_in"]
        num_fc1 = params["num_fc1"]       # LSTM hidden size
        num_fc2 = params["num_fc2"]
        num_classes = params["num_classes"]
        self.dropout_rate = params.get("dropout_rate", 0.0)

        # CNN FEATURE EXTRACTOR
        self.conv1 = nn.Conv2d(Cin, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)   # Downsample by /2

        # After 2 poolings: Hin -> Hin/4 ; Win -> Win/4
        H2, W2 = Hin // 4, Win // 4
        # feature size per time step (channels * width)
        self.feature_size = 64 * W2

        # LSTM LAYER (treat rows as sequence)
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=num_fc1,
            batch_first=True
        )

        # FC LAYERS
        self.fc2 = nn.Linear(num_fc1, num_fc2)
        self.fc3 = nn.Linear(num_fc2, num_classes)

    def forward(self, X):
        # X: (batch, 3, H, W)
        X = F.relu(self.conv1(X))
        X = self.pool(F.relu(self.conv2(X)))  # after two conv+pool => (batch, 64, H/4, W/4)
        X = self.pool(X)                      # safety: if earlier code expected second pool, it's applied

        # Permute and reshape to (batch, seq_len=H_reduced, features)
        # Current shape (batch, C=64, H_reduced, W_reduced)
        X = X.permute(0, 2, 1, 3)  # (batch, H_reduced, C, W_reduced)
        X = X.reshape(X.size(0), X.size(1), -1)  # (batch, seq_len=H_reduced, features=64*W_reduced)

        # LSTM expects (batch, seq_len, input_size)
        lstm_out, (hn, cn) = self.lstm(X)  # lstm_out: (batch, seq_len, hidden)
        # Use the last time-step's output
        last_out = lstm_out[:, -1, :]      # (batch, hidden)

        x = F.relu(self.fc2(last_out))
        logits = self.fc3(x)
        return logits

# ----------------------
# Config / instantiate
# ----------------------
params_model = {
    "shape_in": (3, 256, 256),
    "num_fc1": 100,
    "num_fc2": 50,
    "dropout_rate": 0.25,
    "num_classes": 2
}

CLASS_NAMES = ["Brain Tumor", "Healthy"]  
WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS", "./weights.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM_TUMOR(params_model).to(device)

# Try load weights if available
if os.path.exists(WEIGHTS_PATH):
    try:
        state = torch.load(WEIGHTS_PATH, map_location=device)
        # Support both saved state_dict and full model checkpoint
        if isinstance(state, dict) and not any(k in state for k in ["epoch","optimizer"]):
            # assume it's a state_dict
            model.load_state_dict(state)
        elif isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            # fallback: attempt to load directly (may error if not state_dict)
            try:
                model.load_state_dict(state)
            except Exception:
                # some checkpoints save full model; attempt to load attributes
                print("Warning: could not safely load checkpoint as state_dict. Proceeding without loading.")
        print(f"Loaded model weights from {WEIGHTS_PATH}")
    except Exception as e:
        print(f"Failed to load weights from {WEIGHTS_PATH}: {e}")
else:
    print(f"Weight file not found at {WEIGHTS_PATH}. The app will run with uninitialized weights.")

model.eval()

# ----------------------
# Preprocessing
# ----------------------
# NOTE: If you normalized differently during training, adjust mean/std accordingly.
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225])   # ImageNet std
])

def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """Convert PIL image to model input tensor (batch dimension included)."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    t = transform(pil_image)  # (3,H,W)
    t = t.unsqueeze(0)        # (1,3,H,W)
    return t.to(device)

# ----------------------
# Prediction wrapper
# ----------------------
def predict(pil_image: Image.Image) -> Dict:
    """
    Input: PIL image
    Output: dict with label probabilities and predicted label
    """
    x = preprocess_image(pil_image)
    with torch.no_grad():
        logits = model(x)  # (1, num_classes)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Prepare human-friendly output
    probs_dict = {cls: float(round(float(p), 4)) for cls, p in zip(CLASS_NAMES, probs)}
    top_idx = int(probs.argmax())
    predicted = CLASS_NAMES[top_idx]
    return {"label": predicted, "confidence": probs_dict}

# ----------------------
# Gradio UI
# ----------------------
title = "Brain Tumor Detector (CNN-LSTM)"
description = """
Upload an image and the model will predict whether it contains a brain tumor.
Assumptions: model expects 256x256 RGB images. If your training normalization or labels differ,
please update `transform` and `CLASS_NAMES` in app.py accordingly.
"""

with gr.Blocks() as demo:
    gr.Markdown(f"## {title}")
    gr.Markdown(description)

    with gr.Row():
        inp = gr.Image(type="pil", label="Upload image")
        out_label = gr.Label(num_top_classes=2, label="Predicted probabilities")

    btn = gr.Button("Predict")
    status = gr.Textbox(label="Result", interactive=False)

    def run_predict(image):
        if image is None:
            return None, "No image provided."
        res = predict(image)
        # Label output wants a dict {label:confidence}
        return res["confidence"], f"{res['label']} (highest probability)"

    btn.click(run_predict, inputs=[inp], outputs=[out_label, status])

# For Hugging Face Spaces, host on 0.0.0.0 and allow web access.
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
