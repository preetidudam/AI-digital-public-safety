"""
Transfer-Learning CNN Model for Indian Currency Verification (Genuine vs Counterfeit).
Uses MobileNetV3 architecture optimized for high accuracy and fast CPU/GPU inference.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Union
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

from config import (
    CURRENCY_MODEL_PATH,
    CURRENCY_IMAGE_SIZE,
    CURRENCY_CLASSES,
    CURRENCY_IDX_TO_CLASS
)
from src.data.currency_loader import get_currency_transforms


def build_currency_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Builds MobileNetV3-Small transfer learning model with modified classification head.
    MobileNetV3 is chosen for high accuracy on micro-textures and low inference latency.
    """
    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
    else:
        model = models.mobilenet_v3_small(weights=None)

    # Replace final classification projection
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


class CurrencyVerifier:
    """
    Inference wrapper for Indian Currency Verification.
    Follows lazy-loading pattern and provides honest, non-exaggerated findings.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None, device: Optional[str] = None):
        if model_path is None:
            self.model_path = Path(CURRENCY_MODEL_PATH)
        else:
            self.model_path = Path(model_path)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        _, self.transform = get_currency_transforms(CURRENCY_IMAGE_SIZE)

        self.model = None
        self.is_ready = False

        if self.model_path.exists():
            try:
                print(f"Loading Currency Verifier model from: {self.model_path} ({self.device})...")
                model = build_currency_model(num_classes=len(CURRENCY_CLASSES), pretrained=False)
                state_dict = torch.load(str(self.model_path), map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.model = model
                self.is_ready = True
                print("Currency Verifier model loaded successfully.")
            except Exception as e:
                print(f"Warning: Failed to load currency model from {self.model_path}: {e}")
                self.is_ready = False
        else:
            print(f"Currency model checkpoint not found at: {self.model_path}")
            self.is_ready = False

    def verify(self, image_input: Any) -> Dict[str, Any]:
        """
        Verify an uploaded currency note image.
        Accepts PIL.Image, filepath string, or numpy array.
        Returns verdict string, confidence percentage, and analysis advisory.
        """
        if image_input is None:
            return {
                "verdict": "Please upload a currency note image.",
                "confidence": 0.0,
                "is_genuine": None,
                "analysis": "No image was provided."
            }

        # Convert input to PIL Image
        if isinstance(image_input, (str, Path)):
            if not os.path.exists(str(image_input)):
                return {
                    "verdict": "File not found.",
                    "confidence": 0.0,
                    "is_genuine": None,
                    "analysis": f"Could not find image at path: {image_input}"
                }
            pil_image = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
        else:
            # Handle numpy arrays from Gradio or OpenCV
            try:
                pil_image = Image.fromarray(image_input).convert("RGB")
            except Exception as e:
                return {
                    "verdict": "Invalid image format.",
                    "confidence": 0.0,
                    "is_genuine": None,
                    "analysis": f"Failed to convert image to RGB: {str(e)}"
                }

        # Handle un-trained model state gracefully
        if not self.is_ready or self.model is None:
            return {
                "verdict": "⚠️ Model Not Trained Yet",
                "confidence": 0.0,
                "is_genuine": None,
                "analysis": (
                    "The Currency Verifier CNN model has not been trained yet.\n\n"
                    "Next Steps to Activate:\n"
                    "1. Place genuine and counterfeit note images in 'data/currency_dataset/train/'\n"
                    "2. Run the training script: python scripts/train_currency_verifier.py\n"
                    "3. The saved weights ('models/currency_verifier.pt') will automatically activate in this tab."
                )
            }

        # Preprocess and execute inference
        try:
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = int(torch.argmax(probs).item())
                confidence = float(probs[pred_idx].item()) * 100.0

            predicted_class = CURRENCY_IDX_TO_CLASS.get(pred_idx, "Unknown")
            is_genuine = (predicted_class == "Genuine")

            if is_genuine:
                verdict = f"✅ GENUINE (Likely Authentic Note)"
                analysis = (
                    f"Result: Classified as GENUINE with {confidence:.1f}% model confidence.\n\n"
                    f"Model Findings:\n"
                    f"- Visual surface patterns, color tones, and printing layout match genuine currency benchmarks.\n\n"
                    f"⚠️ Important Verification Notice:\n"
                    f"- This CNN model performs visual surface pattern classification from 2D photos.\n"
                    f"- Physical security features (intaglio tactile ink relief, paper substrate feel, watermark transparency, "
                    f"and UV fluorescence) cannot be verified by standard camera images alone.\n"
                    f"- For official or doubtful currency verification, consult an authorized bank branch."
                )
            else:
                verdict = f"⚠️ POTENTIAL COUNTERFEIT DETECTED"
                analysis = (
                    f"Result: Flagged as POTENTIAL COUNTERFEIT with {confidence:.1f}% model confidence.\n\n"
                    f"Model Findings:\n"
                    f"- Discrepancies detected in visual pattern consistency, edge details, or surface color fidelity compared to genuine currency.\n\n"
                    f"Recommended Actions (RBI Guidelines):\n"
                    f"- Do not accept or circulate suspected fake banknotes.\n"
                    f"- Inspect physical security elements manually (windowed security thread, crispness of Reserve Bank text).\n"
                    f"- Report suspicious notes to the nearest commercial bank branch or local authorities."
                )

            return {
                "verdict": verdict,
                "confidence": round(confidence, 1),
                "is_genuine": is_genuine,
                "analysis": analysis
            }

        except Exception as e:
            return {
                "verdict": "Inference Error",
                "confidence": 0.0,
                "is_genuine": None,
                "analysis": f"Error during model inference: {str(e)}"
            }
