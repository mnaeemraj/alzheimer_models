from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchxrayvision as xrv
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import onnxruntime as ort

app = FastAPI(title="AiCenna Medical Imaging Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================
# Load Models
# ====================
available_models = {
    "ChestX-ray": {
        "DenseNet121 (All)": "densenet121-res224-all",
        "DenseNet121 (NIH)": "densenet121-res224-nih",
        "DenseNet121 (CheXpert)": "densenet121-res224-chex",
    },
    "Brain Tumor": {
        "Version 1": "best.onnx"
    }
}

models = {}
for disease, versions in available_models.items():
    models[disease] = {}
    for version, weights in versions.items():
        if disease == "ChestX-ray":
            model = xrv.models.DenseNet(weights=weights)
            model.eval()
            models[disease][version] = {
                "model": model,
                "targets": model.pathologies,
                "type": "torch"
            }
        elif disease == "Brain Tumor":
            session = ort.InferenceSession(weights, providers=["CPUExecutionProvider"])
            models[disease][version] = {
                "model": session,
                "targets": ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"],
                "type": "onnx"
            }

# Brain tumor dataset classes
brain_classes = ["glioma_tumor", "meningioma_tumor", "pituitary_tumor", "no_tumor"]

# ====================
# Preprocessing
# ====================
def preprocess_xray(image: Image.Image):
    img = np.array(image.convert("L")).astype(np.float32)
    img = xrv.datasets.normalize(img, 255)
    img = torch.from_numpy(img).unsqueeze(0)
    transform = transforms.Resize((224, 224))
    img = transform(img)
    return img.unsqueeze(0)

def preprocess_brain(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    img = transform(image.convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
    return img

# ====================
# Routes
# ====================
@app.get("/models")
async def get_models():
    """Return available models with targets."""
    available = {}
    for disease, versions in models.items():
        available[disease] = {}
        for version, info in versions.items():
            available[disease][version] = {"targets": info["targets"]}
    return {"available_models": available}


@app.post("/predict/{disease}/{version}")
async def predict(
    disease: str,
    version: str,
    file: UploadFile = File(...),
    target: str = Query(...),
):
    if disease not in models or version not in models[disease]:
        return {"error": "Model not found."}

    model_info = models[disease][version]
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    if model_info["type"] == "torch":  # Chest X-ray
        model = model_info["model"]
        img_tensor = preprocess_xray(image)
        with torch.no_grad():
            outputs = model(img_tensor)

        if target not in model.pathologies:
            return {"error": f"Target must be one of: {model.pathologies}"}

        idx = model.pathologies.index(target)
        prob_A = torch.sigmoid(outputs[0, idx]).item()
        prob_B = 1 - prob_A
        return {
            "message": "Prediction successful (Chest X-ray)",
            "target": target,
            "probabilities": {"A": prob_A, "B": prob_B},
        }

    elif model_info["type"] == "onnx":  # Brain Tumor
        if target not in model_info["targets"]:
            return {"error": f"Target must be one of: {model_info['targets']}"}

        session = model_info["model"]
        img_tensor = preprocess_brain(image)
        ort_inputs = {session.get_inputs()[0].name: img_tensor}
        ort_outs = session.run(None, ort_inputs)
        logits = ort_outs[0][0]

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        prob_target = float(probs[brain_classes.index(target)])
        prob_notumor = float(probs[brain_classes.index("no_tumor")])

        return {
            "message": "Prediction successful (Brain Tumor)",
            "target": target,
            "probabilities": {
                target: prob_target,
                "no_tumor": prob_notumor
            },
        }
