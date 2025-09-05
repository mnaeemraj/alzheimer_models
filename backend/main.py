from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torchxrayvision as xrv
from PIL import Image
import io
import numpy as np
import onnxruntime as ort
import warnings
import logging

# model downloader utility

import gdown
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def get_model_path(file_id: str, filename: str):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)
    return path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aicenna-backend")

# =========================================================
# Utilities: flexible state_dict loader
# =========================================================
def is_state_dict(obj):
    return isinstance(obj, dict)

def try_load_state_dict_flexible(model: nn.Module, sd: dict):
    info = {"attempts": []}

    try:
        model.load_state_dict(sd)
        info["attempts"].append(("strict", None))
        return model, info
    except Exception as e:
        info["attempts"].append(("strict_failed", str(e)))

    try:
        res = model.load_state_dict(sd, strict=False)
        info["attempts"].append(("non_strict", {"missing_keys": res.missing_keys, "unexpected_keys": res.unexpected_keys}))
        if len(res.missing_keys) == 0:
            return model, info
    except Exception as e:
        info["attempts"].append(("non_strict_failed", str(e)))

    def transform_keys(sd_orig, add_prefix=None, remove_prefix=None):
        new = {}
        for k, v in sd_orig.items():
            new_k = k
            if remove_prefix and new_k.startswith(remove_prefix):
                new_k = new_k[len(remove_prefix):]
            if add_prefix:
                new_k = add_prefix + new_k
            new[new_k] = v
        return new

    prefixes = ["", "module.", "backbone.", "model.", "backbone.backbone.", "net.", "features."]
    tried = set()
    for add in prefixes:
        for rem in prefixes:
            key = f"add={add}|rem={rem}"
            if key in tried:
                continue
            tried.add(key)
            if add == "" and rem == "":
                continue
            try:
                sd_try = transform_keys(sd, add_prefix=add if add != "" else None, remove_prefix=rem if rem != "" else None)
                res = model.load_state_dict(sd_try, strict=False)
                info["attempts"].append((f"transform(add={add},rem={rem})", {"missing_keys": res.missing_keys, "unexpected_keys": res.unexpected_keys}))
                if len(res.missing_keys) == 0:
                    return model, info
            except Exception as e:
                info["attempts"].append((f"transform_failed(add={add},rem={rem})", str(e)))

    try:
        model_sd = model.state_dict()
        new_sd = {}
        model_keys = list(model_sd.keys())
        sd_keys = list(sd.keys())
        mapping = {}
        for sk in sd_keys:
            for mk in model_keys:
                if mk.endswith(sk) or sk.endswith(mk):
                    mapping[mk] = sd[sk]
                    break
        for mk in model_keys:
            if mk in mapping:
                new_sd[mk] = mapping[mk]
            else:
                new_sd[mk] = model_sd[mk]
        res = model.load_state_dict(new_sd, strict=False)
        info["attempts"].append(("suffix_match", {"missing_keys": res.missing_keys, "unexpected_keys": res.unexpected_keys}))
        if len(res.missing_keys) == 0:
            return model, info
    except Exception as e:
        info["attempts"].append(("suffix_match_failed", str(e)))

    return None, info

# =========================================================
# Softmax numpy
# =========================================================
def softmax_numpy(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# =========================================================
# Alzheimer loaders
# =========================================================
class AlzheimerEnsemble(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.efficientnet = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Identity()
        self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:-1])
        self.fc = nn.Linear(1408 + 4096, num_classes)

    def forward(self, x):
        f1 = self.efficientnet(x)
        f2 = self.vgg(x)
        fused = torch.cat([f1, f2], dim=1)
        return self.fc(fused)

def detect_backbone_from_state_dict(sd_keys):
    keys = list(sd_keys)
    if any("deit.encoder" in k or "transformer" in k for k in keys):
        return "vit"
    if any("features.denseblock" in k for k in keys):
        return "densenet"
    if any("features.0.weight" in k for k in keys):
        return "efficientnet"
    if any("classifier.6.weight" in k for k in keys):
        return "vgg"
    return "ensemble"

def build_model_for_backbone(backbone: str, num_classes=4):
    if backbone == "vit":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif backbone == "densenet":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif backbone == "efficientnet":
        model = models.efficientnet_b2(weights=None)
        model.classifier = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone == "vgg":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, num_classes)
    else:
        model = AlzheimerEnsemble(num_classes=num_classes)
    return model

def load_alzheimer_model_state_dict(weights_path: str, num_classes=4):
    loaded_obj = torch.load(weights_path, map_location="cpu")
    if isinstance(loaded_obj, nn.Module):
        loaded_obj.eval()
        return loaded_obj, {"loaded_as": "full_model"}
    if is_state_dict(loaded_obj):
        backbone = detect_backbone_from_state_dict(loaded_obj.keys())
        model = build_model_for_backbone(backbone, num_classes)
        model_try, info = try_load_state_dict_flexible(model, loaded_obj)
        if model_try is None:
            raise RuntimeError(f"Failed to load Alzheimer weights for backbone={backbone}. Attempts: {info}")
        model_try.eval()
        return model_try, {"loaded_as": "state_dict", "backbone": backbone, "info": info}
    raise RuntimeError("Unsupported Alzheimer weights format")

# =========================================================
# Chest X-ray custom (DenseNet121)
# =========================================================
class ChestXrayCustom(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.densenet121(weights=None)
        in_f = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_f, num_classes)

    def forward(self, x):
        return self.backbone(x)

def load_chest_custom(weights_path: str, num_classes=2):
    loaded = torch.load(weights_path, map_location="cpu")
    if isinstance(loaded, nn.Module):
        loaded.eval()
        return loaded, {"loaded_as": "full_model"}
    if is_state_dict(loaded):
        model = ChestXrayCustom(num_classes=num_classes)
        model_try, info = try_load_state_dict_flexible(model, loaded)
        if model_try is None:
            raise RuntimeError(f"Failed to load ChestX-ray state_dict. Attempts: {info}")
        model_try.eval()
        return model_try, {"loaded_as": "state_dict", "info": info}
    raise RuntimeError("Unsupported chest weights format")

# =========================================================
# Brain Tumor Torch (ResNet18 fallback)
# =========================================================
class BrainTumorTorch(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def load_brain_torch(weights_path: str, num_classes=4):
    loaded = torch.load(weights_path, map_location="cpu")
    if isinstance(loaded, nn.Module):
        loaded.eval()
        return loaded, {"loaded_as": "full_model"}
    if is_state_dict(loaded):
        model = BrainTumorTorch(num_classes=num_classes)
        model_try, info = try_load_state_dict_flexible(model, loaded)
        if model_try is None:
            raise RuntimeError(f"Failed to load Brain-Tumor state_dict. Attempts: {info}")
        model_try.eval()
        return model_try, {"loaded_as": "state_dict", "info": info}
    raise RuntimeError("Unsupported brain weights format")

# =========================================================
# FastAPI App
# =========================================================
app = FastAPI(title="AiCenna Medical Imaging Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# =========================================================
# Config: update paths to your weights
# =========================================================
available_models = {
    "ChestX-ray": {
        "DenseNet121 (All)": "densenet121-res224-all",   
        "DenseNet121 (NIH)": "densenet121-res224-nih",
        "DenseNet121 (CheXpert)": "densenet121-res224-chex",
        "DenseNet (Custom)": {"id": "1Nb6Pa8cXZakEI6ybcWFrhZP4baff1Atr", "filename": "chest_xray.pt"},
    },
    "Brain_Tumor": {
        "Version 1 (ONNX)": {"id": "1uv6lOSa4WfJdHFRfqBaJGrOoqX3KFRtH", "filename": "best.onnx"}, 
        "Version 2 (Torch)": {"id": "1DD2CKrWxEiyjMsIUp3YhXNxgcLK9AzPn", "filename": "brain_tumor.pt"},
    },
    "Alzheimer": {
         "Ensemble Model": {"id": "1LK3T0m3JPbZA_OlWxdmBUcgCZ5AO5_c4", "filename": "alzheimer_model.pt"},
    },
}

brain_classes = ["glioma_tumor", "meningioma_tumor", "pituitary_tumor", "no_tumor"]
alzheimer_classes = ["MildDemented", "ModerateDemented", "VeryMildDemented", "NonDemented"]
chest_custom_classes = ["abnormal", "normal"]

# =========================================================
# Preprocessing
# =========================================================
img224_tfms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

img640_tfms = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
])

def preprocess_xray_for_xrv(pil_img: Image.Image):
    arr = np.array(pil_img.convert("L")).astype(np.float32)
    arr = xrv.datasets.normalize(arr, 255)
    ten = torch.from_numpy(arr)[None, None, ...]
    ten = torch.nn.functional.interpolate(ten, size=(224, 224), mode="bilinear", align_corners=False)
    return ten

def preprocess_generic_224(pil_img: Image.Image):
    return img224_tfms(pil_img.convert("RGB")).unsqueeze(0)

def preprocess_generic_640(pil_img: Image.Image):
    return img640_tfms(pil_img.convert("RGB")).unsqueeze(0)

# =========================================================
# Load models at startup
# =========================================================
models_dict = {}
for disease, versions in available_models.items():
    models_dict[disease] = {}
    for version, meta in versions.items():
        try:
            if isinstance(meta, dict):  # Google Drive download
                file_id = meta["id"]
                filename = meta["filename"]
                weights = get_model_path(file_id, filename)
            else:  # torchxrayvision built-in weights
                weights = meta  
            if disease == "ChestX-ray":
                if weights.endswith(".pt"):
                    model_obj, info = load_chest_custom(weights, num_classes=len(chest_custom_classes))
                    logger.info(f"Loaded ChestX-ray/{version}: {info}")
                    models_dict[disease][version] = {"model": model_obj, "targets": chest_custom_classes, "type": "torch", "load_info": info}
                else:
                    model_obj = xrv.models.DenseNet(weights=weights)
                    model_obj.eval()
                    models_dict[disease][version] = {"model": model_obj, "targets": model_obj.pathologies, "type": "xrv", "load_info": {"loaded_as": "xrv"}}

            elif disease == "Brain_Tumor":
                if weights.endswith(".onnx"):
                    session = ort.InferenceSession(weights, providers=["CPUExecutionProvider"])
                    models_dict[disease][version] = {"model": session, "targets": brain_classes, "type": "onnx", "load_info": {"loaded_as": "onnx"}}
                else:
                    model_obj, info = load_brain_torch(weights, num_classes=len(brain_classes))
                    logger.info(f"Loaded Brain_Tumor/{version}: {info}")
                    models_dict[disease][version] = {"model": model_obj, "targets": brain_classes, "type": "torch", "load_info": info}

            elif disease == "Alzheimer":
                model_obj, info = load_alzheimer_model_state_dict(weights, num_classes=len(alzheimer_classes))
                logger.info(f"Loaded Alzheimer/{version}: {info}")
                models_dict[disease][version] = {"model": model_obj, "targets": alzheimer_classes, "type": "torch", "load_info": info}

        except Exception as e:
            warnings.warn(f"[LOAD WARNING] {disease}/{version} -> {e}")
            logger.exception(e)
            models_dict[disease][version] = {"model": None, "targets": None, "type": "unavailable", "error": str(e)}

# =========================================================
# Routes
# =========================================================
@app.get("/models")
async def get_models():
    out = {}
    for disease, versions in models_dict.items():
        out[disease] = {}
        for version, info in versions.items():
            out[disease][version] = {
                "loaded": info["model"] is not None,
                "type": info.get("type"),
                "targets": info.get("targets"),
                "error": info.get("error", None),
                "load_info": info.get("load_info", None),
            }
    return {"available_models": out}

@app.post("/predict/{disease}/{version}")
async def predict(
    disease: str,
    version: str,
    file: UploadFile = File(...),
    target: str = Query(None),
):
    if disease not in models_dict or version not in models_dict[disease]:
        return {"error": f"Model not found: {disease}/{version}"}

    info = models_dict[disease][version]
    model = info["model"]
    targets = info["targets"]
    mtype = info["type"]

    if model is None or targets is None:
        return {"error": f"{disease}/{version} is not loaded", "details": info.get("error")}

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    if disease == "ChestX-ray":
        if mtype == "xrv":
            x = preprocess_xray_for_xrv(image)
            with torch.no_grad():
                logits = model(x)
            probs = torch.sigmoid(logits[0]).cpu().numpy().tolist()
        else:
            x = preprocess_generic_224(image)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()

    elif disease == "Brain_Tumor":
        if mtype == "onnx":
            x = preprocess_generic_640(image).numpy().astype(np.float32)
            inp_name = model.get_inputs()[0].name
            out = model.run(None, {inp_name: x})
            logits = out[0][0]
            probs = softmax_numpy(logits).tolist()
        else:
            x = preprocess_generic_640(image)
            with torch.no_grad():
                logits = model(x)[0]
                probs = torch.softmax(logits, dim=0).cpu().numpy().tolist()

    elif disease == "Alzheimer":
        x = preprocess_generic_224(image)
        with torch.no_grad():
            logits = model(x)[0]
            probs = torch.softmax(logits, dim=0).cpu().numpy().tolist()

    else:
        return {"error": "Unsupported model type."}
    

    if target in targets:
        idx = targets.index(target)
        return {
            "message": f"Prediction :  { probs[idx]:.3f}",
            "target": target,
            "probability": float(probs[idx]),
        }
    else:
        return {
            "message": f"Prediction successful ({disease})",
            "probabilities": dict(zip(targets, map(float, probs))),
        }