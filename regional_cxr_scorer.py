import torchxrayvision as xrv
import skimage.io
import torch
import torchvision
import json
import pydicom
import numpy as np

# Initialize model globally to prevent bottlenecking inference calls
model = xrv.models.DenseNet(weights="densenet121-res224-all")

with open("state_disease_priors.json", "r") as f:
    REGIONAL_PRIORS = json.load(f)

def get_xrv_predictions(image_path: str):
    # 1. Handle DICOM vs Standard Image formats
    if image_path.lower().endswith(".dcm"):
        dcm = pydicom.dcmread(image_path)
        img = dcm.pixel_array.astype(np.float32)

        # Invert if photometric interpretation specifies MONOCHROME1 (0 is white)
        if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
            img = np.max(img) - img

        # Normalize 12-bit/16-bit dynamic range to 0-255 before XRV normalization
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255.0
        else:
            img = np.zeros_like(img)
    else:
        img = skimage.io.imread(image_path)

    # 2. Preprocess & standardize channels
    img = xrv.datasets.normalize(img, 255)

    if len(img.shape) > 2:
        img = img.mean(2)[None, ...]  # Convert RGB/RGBA to single-channel (1, H, W)
    else:
        img = img[None, ...]          # Add channel dimension for 2D grayscale (1, H, W)

    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])

    img = transform(img)
    img = torch.from_numpy(img).unsqueeze(0)

    # 3. Model Inference
    with torch.no_grad():
        outputs = model(img)[0]
    return dict(zip(model.pathologies, outputs.numpy()))

def calculate_regional_priority(image_path: str, patient_state: str, recent_travel: bool = False):
    likelihoods = get_xrv_predictions(image_path)

    # 4. Emergency Override Filter (High confidence acute findings or travel history)
    if recent_travel or max(likelihoods.values()) > 0.95:
        return dict(sorted(likelihoods.items(), key=lambda x: x[1], reverse=True))

    # 5. Fetch Regional Priors (Fall back to national baseline if state key is missing)
    state_data = REGIONAL_PRIORS.get(patient_state, REGIONAL_PRIORS.get("India", {}))

    # 6. Bayesian Weighting with 2.5% Laplace Smoothing Buffer
    posteriors = {}
    for disease, likelihood in likelihoods.items():
        prior = state_data.get(disease, 0.0)
        safe_prior = max(prior, 0.025)

        posteriors[disease] = likelihood * safe_prior

    return dict(sorted(posteriors.items(), key=lambda x: x[1], reverse=True))