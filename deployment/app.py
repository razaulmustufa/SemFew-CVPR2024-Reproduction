"""SemFew 5-way few-shot image classification deployment.

Research backbone + SemAlign network: Zhang et al., CVPR 2024.
Inference user interface / model-loading integration: project deployment.
Expected: ./weights/ResNet-CIFAR-FS.pth (official pretrained)
          ./weights/semalign_training_checkpoint.pth (team-trained)
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model.res12 import Res12
from method.SemAlign import SemAlign

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights"
SEMANTICS = ROOT / "semantic" / "cifar100_semantic_clip_gpt.pth"
PRETRAINED = WEIGHTS / "ResNet-CIFAR-FS.pth"
TRAINED = WEIGHTS / "semalign_training_checkpoint.pth"

# These 20 held-out CIFAR-FS test classes are used for the live demo.
# Precomputed CLIP features are available for CIFAR-FS categories, not arbitrary unseen names.
TEST_CLASSES = [
    "baby", "bed", "bicycle", "chimpanzee", "fox", "leopard", "man",
    "pickup_truck", "plain", "poppy", "rocket", "rose", "snail",
    "sweet_pepper", "table", "telephone", "wardrobe", "whale", "woman", "worm",
]

TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

st.set_page_config(page_title="SemFew — Few-Shot Classification", page_icon="🧠", layout="wide")
st.title("SemFew · Semantic-Aided Few-Shot Classification")
st.caption("CIFAR-FS · ResNet-12 + CLIP semantic features + trained SemAlign · 5-way classification")

@st.cache_resource(show_spinner="Loading pretrained ResNet-12 and trained SemAlign…")
def load_models():
    missing = [str(p.relative_to(ROOT)) for p in [PRETRAINED, TRAINED, SEMANTICS] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(missing))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vision = Res12(avg_pool=True, drop_block=False)
    backbone_ckpt = torch.load(PRETRAINED, map_location="cpu", weights_only=True)
    raw = backbone_ckpt["params"]
    state_keys = vision.state_dict()
    filtered = {key[8:]: val for key, val in raw.items() if key[8:] in state_keys}
    vision.load_state_dict(filtered, strict=True)
    vision = vision.to(device).eval().requires_grad_(False)

    alignment = SemAlign(v_size=640, s_size=512, h_size=4096, drop=0.0)
    trained_ckpt = torch.load(TRAINED, map_location="cpu", weights_only=True)
    alignment.load_state_dict(trained_ckpt["model_state_dict"], strict=True)
    if int(trained_ckpt.get("epoch", 0)) != 50:
        st.warning("Trained checkpoint is not labeled as epoch 50; check provenance.")
    alignment = alignment.to(device).eval().requires_grad_(False)

    embeddings_raw = torch.load(SEMANTICS, map_location="cpu", weights_only=True)["semantic_feature"]
    embeddings = {str(name): emb.float().to(device) for name, emb in embeddings_raw.items()}
    return vision, alignment, embeddings, device


def to_tensor(file):
    image = Image.open(BytesIO(file.getvalue())).convert("RGB")
    original_size = image.size
    if original_size != (32, 32):
        image = image.resize((32, 32), Image.Resampling.BICUBIC)
    return TEST_TRANSFORM(image), original_size

try:
    vision, alignment, embeddings, device = load_models()
except Exception as exc:
    st.error("Model files could not be loaded. Verify the weights folder and README instructions.")
    st.code(str(exc))
    st.stop()

st.success(f"Model ready • {device.type.upper()} inference • 50-epoch SemAlign checkpoint expected")

with st.sidebar:
    st.header("Episode settings")
    shot = st.selectbox("Support images per class (K)", options=[1, 2, 5, 10], index=0)
    k_defaults = {1: 0.83, 2: 0.68, 5: 0.47, 10: 0.37}
    fusion_k = k_defaults[shot]
    st.write(f"**Fusion factor k:** {fusion_k:.2f} (validation selected)")
    st.info("Select five different CIFAR-FS test classes. Upload the specified number of support images for each, then upload a query image.")
    selected = []
    for i in range(5):
        name = st.selectbox(
            f"Class {i + 1}",
            options=TEST_CLASSES,
            index=i,
            format_func=lambda n: n.replace("_", " ").title(),
            key=f"select_class_{i}",
        )
        selected.append(name)

if len(set(selected)) != 5:
    st.warning("Please select five different classes.")
    st.stop()
if any(name not in embeddings for name in selected):
    st.error("One or more selected classes have no precomputed semantic embedding.")
    st.stop()

st.subheader("1. Labeled support set")
support_files = {}
for name in selected:
    with st.expander(f"{name.replace('_', ' ').title()} · upload exactly {shot} image(s)", expanded=(shot == 1)):
        support_files[name] = st.file_uploader(
            f"{name.replace('_', ' ').title()} support images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key=f"support_{name}_{shot}",
        )

st.subheader("2. Query image")
query_file = st.file_uploader("Upload an image belonging to one of the five selected classes", type=["png", "jpg", "jpeg"], key="query_file")
if query_file is not None:
    st.image(query_file, caption="Query input", width=180)

if st.button("Predict class", type="primary", use_container_width=True):
    errors = [f"{name}: expected {shot}, got {len(support_files[name])}" for name in selected if len(support_files[name]) != shot]
    if errors:
        st.error("Please provide exactly K support images per class:\n" + "; ".join(errors))
    elif query_file is None:
        st.error("Upload a query image first.")
    else:
        try:
            # Episode convention used for testing: K support rows, one from each class per round.
            tensor_list = []
            changed_count = 0
            for i in range(shot):
                for name in selected:
                    tensor, original_size = to_tensor(support_files[name][i])
                    tensor_list.append(tensor)
                    changed_count += int(original_size != (32, 32))
            query_tensor, query_size = to_tensor(query_file)
            changed_count += int(query_size != (32, 32))
            inputs = torch.stack(tensor_list + [query_tensor]).to(device)
            with torch.inference_mode():
                visual_feats = vision(inputs)
                support = visual_feats[:-1]
                query = visual_feats[-1:]
                visual_prototypes = support.reshape(shot, 5, 640).mean(dim=0)
                semantic_support = torch.stack([embeddings[name] for _ in range(shot) for name in selected])
                reconstructed = alignment(semantic_support, support).reshape(shot, 5, 640).mean(dim=0)
                fused = fusion_k * reconstructed + (1 - fusion_k) * visual_prototypes
                similarities = (F.normalize(query, dim=-1) @ F.normalize(fused, dim=-1).T).flatten().cpu().tolist()
            scores = pd.DataFrame({"Class": [n.replace("_", " ").title() for n in selected], "Cosine similarity": similarities}).sort_values("Cosine similarity", ascending=False)
            best_label = scores.iloc[0]["Class"]
            st.subheader("3. Prediction")
            st.success(f"Predicted class: **{best_label}**")
            st.dataframe(scores, use_container_width=True, hide_index=True)
            st.bar_chart(scores.set_index("Class")["Cosine similarity"])
            st.caption("Cosine similarity scores are NOT calibrated probabilities or confidence percentages.")
            if changed_count:
                st.warning(f"{changed_count} uploaded image(s) were resized to 32×32. Real-world images may perform worse than CIFAR-FS benchmark images.")
        except Exception as exc:
            st.error("Inference failed; check uploaded images and model assets.")
            st.exception(exc)

st.divider()
st.caption("Research method and original network code: Hai Zhang et al., 'Simple Semantic-Aided Few-Shot Learning', CVPR 2024. Training and evaluation checkpoints from the project team's Colab workflow. This app uses official precomputed semantic embeddings and does not generate descriptions for arbitrary new categories.")
