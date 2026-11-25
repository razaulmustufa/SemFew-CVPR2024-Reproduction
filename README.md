# SemFew — Final Source-Code and Deployment Package

## Official project and model download links

- **GitHub repository (browse training, evaluation, and deployment source):** https://github.com/razaulmustufa/SemFew-CVPR2024-Reproduction
- **Complete project ZIP with trained model weights (GitHub Release v1.0.0):** https://github.com/razaulmustufa/SemFew-CVPR2024-Reproduction/releases/tag/v1.0.0

**Important:** The GitHub **Code** tab contains source code, experiment files, and documentation, but does not contain the two large ResNet-12 and SemAlign `.pth` weight files. Download the **complete ZIP** from the GitHub **Release assets** to obtain all deployment model weights. After extracting the complete ZIP, the Streamlit app can use `deployment/weights/` directly.


## Project summary
Paper: Hai Zhang et al., *Simple Semantic-Aided Few-Shot Learning* (CVPR 2024).
Course: Machine Learning, IBA Karachi, Fall 2026.
Dataset: CIFAR-FS (class-disjoint 64 train / 16 validation / 20 test classes; 600 images per class).
Method: frozen official pretrained ResNet-12 + official precomputed CLIP/GPT semantic embeddings + team-trained SemAlign network (50 epochs).
This is an **adapted reproduction**, not training the entire architecture from scratch.

## Test results (from executed notebook and supplied JSONs)
5-way 1-shot: 83.8156% +/- 0.6837 (95% CI), fusion k = 0.83; visual-only 73.3667%.
5-way 5-shot: 87.5533% +/- 0.5796 (95% CI), fusion k = 0.47; visual-only 86.5267%.
Additional experiment: 2-shot 84.7444% (k=0.68) and 10-shot 88.5333% (k=0.37).
Each result is computed from 600 randomly sampled episodes, with 15 query images per class in each five-way episode.
Note: a previous screenshot showed 82.82% 1-shot, but the **supplied executed notebook AND final JSON** both show 83.8156%. These files are the recorded final outputs. Re-running may produce different sampled episodes or results.

## Contents
- `notebooks/SemFew_Final_Training.ipynb` — executed Colab notebook: environment setup, cleaning, feature centers, SemAlign training, validation, test evaluation, shot-count extension.
- `results/*.json` and `results/*.csv` — team-generated experimental outputs.
- `deployment/app.py` — Streamlit UI and actual model inference integration.
- `deployment/requirements.txt` — local app dependencies.
- `deployment/model/` and `deployment/method/` — **directly reused** original network code.
- `deployment/semantic/` — **official precomputed** class-semantic representations.
- `official_SemFew_reference/SemFew-main/` — full upstream research code for reference and clear provenance.
- `artifacts/center_CIFAR-FS_resnet.pth` — computed class centers (also under official-reference for matching notebook path).
- `deployment_screenshots/` — example live runs; demonstration images are not benchmark accuracy measurements.

## Model files and package variants

The **complete ZIP linked above** contains the two `.pth` files required by Streamlit in `deployment/weights/`. The **GitHub Code tab/source-only download** omits those large model weights, so obtain the full ZIP from the Release when running the app:

1. `ResNet-CIFAR-FS.pth` — research authors' pretrained ResNet-12 backbone (reused, **NOT trained from scratch by the team**).
2. `semalign_training_checkpoint.pth` — team's SemAlign network trained for 50 epochs in Colab.
3. `artifacts/center_CIFAR-FS_resnet.pth` — CIFAR-FS visual centers calculated during the project (also copied into the reference-code location used by the notebook).

For submission, the publicly accessible GitHub Release above provides the complete project ZIP with both required checkpoint files. Include both GitHub links in the final PDF report's first page and in LMS submission comments. The complete ZIP can alternatively be uploaded to LMS directly if LMS file-size limits permit.

The dataset (`CIFAR-FS.zip`, around 197 MB) is **not** included in this ZIP: it was used for experiments and is required only to rerun dataset-based notebook training/evaluation, not for live Streamlit prediction. Keep an accessible dataset source/reference in your reproducibility notes.

## Run the local deployment (Windows / Anaconda Prompt)
1. Open Anaconda Prompt, move to the extracted `deployment/` folder.
2. `conda create -n semfewdemo python=3.11 -y` (only if the environment does not already exist)
3. `conda activate semfewdemo`
4. `python -m pip install -r requirements.txt` (only if needed)
5. `python -m streamlit run app.py`
6. Open http://localhost:8501 on the **same laptop** with the server running.

### Inputs and outputs
Select five distinct CIFAR-FS test classes; K in {1, 2, 5, 10}; upload exactly K labeled support images for each class, plus one distinct query image. Use 32x32 RGB PNG/JPEGs from CIFAR-FS/test for benchmark-like demonstrations. The app returns the predicted selected class and cosine similarity scores (not calibrated probabilities).

## Notebook / training reproduction
`notebooks/SemFew_Final_Training.ipynb` is the executed Colab record. It expects:
`/content/drive/MyDrive/SemFew_Final_Project/SemFew-main.zip`, `CIFAR-FS.zip`, and `ResNet-CIFAR-FS.pth`.
The training data is loaded from CIFAR-FS.zip in the notebook. It uses the original official `SemAlign` and `Res12` architecture, but custom notebook cells for centers/training/evaluation. Training was completed earlier; **DO NOT choose Run All** merely to demonstrate an already trained checkpoint, because that retrains/overwrites states. For live demo, use the local app and saved weights.
The notebook's experiments were performed in Google Colab on a Tesla T4; deployment is CPU-capable.

## Attribution and integrity
**Directly reused** from the original researchers (Zhang et al., CVPR 2024): ResNet-12 architecture and pretrained weights, SemAlign architecture code, episodic sampler, CIFAR transforms, precomputed CLIP/GPT semantics.
**Project-specific code/integration:** data verification/cleaning, training orchestration with a frozen backbone, class centers, checkpoint save/restore, validation-selected fusion, CIFAR-FS episodic evaluation, 1/2/5/10-shot extra experiment, visual-only comparator and Streamlit application/inference workflow. Any AI-assisted code should be disclosed in accordance with course policy.
**Not reproduced:** upstream ResNet-12 pretraining and the original semantic-description generation pipeline. Trained SemAlign uses final epoch 50 rather than best validation epoch. The learning-rate scheduler was adjusted after epoch 1. See report for limitations.

Official research reference: Zhang et al. (2024), https://github.com/zhangdoudou123/SemFew
The upstream files are provided for attribution and reproducibility; do not describe the reused architecture files as independently authored.
