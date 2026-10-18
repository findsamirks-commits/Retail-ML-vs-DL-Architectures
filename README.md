# Retail-ML-vs-DL-Architectures
# ⚖️ Classical ML vs. Deep Learning: Retail Architectures

A hands-on comparative benchmark contrasting Classical Machine Learning and Deep Learning paradigms across retail enterprise use cases.

This repository explores the architectural boundary between structured tabular modeling requiring manual domain-driven feature engineering and unstructured high-dimensional tensor processing via convolutional feature extraction.

---

## 📊 Core Architectural Comparison

| Dimension | Classical Machine Learning (XGBoost) | Deep Learning (PyTorch CNN) |
| :--- | :--- | :--- |
| **Data Modality** | Structured, tabular rows & columns (CSVs, SQL tables) | Unstructured high-dimensional tensors (images, audio, text) |
| **Input Shape** | 2D Matrix: `(n_samples, n_features)` | 4D Tensor: `(Batch, Channels, Height, Width)` |
| **Feature Extraction** | Manual engineering required (e.g., `Days_of_Cover`) | Automatic feature representation via learned convolutional filters |
| **Hardware Requirement** | CPU-efficient, lightweight execution | GPU-accelerated tensor math (NVIDIA T4 / CUDA cores) |
| **Optimization Target** | Tree splits minimizing objective loss (e.g., logloss) | Gradient descent updating weights via backpropagation (`loss.backward()`) |
| **Model Interpretability** | Transparent feature importance rankings | Black-box latent feature maps and filter weights |
| **Best Retail Use Case** | Inventory replenishment, markdown classification, churn | Visual catalog QA, defect detection, packaging validation |

---

## 🛠️ Implemented Notebooks

### 1. `notebooks/FNV-Markdown-Classifier-ML.ipynb`
* **Objective**: Predict perishable inventory SKUs requiring immediate markdowns to prevent waste.
* **Technique**: Manual feature engineering of `Days_of_Cover` ($\frac{\text{Inventory}}{\text{Avg Daily Sales}}$) versus `Days_to_Expiry`.
* **Model**: XGBoost Classifier with One-Hot categorical encoding.
* **Result**: Achieved 99% precision/recall with transparent feature importance ranking.

### 2. `notebooks/Spice-Catalog-Vision-DL.ipynb`
* **Objective**: Visual validation of spice catalog imagery to identify packaging defects or blurriness.
* **Technique**: Direct ingestion of raw RGB tensors (`3 x 64 x 64`) without manual feature extraction.
* **Model**: 2-stage Convolutional Neural Network (`Conv2d`, `ReLU`, `MaxPool2d`, `Linear`) built with PyTorch and accelerated on NVIDIA T4 GPU (`cuda`).
* **Introspection**: Extracted and visualized the 16 learned spatial filter kernels and resulting activation feature maps.

---

## 🚀 Environment Setup

```bash
# Clone the repository
git clone [https://github.com/findsamirks-commits/Retail-ML-vs-DL-Architectures.git](https://github.com/findsamirks-commits/Retail-ML-vs-DL-Architectures.git)
cd Retail-ML-vs-DL-Architectures

# Install dependencies
pip install torch torchvision xgboost scikit-learn pandas numpy matplotlib seaborn
