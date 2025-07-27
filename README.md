# explainable-wifi-sensing

This repository provides the implementation of the paper:  
**"Beyond the Black Box: Explainability for Multi-Activity Multi-environment WiFi Sensing"**

The code covers data preparation, model training, evaluation, and explainability techniques (e.g., Grad-CAM and SHAP) applied to WiFi sensing tasks involving multiple activities and environments.

---

## Environment

To run the scripts and notebooks, you need a Python environment with the following dependencies:

- `gdown` — used to download the dataset from Google Drive.
- `tensorflow`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `shap`

The file [`environment.yml`](environment.yml) was generated using Conda and contains all the required dependencies.  
To create the environment, run:

```bash
conda env create -f environment.yml
conda activate explainable-wifi-sensing 
```

---

## Scripts

All shell scripts are located in the [`scripts/`](scripts/) folder and are **numbered in dependency order**, from data download to explainability visualization.  
However, **not all scripts need to be executed every time** — each script is small, self-contained, and easy to understand.

### Script Overview

- `01-download-data.sh`  
  Download the dataset using `gdown`.

- `02-prepare-data.sh`  
  Preprocess and organize the dataset.

- `03-recreate-source-study-model.sh`  
  Recreate the baseline model from the original study.

- `04-add_DataGeneratorUnified.sh`  
  Patch to enable multiple monitor and environment data handling.

- `05-train_multimonitor_models.sh`  
  Train models on multiple monitor and environments.

- `06-evaluate_models.sh`  
  Evaluate trained models.

- `07-plot_tsne_dataset.sh`  
  Visualize raw dataset using t-SNE.

- `08-plot_gradcam_shap.sh`  
  Generate and save Grad-CAM and SHAP aggregated explanations.

---

## Results

All outputs from the experiments are saved under the `results/` directory. This includes both raw data and visualizations used in the paper.

- The `csvs/` folder contains evaluation metrics in tabular format.
- The `figures/` folder stores plots generated for the paper.
- The `gradcam_shap/` folder includes `.npz` files with serialized Grad-CAM and SHAP outputs.

You can load these `.npz` files in Python using:

```python
import numpy as np

data = np.load("results/gradcam_shap/selected_file.npz")
gradcam = data["gradcam_heatmaps"]
shap = data["shap_gradients_heatmaps"]
```

## Citation

If you find this work useful, please consider citing the original paper (citation to be added soon).

## License

This project is licensed under the GNU General Public License v3.0.