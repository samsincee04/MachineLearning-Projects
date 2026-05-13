# ECGR-4116

## Midterm Project for AI for Biomedical Applications

This project investigates whether **super-resolution generated retinal OCT images** can improve binary classification performance. The project follows the midterm prompt by first building a baseline transfer-learning classifier (**Model A**) on **128x128** images, then training an **SRGAN** to generate **128x128** images from **32x32** inputs, and finally using those generated images to train a second classifier (**Model B**). The final comparison is based on **Accuracy, F1 score, and AUC**.

---

## Project Objective

The goals of this project are to:

1. Train a binary classifier (**Model A**) on the retinal OCT dataset using transfer learning.
2. Train an SRGAN to generate **128x128** images from **32x32** low-resolution inputs.
3. Use the SRGAN-generated images to train a second classifier (**Model B**).
4. Compare Model A and Model B using **Accuracy, F1 score, and AUC**.

---

## Dataset

**Dataset used:** Retinal OCT images (Srinivasan)

For this implementation, the binary classes used were:

- `NORMAL`
- `DME`

The original dataset contained TIFF images organized into `train` and `test` folders. For this project, the TIFF images were converted to PNG and reorganized into a clean binary dataset for reproducible processing.

---

## Final Workflow

### 1. Data Preparation

- Mounted Google Drive in Colab
- Converted TIFF OCT images to PNG
- Selected the binary classes `NORMAL` and `DME`
- Created a **70% training / 30% testing** split
- Split part of the training set again into validation data
- Applied normalization and augmentation
- Displayed transformed sample images in the notebook

### 2. Model A: Baseline Classifier

- Used **ResNet50** with ImageNet pretrained weights
- Input images were resized to **128x128**
- Converted grayscale OCT images to RGB so they could be used with ResNet50
- Trained a baseline binary classifier and evaluated it with:
  - Accuracy
  - F1 score
  - AUC

### 3. SRGAN

- Prepared paired training data:
  - **Low-resolution input:** 32x32
  - **High-resolution target:** 128x128
- Built and trained a simplified SRGAN-style generator and discriminator
- Saved checkpoints regularly
- Saved generated sample images across training
- Compared:
  - LR input
  - Generated SR output
  - Real HR target

### 4. Model B: Classifier on Generated Images

- Used the trained SRGAN generator to create a new image set
- Trained **Model B** on the generated **128x128** images
- Evaluated Model B using:
  - Accuracy
  - F1 score
  - AUC

### 5. Final Comparison

- Compared Model A and Model B directly to determine whether SRGAN-generated images improved downstream classification performance

---

## Project Structure

```text
ECGR-4116-Midterm/
├── Midterm_proj_4116.ipynb
├── README.md
├── models/
│   ├── model_a_best.keras
│   ├── model_a_finetuned_best.keras
│   ├── model_b_best.keras
│   ├── model_b_finetuned_best.keras
│   └── srgan_checkpoints/
├── outputs/
│   ├── model_a_training_curves.png
│   ├── model_a_confusion_matrix.png
│   ├── model_b_confusion_matrix.png
│   ├── model_a_vs_model_b_barplot.png
│   ├── srgan_training_curves.png
│   ├── model_a_metrics.txt
│   ├── model_b_metrics.txt
│   └── final_model_comparison.txt
├── srgan_samples/
└── Dataset_png/
```

## Repository Notes

Some trained model files were too large for standard GitHub upload limits and were not included in this repository:

- `model_a_finetuned_best.keras`
- `model_b_finetuned_best.keras`

The baseline model files, notebook, outputs, and other project artifacts are included so that the workflow and results can still be reviewed and reproduced.

## Environment / Requirements

This project was developed in **Google Colab** using Python and TensorFlow.

### Main libraries used

- `tensorflow`
- `keras`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `Pillow`
- `tifffile`

Install missing packages if needed:

```bash
pip install tensorflow numpy matplotlib scikit-learn pillow tifffile
```

## Reproducibility Steps

To reproduce similar results, follow these steps in order.

### 1. Prepare the dataset

Place the **Srinivasan** dataset in Google Drive with this structure:

```text
Srinivasan/
├── train/
│   ├── NORMAL/
│   ├── DME/
│   └── DRUSEN/
└── test/
    ├── NORMAL/
    ├── DME/
    └── DRUSEN/
```

### 2. Open the notebook in Google Colab

Open the main notebook: `Midterm_proj_4116.ipynb`

### 3. Mount Google Drive

Mount Google Drive in Colab so the notebook can access:

- The source dataset
- Generated outputs
- Saved models
- Project folders

### 4. Set project paths

Create a project root in Google Drive, for example:

```text
MyDrive/
├── Srinivasan/
└── Midterm_OCT_Project/
```

Set the notebook paths so that:

- `SRC_ROOT` points to the Srinivasan dataset folder
- `PROJECT_ROOT` points to `Midterm_OCT_Project`

### 5. Convert TIFF images to PNG

Run the TIFF-to-PNG conversion cells to:

- Read OCT TIFF images
- Keep only `NORMAL` and `DME`
- Save converted PNG images into the project folder

### 6. Build Model A

Run the Model A cells to:

- Create the 70/30 train-test split
- Create validation data from the training split
- Resize images to 128x128
- Normalize images to [0, 1]
- Convert grayscale images to RGB
- Apply augmentation
- Train the ResNet50 baseline
- Fine-tune the upper layers
- Save metrics, plots, and confusion matrix

### 7. Prepare SRGAN training pairs

Run the SRGAN data preparation cells to:

- Create 128x128 high-resolution targets
- Create 32x32 low-resolution inputs from those targets
- Build (LR, HR) training pairs
- Visualize LR vs HR examples

### 8. Train the SRGAN

Run the SRGAN training cells to:

- Train the generator and discriminator
- Save checkpoints after epochs
- Save generated image samples during training
- Inspect visual results

### 9. Generate images for Model B

Use the trained SRGAN generator to create a new set of generated 128x128 images from the original low-resolution inputs. Save these images into class-based folders for Model B.

### 10. Train Model B

Run the Model B cells to:

- Build datasets from SRGAN-generated images
- Train the classifier
- Fine-tune the classifier
- Evaluate test performance
- Save metrics and confusion matrix

### 11. Compare the models

Run the final comparison cells to:

- Compare Model A and Model B using Accuracy, F1 score, and AUC
- Save the comparison plot
- Save the final summary metrics

### 12. Review saved outputs

Make sure the following are saved:

- Trained models
- SRGAN checkpoints
- Training curves
- Confusion matrices
- Generated sample images
- Metrics text files

---

## Results

### Model A

- Accuracy: 0.5750
- F1 Score: 0.0698
- AUC: 0.6873

### Model B

- Accuracy: 0.8579
- F1 Score: 0.8190
- AUC: 0.9510

### Comparison

- Accuracy improved from 0.5750 to 0.8579
- F1 improved from 0.0698 to 0.8190
- AUC improved from 0.6873 to 0.9510

These results suggest that the SRGAN-generated images provided a much more useful representation for downstream binary classification than the original baseline pipeline.

---

## Key Takeaway

The baseline transfer-learning classifier struggled and showed a strong bias toward predicting the `NORMAL` class. After training an SRGAN to generate 128x128 OCT images from 32x32 inputs, the generated images preserved plausible retinal structure and led to a major improvement in binary classification performance when used to train Model B.


