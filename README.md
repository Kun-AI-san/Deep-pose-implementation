# 🏃 DeepPose Implementation — Human Pose Estimation Pipeline

This project implements a two-stage deep learning pipeline for human pose estimation based on the **DeepPose** architecture. The models used include a ResNet backbone followed by cascaded AlexNet-based networks.

---

## 📋 Instructions

Follow these steps in sequence to run the full pipeline. All required dependencies and scripts are included.

### 1️⃣ Run `LSP_data.py`

- Prepares the **Leeds Sports Pose (LSP)** and **Extended LSP (LSPe)** datasets.
- Applies data augmentations: cropping, translations, flips.
- Saves output:
  - `train_images.npy`, `train_labels.npy`
  - `test_images.npy`, `test_labels.npy`

**Download LSPe dataset:** [LSPe Dataset](http://sam.johnson.io/research/lspet.html)

```bash
python LSP_data.py
```

---

### 2️⃣ Run `Train_stage_1.py`

- Loads and trains a ResNet model (from `Resnet_stage_1.py`) on the training data.
- Saves the model for use in stage 2.
- Optional: Use `Generate_train_output.py` to generate `training_output.npy` instead of re-training.

```bash
python Train_stage_1.py
```

---

### 3️⃣ Run `test_stage_1.py`

- Loads the ResNet model and evaluates it on the test set.

```bash
python test_stage_1.py
```

---

### 4️⃣ Run `Generate_train_output.py`

- Generates `training_output.npy`, used for cascade training in stage 2.

```bash
python Generate_train_output.py
```

---

### 5️⃣ Run `Train_stage_2.py`

- Trains 14 independent AlexNet-based models (from `Cascade_1_net.py`) for each joint.
- Uses output from stage 1 and performs data augmentation using `deltas.npy`.
- Saves one model per joint.

```bash
python Train_stage_2.py
```

---

### 6️⃣ Run `test_stage_2.py`

- Evaluates stage 2 (joint-level) models on the test data.

```bash
python test_stage_2.py
```

---

## 📁 Important Files

- `Resnet_stage_1.py`: Defines the ResNet model.
- `Cascade_1_net.py`: Defines the AlexNet cascade (`Alex_net_c1` class).
- `train_output.npy`: Output from ResNet stage used for stage 2 training.
- `deltas.npy`: Simulated prediction offsets for data augmentation.
- `images_1/` and `images_2/`: Image folders from LSP/LSPe.
- `joints_1.mat` and `joints_2.mat`: Joint annotations.

---

## 🧰 Dependencies

Make sure the following packages are installed:

- PyTorch
- torchvision
- NumPy
- CUDA (if using GPU)

---

## 📄 Disclaimer

The `Cascade_1_net.py` file contains an adaptation of the original **AlexNet** architecture used in the DeepPose paper.

> I do not claim authorship for the `Alex_net_c1` class.

### 📚 Reference

Toshev, A., & Szegedy, C. (2014). DeepPose: Human Pose Estimation via Deep Neural Networks.  
*CVPR 2014*. [arXiv:1312.4659](https://arxiv.org/abs/1312.4659)

---

## ✅ Summary

This pipeline reconstructs the DeepPose model across two training stages, with careful augmentation and modular training scripts for reproducibility.
