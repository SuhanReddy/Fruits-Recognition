# 🍎 Fruits Recognition using Deep Learning

A PyTorch-based fruit recognition project using **ResNet18 transfer learning** and **OpenCV** for real-time webcam recognition.

## Run modes

- `main` is the local Python workflow. Install the dependencies and run the commands below on your computer.
- `github-actions` runs training and evaluation on GitHub Actions when changes are pushed to that branch. It does not change the local workflow on `main`.

### Run on GitHub

The dataset is not stored in this repository, so the GitHub runner needs a direct URL to a ZIP archive containing the `Training/` and `Test/` folders. Before pushing `github-actions`:

1. In the GitHub repository, open **Settings → Secrets and variables → Actions → Variables**.
2. Add a repository variable named `FRUITS_360_ZIP_URL` whose value is the direct download URL for the dataset ZIP.
3. Push the `github-actions` branch. Each push that changes Python files, dependencies, or the workflow starts training and evaluation.

When the run finishes, open the repository's **Actions** tab, select the run, and download the `fruit-recognition-results` artifact. It contains the model checkpoint and files from `outputs/`.

The GitHub run uses a CPU runner and one frozen-backbone training epoch by default. The local commands below remain available on `main` for running with your own hardware and settings.

## ✨ Features

- 🧠 ResNet18 transfer learning
- 🖼️ Single-image prediction
- 📷 Real-time webcam recognition
- 📊 Training curves and confusion matrix
- 💾 Automatic model saving

## 📁 Project Structure

Fruit Recognition/
├── data.py
├── model.py
├── train.py
├── evaluate.py
├── predict.py
├── webcam.py
├── requirements.txt
├── outputs/
├── models/
└── README.md

> **The dataset is NOT included in this repository.**  
> Download Fruits-360 separately and place it in a folder called `dataset/`.

## 📥 1. Download the Dataset

Download **Fruits-360 100x100** from:

🔗 https://github.com/fruits-360/fruits-360-100x100
or
🔗 https://www.kaggle.com/moltean/fruits

After downloading, your project should contain:

dataset/
├── Training/
└── Test/

### Important

Do **not** use:
dataset/fruits-360-100x100/Training

The correct structure is:
dataset/Training
dataset/Test

## ⚙️ 2. Install Dependencies

Open a terminal in the project folder and run:
pip install -r requirements.txt
pip install opencv-python

Check that everything works:
python -c "import torch, torchvision, cv2; print('OK')"

You should see: OK

## 🚀 3. Train the Model

Run:
python train.py --data_dir "dataset" --epochs 1 --freeze_backbone --num_workers 0

### What this does

- `--data_dir "dataset"` → uses the downloaded dataset
- `--epochs 1` → runs 1 training epoch
- `--freeze_backbone` → freezes the pretrained ResNet18 backbone
- `--num_workers 0` → uses a simple data-loading setup

⏱️ On some computers, this run can take around **25–30 minutes**. Actual time depends on your hardware.

The trained model is saved in: models/

## 📊 4. Evaluate the Model

Run:
python evaluate.py

This evaluates the trained model and can generate results such as:

outputs/
├── training_curves.png
└── confusion_matrix.png

## 🖼️ 5. Predict an Image
Check the available prediction options:
python predict.py --help

The script can be used to classify an image and display the top predictions.

## 📷 6. Run Real-Time Webcam Recognition
After training, run:
python webcam.py

The webcam pipeline captures frames, detects/crops the object, and sends it through the trained ResNet18 model for prediction.

## 🔄 How It Works
Fruits-360 Dataset
        ↓
Preprocessing
        ↓
ResNet18 Transfer Learning
        ↓
Trained Model
     ↙     ↘
Image      Webcam
Prediction Recognition

## 📌 Notes

- The **dataset is downloaded separately**; it is not stored in this GitHub repository.
- The **trained `.pth` model** is generated after training.
- A GPU can speed up training, but it is not required.
- Fruits-360 images are more controlled than typical real-world webcam images, so real-time results may vary.

## 🛠️ Technologies

**Python · PyTorch · Torchvision · ResNet18 · OpenCV · NumPy · Matplotlib · Scikit-learn**

## ⭐ Quick Start

pip install -r requirements.txt
pip install opencv-python
python -c "import torch, torchvision, cv2; print('OK')"
python train.py --data_dir "dataset" --epochs 1 --freeze_backbone --num_workers 0
python evaluate.py
python webcam.py