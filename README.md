# Skin Cancer Detection Project

This project trains a deep learning model to classify skin lesion images as:
- Benign (No Cancer)
- Malignant (Cancer Detected)

## Final Project Structure

```text
SkinCancerProject
│
├── dataset
│   └── train
│       ├── benign
│       │   ├── img1.jpg
│       │   ├── img2.jpg
│       │   └── ...
│       └── malignant
│           ├── img1.jpg
│           ├── img2.jpg
│           └── ...
├── model
│   └── skin_cancer_model.h5
├── static
│   └── uploads
├── templates
│   └── index.html
├── train_model.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Steps

1. Check dataset structure:

```text
dataset/
  train/
    benign/
    malignant/
```

2. Train model:

```bash
python train_model.py
```

The model will be saved as:

```text
model/skin_cancer_model.h5
```

3. Run Flask app:

```bash
python app.py
```

4. Open in browser:

```text
http://127.0.0.1:5000
```

5. Upload image and check result:
- Benign (No Cancer)
- Malignant (Cancer Detected)
