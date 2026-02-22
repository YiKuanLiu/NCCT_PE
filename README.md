# Pulmonary Embolism (PE) Classification with non-contrast CT images (NCCT)
This project aims to verify the idea of detecting Pulmonary Embolism with non-contrast CT images.
We applied VoComni pretrained SwinUNETR as our backbone and finetune the weights to classify PE.
Also, we experimented two different input types: only inhale CT images, and inhale and exhale images (IECT).
For IECT case, we duplicate two identical swinViT and encoder for each CT.
As a baseline to compare with, we extract the normalized percentile HU values from NCCT as input for a MLP model for the task.

# Data
Our dataset includes CTPA-labeled PE cases, with their inhale and exhale NCCT. 
However, our dataset is not open-sourced due to IRB regulation.
The images should be preprocessed follow the pipeline:
<img width="643" height="179" alt="image" src="https://github.com/user-attachments/assets/cf12bb19-c69e-44b3-869a-735314e84319" />



# Results
Our results shows a acceptable performance for a first-tier screening tool for PE.
<img width="741" height="226" alt="image" src="https://github.com/user-attachments/assets/499f3fbf-5a73-4210-a50a-adb78e7a84f0" />

## 🛠 Requirements

```bash
conda env create -f environment.yml
```

## 📖 Usage
```bash
bash run_10folds.sh
```



## 📄 License
This project is licensed under the MIT License.
