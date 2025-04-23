# Advanced X-ray Analysis System

A deep learning-based system for analyzing chest X-rays to detect COVID-19 and pneumonia using ResNet50 and LLaMA models.

## Features
- Lung X-ray validation using SVM
- COVID-19 and pneumonia detection using ResNet50
- Detailed analysis generation using LLaMA deployed by using Ollama to run locally
- Interactive web interface using Streamlit
- GradCAM visualization for model interpretability

## Project Structure
```python
.
├── app.py                  # Main Streamlit application
├── train.py               # Training script for ResNet and LLaMA
├── resnet50.py            # ResNet50 model architecture
├── vision_encoder.py      # Vision encoder for feature processing
├── llama.py              # LLaMA model implementation
├── gradcam.py            # GradCAM visualization
└── requirements.txt      # Project dependencies
```

## Setup Instructions

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv covllama_venv

# Activate virtual environment
# On Windows:
covllama_venv\Scripts\activate
# On Unix or MacOS:
source covllama_venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Model Setup
Download required model files and place them in the `checkpoints` directory:
- ResNet50 checkpoint: `resnet50_rand.pth`
- LLaMA model: `covllama_best_rand.pth`
- Replace YOUR DIRECTORY to your own directory in app.py & Train.py\
- Install Ollama 

### 3. Running the Application
```bash
streamlit run app.py
```

## Training

### Training New Models
```bash
python train.py
```
Training options include:
- Dataset selection (COVID-19, Pneumonia, or combined)
- Early stopping configuration
- Model loading from checkpoints
- Fine-tuning options (LoRA/QLoRA)

## Usage
1. Launch the application
2. Upload a chest X-ray image
3. Click "Analyze Image"
4. View results including:
   - Classification result
   - Confidence score
   - GradCAM visualization
   - Detailed analysis

## Model Architecture
- **ResNet50**: Feature extraction and classification
- **Vision Encoder**: Feature processing for LLaMA
- **LLaMA**: Natural language analysis generation
- **SVM**: Lung X-ray validation

## Dependencies
- PyTorch
- Streamlit
- transformers
- scikit-learn
- PIL
- numpy
- pandas
- tqdm
- ollama

## Notes
- Large model files are not included in the repository
- Training data and checkpoints should be downloaded separately
- Requires CUDA-capable GPU for optimal performance
- INSTALL OLLAMA!! else it won't work!

## License
[Your chosen license]

## Acknowledgments
- ImageNet for pretrained weights (RESNET-50)
- Meta for LLaMA model 3.2 1b model
- Ollama
- COVID19+PNEUMONIA+NORMAL Chest X-Ray Image Dataset (https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images)
- COVIDx CXR-4 (https://www.kaggle.com/datasets/andyczhao/covidx-cxr2)
- My Project Supervisor Dr. Shamsul Masum

  ![covllama logo](https://github.com/user-attachments/assets/19e1fdf3-de89-481e-af34-d15133a5983d)
