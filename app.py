import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import io

# Import our modules
from resnet_50 import create_resnet50_model
from vision_encoder import create_vision_encoder
from gradcam import create_gradcam
from llama import create_llama_model

class COVIDDetectionApp:
    def __init__(self):
        """Initialize the COVID-19 detection application."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Set up model paths
        self.checkpoint_dir = "./checkpoints"
        self.best_model_path = os.path.join(self.checkpoint_dir, "CovLlama_best.pth")
        self.llama_model_path = os.path.join(self.checkpoint_dir, "covllama_model")
        
        # Load models when initialized
        self.load_models()
        
    def load_models(self):
        """Load all pre-trained models."""
        # Check if models exist
        if not os.path.exists(self.best_model_path):
            st.error("Model checkpoint not found. Please train the model first.")
            return False
            
        try:
            # Create ResNet-50 model
            self.resnet_model = create_resnet50_model(pretrained=False).to(self.device)
            
            # Create Vision Encoder
            self.vision_encoder = create_vision_encoder().to(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.resnet_model.load_state_dict(checkpoint['resnet_state_dict'])
            self.vision_encoder.load_state_dict(checkpoint['vision_encoder_state_dict'])
            
            # Create GradCAM
            self.gradcam = create_gradcam(self.resnet_model)
            
            # Use the specified fine-tuned model
            fine_tuned_model_path = r"YOUR DIRECTORY"
            
            # Check if fine-tune model exists, otherwise fall back to Ollama
            if os.path.exists(fine_tuned_model_path):
                self.llama_model = create_llama_model(
                    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    checkpoint_path=fine_tuned_model_path
                )
                print(f"Using fine-tuned TinyLlama model from: {fine_tuned_model_path}")
            else:
                self.llama_model = create_llama_model()
                print(f"Fine-tuned model not found at {fine_tuned_model_path}, using Ollama")
            
            # Set models to evaluation mode
            self.resnet_model.eval()
            self.vision_encoder.eval()
            
            return True
            
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False
            
    def preprocess_image(self, image):
        """Preprocess the input image for model inference."""
        from torchvision import transforms
        
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply preprocessing
        input_tensor = preprocess(image)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        
        return input_tensor
        
    def predict(self, image, patient_id="Unknown"):
        """Run the full prediction pipeline on an input image."""
        # Preprocess image
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # Explicitly set requires_grad=True for GradCAM
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Get model predictions
        with torch.set_grad_enabled(True):  # Ensure gradients are enabled
            # Extract features and get classification logits
            features, logits = self.resnet_model(input_tensor)
            
            # Generate GradCAM heatmap
            try:
                heatmap, pred_class, pred_prob = self.gradcam.generate_heatmap(input_tensor)
                heatmap_image = self.gradcam.overlay_heatmap(image, heatmap)
            except RuntimeError as e:
                print(f"GradCAM error: {e}")
                # Fallback: Create a blank heatmap
                heatmap = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
                heatmap_image = image
                # Still get prediction from the logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                pred_prob = probs[0, pred_class].item()  # Convert to Python scalar immediately
            
            # Encode features
            encoded_features = self.vision_encoder(features)
        
        # After we're done with gradient computation, detach everything
        # Make sure features no longer require gradients for the language model
        encoded_features = encoded_features.detach()
            
        # Convert prediction to text
        prediction = 1 if pred_class == 1 else 0
        prediction_text = "COVID-19 Positive" if prediction == 1 else "Normal"
        
        # Calculate confidence percentage safely
        try:
            if torch.is_tensor(pred_prob):
                confidence = pred_prob.item() * 100
            else:
                confidence = float(pred_prob) * 100
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            confidence = 0.0  # Default value
        
        # Prepare prompt for Llama including the prediction
        prompt = f"Patient ID: {patient_id}\nModel Prediction: {prediction_text} (Confidence: {confidence:.1f}%)"
        
        # Generate response using Llama model
        responses = self.llama_model.generate_response(
            encoded_features.cpu().numpy(), 
            [heatmap], 
            [prompt]
        )
        
        # Extract the first response
        llama_response = responses[0] if responses else "No response generated."
        
        # Return results
        result = {
            'prediction': "COVID-19 Positive" if prediction == 1 else "Normal",
            'confidence': confidence,
            'heatmap_image': heatmap_image,
            'llama_response': llama_response
        }
        
        return result

def main():
    # Set page title and config
    st.set_page_config(
        page_title="COVID-19 Detection System",
        page_icon="🫁",
        layout="wide"
    )
    
    # Page title
    st.title("COVID-19 Detection from Chest X-rays")
    st.markdown("### Using ResNet-50 and Llama 3.2")
    
    # Initialize application
    if 'app' not in st.session_state:
        with st.spinner("Loading models... This may take a minute."):
            st.session_state.app = COVIDDetectionApp()
    
    app = st.session_state.app
    
    # Create sidebar
    st.sidebar.title("Upload X-ray Image")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])
    
    # Patient ID input
    patient_id = st.sidebar.text_input("Patient ID", "P0001")
    
    # Main content area
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Create columns for display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original X-ray Image")
            st.image(image, use_column_width=True)
        
        # Run prediction
        with st.spinner("Analyzing image..."):
            result = app.predict(image, patient_id)
        
        # Display results
        with col2:
            st.subheader("GradCAM Visualization")
            st.image(result['heatmap_image'], use_column_width=True)
            
            # Display prediction with color
            prediction_color = "red" if result['prediction'] == "COVID-19 Positive" else "green"
            st.markdown(f"### Prediction: <span style='color:{prediction_color}'>{result['prediction']}</span>", unsafe_allow_html=True)
            st.markdown(f"### Confidence: {result['confidence']:.2f}%")
        
        # Display Llama analysis
        st.subheader("Detailed Analysis")
        st.markdown(result['llama_response'])
        
    else:
        # Display instructions when no image is uploaded
        st.info("Please upload a chest X-ray image to get COVID-19 detection results.")
        
        # Add some sample images if available
        sample_dir = "sample_images"
        if os.path.exists(sample_dir):
            st.subheader("Or try with a sample image:")
            
            sample_images = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if sample_images:
                sample_cols = st.columns(min(3, len(sample_images)))
                
                for i, img_file in enumerate(sample_images[:3]):
                    img_path = os.path.join(sample_dir, img_file)
                    img = Image.open(img_path).convert('RGB')
                    
                    with sample_cols[i]:
                        st.image(img, caption=img_file, use_column_width=True)
                        if st.button(f"Use this image", key=f"sample_{i}"):
                            # Process the sample image
                            with st.spinner("Analyzing sample image..."):
                                result = app.predict(img, f"Sample_{i+1}")
                                
                            # Display results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Original X-ray Image")
                                st.image(img, use_column_width=True)
                                
                            with col2:
                                st.subheader("GradCAM Visualization")
                                st.image(result['heatmap_image'], use_column_width=True)
                                
                                # Display prediction with color
                                prediction_color = "red" if result['prediction'] == "COVID-19 Positive" else "green"
                                st.markdown(f"### Prediction: <span style='color:{prediction_color}'>{result['prediction']}</span>", unsafe_allow_html=True)
                                st.markdown(f"### Confidence: {result['confidence']:.2f}%")
                            
                            # Display Llama analysis
                            st.subheader("Detailed Analysis")
                            st.markdown(result['llama_response'])
    
    # Add information about the project
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This application uses a ResNet-50 model to extract features from chest X-rays, "
        "visualizes important regions using GradCAM, and provides a detailed analysis using "
        "a fine-tuned Llama 3.2 language model. The system can help identify COVID-19 cases "
        "from chest X-ray images."
    )

if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    main() 
