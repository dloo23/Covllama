import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Grad-CAM implementation for COVID-19 detection.
        
        Args:
            model: ResNet-50 model
            target_layer: Target layer for visualization (typically the last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        
        # Gradients and activations
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.register_hooks()
        
    def register_hooks(self):
        """Register hooks for gradient and activation capture."""
        # Forward hook to capture activations
        def forward_hook(module, input, output):
            self.activations = output
            return None
            
        # Backward hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            return None
        
        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
        
    def generate_heatmap(self, input_tensor):
        """
        Generate a GradCAM heatmap for the input image.
        
        Args:
            input_tensor: Input image tensor
            
        Returns:
            heatmap: CAM heatmap (numpy array)
            pred_class: Predicted class (0 or 1)
            pred_prob: Prediction probability
        """
        # Ensure input requires gradients
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass through ResNet
        self.model.eval()
        
        # Reset gradients and activations
        self.gradients = None
        self.activations = None
        
        # Get features and prediction
        features, logits = self.model(input_tensor)
        
        # Get prediction probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)
        _, pred_class = torch.max(probs, dim=1)
        pred_class = pred_class.item()
        pred_prob = probs[0, pred_class].item()
        
        # Set up target for backpropagation
        target = torch.zeros_like(logits)
        target[0, pred_class] = 1
        
        # Make sure we zero any existing gradients
        self.model.zero_grad()
        
        # Backpropagate to get gradients
        logits.backward(gradient=target, retain_graph=True)
        
        # Check if gradients were captured
        if self.gradients is None:
            print("Warning: Gradients not captured, creating fallback heatmap")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3])), pred_class, pred_prob
        
        # Calculate gradient-weighted activations
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the activations by the average gradients
        activations = self.activations
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the weighted activations
        heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu().numpy()
        
        # ReLU on the heatmap
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize the heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
            
        return heatmap, pred_class, pred_prob
        
    def overlay_heatmap(self, image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on the original image.
        
        Args:
            image: Original image (PIL or tensor)
            heatmap: Generated heatmap
            alpha: Transparency factor
            colormap: Colormap for heatmap
            
        Returns:
            Overlaid image as numpy array
        """
        # If image is a tensor, convert to numpy
        if isinstance(image, torch.Tensor):
            # Denormalize and convert to numpy
            image = image.squeeze().permute(1, 2, 0).cpu().numpy()
            image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            image = image.astype(np.uint8)
        elif isinstance(image, Image.Image):
            image = np.array(image)
            
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Apply colormap to heatmap
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        
        # Convert RGB to BGR for cv2
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        # Overlay heatmap on image
        overlaid = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        # Convert back to RGB
        overlaid = cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)
        
        return overlaid

def create_gradcam(model):
    """Create GradCAM instance for the given model."""
    # Find the last convolutional layer in the model
    # For ResNet-50, it's usually in layer4
    target_layer = None
    
    # Try to find the last conv layer in layer4
    try:
        if hasattr(model, 'resnet'):
            target_layer = model.resnet.layer4[-1].conv3
        elif hasattr(model, 'layer4'):
            target_layer = model.layer4[-1].conv3
        else:
            # Find any conv layer
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
            
        print(f"Using target layer: {target_layer}")
    except Exception as e:
        print(f"Error finding target layer: {e}")
        # Fallback to some likely layer
        try:
            target_layer = model.resnet.layer4[-1].conv3
        except:
            # Last resort
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    break
    
    return GradCAM(model, target_layer) 