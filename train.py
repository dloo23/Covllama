import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import time
import datetime
import json
import argparse

# Import our modules
from dataset import get_data_loaders
from resnet_50 import create_resnet50_model
from vision_encoder import create_vision_encoder
from gradcam import create_gradcam
from llama import create_llama_model

class COVIDDetectionTrainer:
    def __init__(self, config):
        """
        Trainer for COVID-19 detection model.
        
        Args:
            config: Configuration for training
        """
        self.config = config
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                print("CUDA is not available, using CPU")
                print(f"Reason: PyTorch built with CUDA: {torch.backends.cudnn.enabled}")
        except Exception as e:
            print(f"Error setting up CUDA: {e}")
            self.device = torch.device("cpu")
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.plot_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Set up dataloaders
        self.train_loader, self.val_loader = get_data_loaders(
            base_dir=config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        
        # Training metrics
        self.best_val_acc = 0.0
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.confusion_matrices = []
        
        # Set up models first without loading checkpoint
        self._setup_models(load_checkpoint=False)
        
        # Add loss function (criterion)
        self.criterion = nn.CrossEntropyLoss()
        
        # Set up optimizers
        self._setup_optimizers()
        
        # Now load checkpoint if exists
        checkpoint_path = os.path.join(config.checkpoint_dir, "CovLlama_best.pth")
        if os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        
    def _setup_models(self, load_checkpoint=True):
        """Set up models for training."""
        # Create ResNet-50 model
        self.resnet_model = create_resnet50_model(pretrained=True).to(self.device)
        
        # Create vision encoder
        self.vision_encoder = create_vision_encoder().to(self.device)
        
        # Create GradCAM
        self.gradcam = create_gradcam(self.resnet_model)
        
        # Create Llama model
        self.llama_model = create_llama_model(model_id=self.config.llama_model_id)
        
        # Apply weight freezing if configured
        if self.config.freeze_backbone:
            # Freeze ResNet-50 backbone except the final layer
            for name, param in self.resnet_model.named_parameters():
                if "fc" not in name:  # Freeze all layers except fully connected
                    param.requires_grad = False
                
            # Print which layers are trainable
            print("Trainable layers:")
            for name, param in self.resnet_model.named_parameters():
                if param.requires_grad:
                    print(f"  {name}")
        
        # Load checkpoint if exists and loading is requested
        if load_checkpoint:
            checkpoint_path = os.path.join(self.config.checkpoint_dir, "CovLlama_best.pth")
            if os.path.exists(checkpoint_path):
                self._load_checkpoint(checkpoint_path)
        
    def _setup_optimizers(self):
        """Set up optimizers and learning rate schedulers."""
        # Optimizers
        self.resnet_optimizer = optim.Adam(
            self.resnet_model.parameters(),
            lr=self.config.resnet_lr,
            weight_decay=self.config.weight_decay
        )
        
        # Change the name here to match what's used in _save_checkpoint
        self.encoder_optimizer = optim.Adam(  # Changed from vision_encoder_optimizer
            self.vision_encoder.parameters(),
            lr=self.config.encoder_lr,
            weight_decay=self.config.weight_decay
        )
        
        # Schedulers - reduce LR when validation accuracy plateaus
        self.resnet_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.resnet_optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        self.encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.encoder_optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
    def _load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model states
            self.resnet_model.load_state_dict(checkpoint['resnet_state_dict'])
            self.vision_encoder.load_state_dict(checkpoint['vision_encoder_state_dict'])
            
            # Load optimizer states if they exist in the checkpoint
            if 'resnet_optimizer' in checkpoint and hasattr(self, 'resnet_optimizer'):
                self.resnet_optimizer.load_state_dict(checkpoint['resnet_optimizer'])
            
            # Check for either name version in the checkpoint
            if 'encoder_optimizer' in checkpoint and hasattr(self, 'encoder_optimizer'):
                self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            elif 'vision_encoder_optimizer' in checkpoint and hasattr(self, 'encoder_optimizer'):
                # Handle old checkpoint format
                self.encoder_optimizer.load_state_dict(checkpoint['vision_encoder_optimizer'])
            
            # Load training state
            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch']
            
            if 'best_val_acc' in checkpoint:
                self.best_val_acc = checkpoint['best_val_acc']
            
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
                self.val_losses = checkpoint['val_losses']
                self.train_accs = checkpoint['train_accs']
                self.val_accs = checkpoint['val_accs']
            
            print(f"Checkpoint loaded. Resuming from epoch {self.epoch+1} with best validation accuracy: {self.best_val_acc:.2f}%")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Training from scratch.")
        
    def _save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'resnet_state_dict': self.resnet_model.state_dict(),
            'vision_encoder_state_dict': self.vision_encoder.state_dict(),
            'resnet_optimizer': self.resnet_optimizer.state_dict(),
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            # Include old name for backward compatibility
            'vision_encoder_optimizer': self.encoder_optimizer.state_dict(),
            'resnet_scheduler': self.resnet_scheduler.state_dict(),
            'encoder_scheduler': self.encoder_scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
        
        # Save the latest checkpoint
        torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, "CovLlama_latest.pth"))
        
        # Save the best checkpoint separately
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, "CovLlama_best.pth"))
            
            # Also save the Llama model
            llama_path = os.path.join(self.config.checkpoint_dir, "covllama_model")
            self.llama_model.save_model(llama_path)
            
    def train_resnet_epoch(self):
        """Train ResNet-50 for one epoch."""
        self.resnet_model.train()
        self.vision_encoder.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {self.epoch+1}/{self.config.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Zero the gradients
            self.resnet_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            
            # Forward pass
            features, logits = self.resnet_model(images)
            encoded_features = self.vision_encoder(features)
            
            # Calculate loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.resnet_optimizer.step()
            self.encoder_optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
            
        # Calculate epoch statistics
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        # Store metrics
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/train', train_loss, self.epoch)
        self.writer.add_scalar('Accuracy/train', train_acc, self.epoch)
        
        return train_loss, train_acc
        
    def validate_resnet(self):
        """Validate ResNet-50 model."""
        self.resnet_model.eval()
        self.vision_encoder.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.val_loader, desc=f"Validating Epoch {self.epoch+1}/{self.config.num_epochs}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                features, logits = self.resnet_model(images)
                encoded_features = self.vision_encoder(features)
                
                # Calculate loss
                loss = self.criterion(logits, labels)
                
                # Track statistics
                running_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Store predictions and labels for confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
                
        # Calculate validation statistics
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        # Store metrics
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        self.confusion_matrices.append(cm)
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/val', val_loss, self.epoch)
        self.writer.add_scalar('Accuracy/val', val_acc, self.epoch)
        
        # Create and log confusion matrix figure
        fig = plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (Epoch {self.epoch+1})')
        self.writer.add_figure('Confusion Matrix', fig, self.epoch)
        
        # Calculate precision, recall, and F1-score
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Log metrics to TensorBoard
        self.writer.add_scalar('Metrics/precision', precision, self.epoch)
        self.writer.add_scalar('Metrics/recall', recall, self.epoch)
        self.writer.add_scalar('Metrics/f1', f1, self.epoch)
        
        return val_loss, val_acc, cm
        
    def prepare_llama_dataset(self):
        """Prepare dataset for fine-tuning the Llama model."""
        # Prepare dataset for fine-tuning Llama
        train_features = []
        train_heatmaps = []
        train_prompts = []
        train_responses = []
        train_predictions = []  # Add this line to store model predictions
        
        # Set models to eval mode
        self.resnet_model.eval()
        self.vision_encoder.eval()
        
        progress_bar = tqdm(self.train_loader, desc="Preparing Llama Training Data")
        
        for batch in progress_bar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # IMPORTANT: For feature extraction - use no_grad
            with torch.no_grad():
                # Extract features and get prediction logits
                features, logits = self.resnet_model(images)
                encoded_features = self.vision_encoder(features)
                
                # Get model predictions (this is what ResNet actually predicts)
                probs = torch.nn.functional.softmax(logits, dim=1)
                _, predictions = torch.max(probs, dim=1)
            
            # Generate GradCAM heatmaps - MUST NOT be in no_grad block
            batch_heatmaps = []
            for i in range(images.size(0)):
                img = images[i].unsqueeze(0)
                # We need gradients for GradCAM
                img.requires_grad = True
                heatmap, pred_class, pred_prob = self.gradcam.generate_heatmap(img)
                batch_heatmaps.append(heatmap)
            
            # Process the results
            for i in range(images.size(0)):
                # Patient info
                patient_id = batch['patient_id'][i]
                
                # Get true label and model prediction
                true_label = labels[i].item()
                model_prediction = predictions[i].item()
                
                # Create prompt including model's prediction
                prediction_text = "COVID-19 Positive" if model_prediction == 1 else "Normal"
                prompt = f"Patient ID: {patient_id}, Model Prediction: {prediction_text}"
                
                # Create response that AGREES with the model's prediction, not the true label
                # This is crucial - we want the LLM to explain the model's prediction, not contradict it
                if model_prediction == 1:  # Model predicts COVID-19
                    response = (
                        f"Based on my analysis of this chest X-ray, I agree with the model's prediction that this patient has COVID-19. "
                        f"The X-ray shows characteristic patterns consistent with COVID-19 pneumonia, including ground-glass opacities "
                        f"and consolidations in the lung fields. These findings are typical radiographic manifestations of COVID-19 infection. "
                        f"The distribution pattern and appearance of these opacities align with what we commonly observe in confirmed COVID-19 cases."
                    )
                else:  # Model predicts Normal
                    response = (
                        f"Based on my analysis of this chest X-ray, I agree with the model's prediction that this patient does not have COVID-19. "
                        f"The X-ray appears normal with clear lung fields. There are no visible ground-glass opacities, consolidations, or other "
                        f"patterns typically associated with COVID-19 pneumonia. The heart size and vascular markings appear within normal limits. "
                        f"This is consistent with a normal chest X-ray without radiographic evidence of COVID-19 infection."
                    )
                
                # Add to dataset
                train_features.append(encoded_features[i].cpu().numpy().tolist())
                train_heatmaps.append(batch_heatmaps[i].tolist() if isinstance(batch_heatmaps[i], np.ndarray) else batch_heatmaps[i])
                train_prompts.append(prompt)
                train_responses.append(response)
                train_predictions.append(model_prediction)  # Store the model's prediction
        
        # After collecting all data, save it to a file
        data = {
            'features': train_features,
            'heatmaps': train_heatmaps,
            'prompts': train_prompts,
            'responses': train_responses,
            'predictions': train_predictions  # Include model predictions
        }
        
        # Save the data to a file
        output_path = os.path.join(self.config.checkpoint_dir, "llama_training_data.json")
        with open(output_path, 'w') as f:
            json.dump(data, f)
        
        print(f"Saved Llama training data to {output_path}")
        print(f"Generated {len(train_features)} training examples")
        
        # Validate data
        self._validate_llama_data(output_path)
        
        return data
    
    def _validate_llama_data(self, output_path):
        """Validate that the generated data is properly formatted and can be loaded."""
        if os.path.exists(output_path):
            try:
                # Check that we can read the file back
                with open(output_path, 'r') as f:
                    data = json.load(f)
                
                # Print validation info
                print(f"✅ Successfully verified data file at {output_path}")
                print(f"  - Features: {len(data['features'])} samples")
                print(f"  - File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
                print("You can now run: python llama_finetuning.py --data_path " + output_path + " --use_tinyml")
                
                # Check for prompt-response agreement
                pos_count = sum(1 for pred in data['predictions'] if pred == 1)
                neg_count = sum(1 for pred in data['predictions'] if pred == 0)
                print(f"  - COVID-19 Positive examples: {pos_count}")
                print(f"  - Normal examples: {neg_count}")
                
            except json.JSONDecodeError:
                print(f"❌ ERROR: The file at {output_path} is not valid JSON.")
                print("   The data generation process might have produced corrupt data.")
        else:
            print(f"❌ ERROR: Expected to find {output_path} but the file does not exist.")
            print("   Check for errors in the data generation process.")
        
    def fine_tune_llama(self, train_dataset):
        """
        With Ollama, we don't directly fine-tune the model.
        This method prepares the data and prints a message.
        """
        print("Note: Direct fine-tuning of Ollama models is not supported through this interface.")
        print("The ResNet-50 model and vision encoder are trained and will be used with Ollama for inference.")
        
        # Future enhancement: You could export the trained dataset to a format 
        # that can be used with Ollama fine-tuning tools
        
    def plot_training_metrics(self):
        """Plot and save training metrics."""
        # Create figure directory
        os.makedirs(self.config.plot_dir, exist_ok=True)
        
        # Plot training and validation accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_accs) + 1), self.train_accs, 'b-', label='Training Accuracy')
        plt.plot(range(1, len(self.val_accs) + 1), self.val_accs, 'r-', label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config.plot_dir, 'accuracy_plot.png'), dpi=300)
        
        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, 'b-', label='Training Loss')
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config.plot_dir, 'loss_plot.png'), dpi=300)
        
        # Plot all confusion matrices
        if len(self.confusion_matrices) > 0:
            num_epochs = len(self.confusion_matrices)
            rows = (num_epochs + 3) // 4  # Ceiling division
            plt.figure(figsize=(20, 5 * rows))
            
            for i, cm in enumerate(self.confusion_matrices):
                plt.subplot(rows, 4, i + 1)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Epoch {i + 1}')
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.plot_dir, 'confusion_matrices.png'), dpi=300)
        
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        # Resume from the last epoch if loading from checkpoint
        start_epoch = self.epoch
        
        # Train for the full configured number of epochs
        for epoch in range(start_epoch, self.config.num_epochs):
            self.epoch = epoch
            # Train ResNet-50 for one epoch
            train_loss, train_acc = self.train_resnet_epoch()
            
            # Validate ResNet-50
            val_loss, val_acc, cm = self.validate_resnet()
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.config.num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Update learning rate schedulers
            self.resnet_scheduler.step(val_acc)
            self.encoder_scheduler.step(val_acc)
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                print(f"New best model with validation accuracy: {val_acc:.2f}%")
                
            self._save_checkpoint(is_best=is_best)
            
            # Plot training metrics every few epochs
            if (epoch + 1) % 5 == 0 or epoch == self.config.num_epochs - 1:
                self.plot_training_metrics()
        
        # Fine-tune Llama model after ResNet training is complete
        llama_dataset = self.prepare_llama_dataset()
        self.fine_tune_llama(llama_dataset)
        
        # Final evaluation and plots
        self.plot_training_metrics()
        
        print(f"Training complete! Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Model saved to {self.config.checkpoint_dir}")
        print(f"Plots saved to {self.config.plot_dir}")

class Config:
    def __init__(self, data_dir, checkpoint_dir="./checkpoints", log_dir="./logs", plot_dir="./plots",
                 batch_size=16, num_workers=4, num_epochs=20, resnet_lr=1e-4, encoder_lr=1e-4,
                 weight_decay=1e-4, freeze_backbone=False, llama_model_id="llama3"):
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.plot_dir = plot_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.resnet_lr = resnet_lr
        self.encoder_lr = encoder_lr
        self.weight_decay = weight_decay
        self.freeze_backbone = freeze_backbone
        self.llama_model_id = llama_model_id

# At the end of the file, add this function for direct data generation
def generate_llama_data():
    """Generate the Llama training data without full training."""
    # Set the default data directory
    data_dir = "YOUR DIRECTORY"
    
    # Create configuration
    config = Config(
        data_dir=data_dir,
        checkpoint_dir="./checkpoints",
        log_dir="./logs",
        plot_dir="./plots",
        batch_size=16,
        num_workers=4,
        num_epochs=1,  # Only need 1 epoch for data preparation
        resnet_lr=1e-4,
        encoder_lr=1e-4,
        weight_decay=1e-4,
        freeze_backbone=False,
        llama_model_id="llama3"
    )
    
    # Create trainer just for data generation
    print("Initializing COVID Detection Trainer...")
    trainer = COVIDDetectionTrainer(config)
    
    # Generate Llama training data only
    print("Generating Llama training data...")
    llama_dataset = trainer.prepare_llama_dataset()
    print("Data generation complete!")

# Update the main function to include the generate_data flag
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COVID-19 Detection Model Training")
    parser.add_argument('--generate-data', action='store_true', help='Generate Llama training data only')
    parser.add_argument('--data-dir', type=str, default="YOUR DIRECTORY", 
                        help='Directory containing the dataset')
    args = parser.parse_args()
    
    if args.generate_data:
        # Just run the data generation function
        generate_llama_data()
    else:
        # Run the original training code
        # Set the default data directory
        data_dir = args.data_dir
        
        # Create configuration
        config = Config(
            data_dir=data_dir,
            checkpoint_dir="./checkpoints",
            log_dir="./logs",
            plot_dir="./plots",
            batch_size=16,
            num_workers=4,
            num_epochs=30,
            resnet_lr=1e-4,
            encoder_lr=1e-4,
            weight_decay=1e-4,
            freeze_backbone=False,
            llama_model_id="llama3"
        )
        
        # Create trainer
        trainer = COVIDDetectionTrainer(config)
        trainer.train() 
