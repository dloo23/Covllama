import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
import json
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisionLanguageDataset(Dataset):
    """
    Dataset for fine-tuning the Llama model with vision features and GradCAM visualizations.
    Expects data from the output of train.py's prepare_llama_dataset method.
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset file (pickle or json)
            tokenizer: Tokenizer for the Llama model
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load the dataset
        logger.info(f"Loading dataset from {data_path}")
        if data_path.endswith('.pkl') or data_path.endswith('.pickle'):
            import pickle
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Extract data components
        self.features = self.data.get('features', [])
        self.heatmaps = self.data.get('heatmaps', [])
        self.prompts = self.data.get('prompts', [])
        self.responses = self.data.get('responses', [])
        
        # Validate data
        assert len(self.features) == len(self.prompts), "Features and prompts must have the same length"
        logger.info(f"Loaded dataset with {len(self.features)} examples")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        prompt = self.prompts[idx]
        
        # Extract model prediction from prompt (if available)
        prediction_text = "unknown"
        if "Model Prediction:" in prompt:
            prediction_parts = prompt.split("Model Prediction:")
            if len(prediction_parts) > 1:
                prediction_text = prediction_parts[1].strip()
        
        # Include prediction information
        prediction_info = f"\nThe model's prediction for this X-ray is: {prediction_text}"
        
        # Include heatmap information if available
        heatmap_info = ""
        if idx < len(self.heatmaps) and self.heatmaps[idx] is not None:
            heatmap = self.heatmaps[idx]
            # Describe key areas in the heatmap
            heatmap_info = "\nThe GradCAM visualization shows significant activation in "
            if np.mean(heatmap) > 0.6:
                heatmap_info += "multiple areas of the lungs, indicating widespread patterns."
            elif np.mean(heatmap) > 0.3:
                heatmap_info += "specific regions of the lungs, suggesting localized patterns."
            else:
                heatmap_info += "minimal areas, suggesting subtle or no significant findings."
        
        # Format input with explicit instruction to agree with the model's prediction
        formatted_input = f"""<|system|>
You are an expert radiologist analyzing chest X-rays for COVID-19 detection. Your task is to explain and support the model's prediction, not to contradict it. If the model predicts COVID-19, explain why the image shows signs of COVID-19. If the model predicts Normal, explain why the image appears normal without COVID-19 signs.
</s>
<|user|>
Analyze this chest X-ray:
{prompt}

The deep learning model has analyzed this X-ray and extracted key features.{heatmap_info}{prediction_info}
Please provide your analysis that explains and supports the model's prediction.
</s>
<|assistant|>
"""
        
        # Include model-generated response if available for training
        if idx < len(self.responses) and self.responses[idx]:
            formatted_input += self.responses[idx] + "</s>"
        
        # Tokenize
        tokenized_input = self.tokenizer(
            formatted_input,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Prepare inputs
        input_ids = tokenized_input.input_ids.squeeze()
        attention_mask = tokenized_input.attention_mask.squeeze()
        
        # Create labels (same as input_ids for causal language modeling)
        labels = input_ids.clone()
        
        # Mask out the padding tokens in the labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # For tokens before the assistant's response, mask out the labels
        assistant_token_idx = (input_ids == self.tokenizer.encode("<|assistant|>", add_special_tokens=False)[0]).nonzero(as_tuple=True)[0]
        if len(assistant_token_idx) > 0:
            assistant_idx = assistant_token_idx[0]
            labels[:assistant_idx+1] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def load_and_prepare_model(model_id="meta-llama/Llama-3.2-1B-Instruct", device_map="auto"):
    """
    Load and prepare the model and tokenizer for fine-tuning.
    
    Args:
        model_id: Hugging Face model ID
        device_map: Device mapping strategy
        
    Returns:
        model: Prepared model
        tokenizer: Tokenizer
    """
    logger.info(f"Loading model {model_id}")
    
    # Configure BitsAndBytes for quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    
    # Make sure the tokenizer has padding token
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token = tokenizer.eos_token = "</s>"
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float16
    )
    
    # Prepare the model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,               # Rank of the update matrices
        lora_alpha=32,      # Parameter for scaling
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Target attention modules
        lora_dropout=0.05,  # Dropout probability
        bias="none",        # Don't train bias parameters
        task_type=TaskType.CAUSAL_LM  # Task type
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params} ({100 * trainable_params / total_params:.2f}% of total)")
    
    return model, tokenizer

def train_llama_model(
    model,
    tokenizer,
    train_dataset,
    val_dataset=None,
    output_dir="./llama_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4
):
    """
    Fine-tune the Llama model.
    
    Args:
        model: Model to fine-tune
        tokenizer: Tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Output directory
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device during training
        learning_rate: Initial learning rate
    """
    # Check if datasets have examples
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty! Cannot proceed with training.")
    
    logger.info(f"Training with {len(train_dataset)} examples, validating with {len(val_dataset)} examples")
    
    # Define training arguments with ALL required fields explicitly set
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        fp16=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        report_to="tensorboard",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Explicitly set these to avoid None values
        max_steps=-1,
        warmup_steps=0,
        gradient_accumulation_steps=1,
        dataloader_num_workers=2
    )
    
    # Create a data collator that handles padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Not using masked language modeling
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    logger.info("Starting model fine-tuning")
    trainer.train()
    
    # Save the fine-tuned model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

def prepare_data_from_train_output(train_output_path, output_path="llama_training_data.json"):
    """
    Process the output from train.py's prepare_llama_dataset method.
    
    Args:
        train_output_path: Path to the output from train.py
        output_path: Path to save the processed data
        
    Returns:
        Path to the processed data
    """
    # Load the data from the train.py output
    logger.info(f"Loading data from {train_output_path}")
    
    # Check if the file exists first
    if not os.path.exists(train_output_path):
        raise FileNotFoundError(
            f"The data file {train_output_path} does not exist.\n"
            f"Did you run 'python generate_llama_data.py' first?\n"
            f"Current working directory: {os.getcwd()}"
        )
    
    try:
        if train_output_path.endswith('.pkl') or train_output_path.endswith('.pickle'):
            import pickle
            with open(train_output_path, 'rb') as f:
                data = pickle.load(f)
        elif train_output_path.endswith('.json'):
            with open(train_output_path, 'r') as f:
                try:
                    data = json.load(f)
                    logger.info(f"Successfully loaded JSON with {len(data.get('features', []))} samples")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Could not parse JSON file: {str(e)}")
        else:
            raise ValueError(f"Unsupported file format: {train_output_path}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
    
    # Validate the loaded data
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dictionary, got {type(data)}")
    
    required_keys = ['features', 'prompts']
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValueError(f"Data is missing required keys: {missing_keys}")
    
    # Extract and process the data
    features = data.get('features', [])
    heatmaps = data.get('heatmaps', [])
    prompts = data.get('prompts', [])
    responses = data.get('responses', [])
    
    # Check if we have any data
    if len(features) == 0:
        raise ValueError(f"The data file {train_output_path} contains no features!")
    
    # Create splits for training and validation
    num_samples = len(features)
    logger.info(f"Found {num_samples} total examples in the dataset")
    
    # For very small datasets, ensure we have at least 1 example in each split
    if num_samples == 1:
        # Duplicate the single example for both train and val
        train_indices = [0]
        val_indices = [0]
    else:
        # Regular split
        indices = np.random.permutation(num_samples)
        train_size = max(1, int(0.9 * num_samples))  # Ensure at least 1 training example
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:] if train_size < num_samples else [indices[0]]  # Fallback
    
    logger.info(f"Split into {len(train_indices)} training and {len(val_indices)} validation examples")
    
    # Prepare the training data
    train_data = {
        'features': [features[i] for i in train_indices],
        'heatmaps': [heatmaps[i] if i < len(heatmaps) else None for i in train_indices],
        'prompts': [prompts[i] for i in train_indices],
        'responses': [responses[i] if i < len(responses) else None for i in train_indices]
    }
    
    # Prepare the validation data
    val_data = {
        'features': [features[i] for i in val_indices],
        'heatmaps': [heatmaps[i] if i < len(heatmaps) else None for i in val_indices],
        'prompts': [prompts[i] for i in val_indices],
        'responses': [responses[i] if i < len(responses) else None for i in val_indices]
    }
    
    # Save the processed data
    train_path = output_path.replace('.json', '_train.json')
    val_path = output_path.replace('.json', '_val.json')
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f)
    
    with open(val_path, 'w') as f:
        json.dump(val_data, f)
    
    logger.info(f"Saved training data to {train_path} and validation data to {val_path}")
    return train_path, val_path

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama model for COVID-19 X-ray analysis")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data from train.py")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--output_dir", type=str, default="./llama_finetuned", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--use_tinyml", action="store_true", help="Use TinyLlama instead of Llama")
    
    args = parser.parse_args()
    
    # Use TinyLlama if specified (for users without access to Llama models)
    if args.use_tinyml:
        args.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Prepare the data
    train_data_path, val_data_path = prepare_data_from_train_output(args.data_path)
    
    # Load and prepare the model
    model, tokenizer = load_and_prepare_model(args.model_id)
    
    # Create datasets
    train_dataset = VisionLanguageDataset(train_data_path, tokenizer)
    val_dataset = VisionLanguageDataset(val_data_path, tokenizer)
    
    # Train the model
    train_llama_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    logger.info(f"Fine-tuning complete! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main() 