import os
import torch
import requests
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from PIL import Image

class LlamaCovidModel:
    def __init__(self, model_id="llama3", device="cuda"):
        """
        Initialize the Llama model for COVID-19 detection.
        Can use either Ollama API or local fine-tuned model.
        
        Args:
            model_id: Ollama model name or path to fine-tuned model
            device: Device to run the model on
        """
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self.use_local_model = False
        self.ollama_url = "http://localhost:11434/api/generate"
        
    def load_model(self, checkpoint_path=None):
        """
        Load the model from checkpoint or initialize for API usage.
        
        Args:
            checkpoint_path: Path to fine-tuned model checkpoint
        """
        print(f"Loading model {self.model_id}...")
        
        # Try to load local fine-tuned model first
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                print(f"Loading fine-tuned model from {checkpoint_path}")
                self.use_local_model = True
                
                # For quantized inference
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                
                # Load base model with quantization
                base_model_id = self.model_id
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
                
                if self.device == "cuda" and torch.cuda.is_available():
                    print("Using quantized model on CUDA")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        base_model_id,
                        quantization_config=bnb_config,
                        device_map="auto"
                    )
                else:
                    print("Using CPU model")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        base_model_id,
                        device_map="auto"
                    )
                
                # Load the adapter weights
                print(f"Loading adapter from {checkpoint_path}")
                if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        checkpoint_path,
                        is_trainable=False
                    )
                    print("Successfully loaded fine-tuned model with adapter")
                else:
                    print(f"Warning: No adapter_config.json found in {checkpoint_path}. Continuing with base model.")
                
                # Set to evaluation mode
                self.model.eval()
                print("Model loaded successfully!")
                return True
                
            except Exception as e:
                print(f"Error loading fine-tuned model: {str(e)}")
                print("Falling back to Ollama API")
                self.use_local_model = False
        
        # Check if we need to use Ollama as fallback
        if not self.use_local_model:
            try:
                # Test if Ollama is available with this model
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.model_id,
                        "prompt": "test",
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    print(f"Using Ollama API with model {self.model_id}")
                    return True
                else:
                    print(f"Model {self.model_id} not found in Ollama. Please pull it using 'ollama pull {self.model_id}'")
                    return False
                    
            except Exception as e:
                print(f"Error connecting to Ollama: {str(e)}")
                return False
                
        return True

    def generate_response(self, features, heatmaps, prompts):
        """
        Generate response for the given features and prompts.
        
        Args:
            features: Vision features (batch_size, feature_dim)
            heatmaps: GradCAM heatmaps (batch_size, H, W)
            prompts: Text prompts (batch_size,)
            
        Returns:
            List of responses
        """
        formatted_prompts = self.prepare_inputs_for_generation(features, heatmaps, prompts)
        responses = []
        
        if self.use_local_model:
            # Generate with local model
            for prompt in formatted_prompts:
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    
                    # Generate with local model
                    with torch.no_grad():
                        output = self.model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            repetition_penalty=1.1
                        )
                    
                    # Decode output
                    response_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    
                    # Extract just the assistant's response from template
                    if "<|assistant|>" in response_text:
                        response_parts = response_text.split("<|assistant|>")
                        if len(response_parts) > 1:
                            response_text = response_parts[1].strip()
                            if "</s>" in response_text:
                                response_text = response_text.split("</s>")[0].strip()
                    
                    responses.append(response_text)
                    
                except Exception as e:
                    print(f"Error generating response with local model: {str(e)}")
                    responses.append("Error generating response from local model")
        else:
            # Generate with Ollama API
            for prompt in formatted_prompts:
                try:
                    response = requests.post(
                        self.ollama_url,
                        json={
                            "model": self.model_id,
                            "prompt": prompt,
                            "stream": False
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        responses.append(data["response"])
                    else:
                        print(f"Error from Ollama API: {response.text}")
                        responses.append("Error generating response from Ollama")
                        
                except Exception as e:
                    print(f"Error calling Ollama API: {str(e)}")
                    responses.append("Error connecting to Ollama")
        
        return responses
        
    def prepare_inputs_for_generation(self, features, heatmaps, prompts):
        """
        Prepare inputs for the Llama model.
        
        Args:
            features: Encoded vision features from the vision encoder
            heatmaps: GradCAM heatmaps
            prompts: Text prompts for the model
            
        Returns:
            List of formatted prompts
        """
        # For each example in the batch, create a prompt with the feature information
        formatted_prompts = []
        
        for i, (prompt, feature) in enumerate(zip(prompts, features)):
            # Create a comprehensive prompt
            if self.use_local_model:
                # Format for fine-tuned model using a more concise prompt template
                formatted_prompt = f"""<|system|>
You are an expert radiologist analyzing chest X-rays for COVID-19. Provide a clear, concise analysis of the X-ray based on the model's prediction. Focus on relevant clinical findings.
</s>
<|user|>
{prompt}

Please provide a brief radiological assessment.
</s>
<|assistant|>
"""
            else:
                # Format for Ollama
                formatted_prompt = f"Analyze this chest X-ray image for COVID-19 detection.\n\n"
                formatted_prompt += f"Patient Information: {prompt}\n\n"
                formatted_prompt += f"Provide a brief medical assessment of the X-ray findings."
            
            formatted_prompts.append(formatted_prompt)
            
        return formatted_prompts
        
    def apply_qlora(self):
        """No-op for inference"""
        pass
        
    def save_model(self, output_dir="./covllama_model"):
        """
        Not needed for inference, kept for compatibility.
        """
        print(f"Note: No saving needed for inference")

def create_llama_model(model_id="llama3", checkpoint_path=None):
    """Create and initialize the Llama model."""
    llama_model = LlamaCovidModel(model_id=model_id)
    llama_model.load_model(checkpoint_path=checkpoint_path)
    return llama_model 