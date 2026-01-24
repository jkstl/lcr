"""
Transformers client for running HuggingFace models directly.

Used for the fine-tuned observer model since GGUF conversion failed.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TransformersClient:
    """Client for running HuggingFace transformers models directly."""
    
    def __init__(self, model_path: str):
        """
        Initialize transformers client with a local model.
        
        Args:
            model_path: Path to the HuggingFace model directory
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer into memory."""
        logger.info(f"Loading transformers model from {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        logger.info("Model loaded successfully")
    
    async def generate(
        self,
        model: str,  # Ignored (using loaded model)
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate text using the loaded model.
        
        Args:
            model: Ignored (using self.model_path)
            prompt: User prompt
            system: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # Use tokenizer's apply_chat_template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=temperature if temperature > 0 else None,
                top_k=50,
                top_p=0.1,
            )
        
        # Decode only the generated part
        generated = self.tokenizer.decode(
            outputs[0][len(inputs[0]):],
            skip_special_tokens=True
        )
        
        return generated.strip()
    
    async def close(self):
        """Cleanup resources."""
        # Move model to CPU and clear cache
        if self.model is not None:
            self.model.to("cpu")
            del self.model
            self.model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
