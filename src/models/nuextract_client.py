"""
NuExtract client for structured entity/relationship extraction.

NuExtract is a specialized extraction model that uses JSON templates
to guide extraction and prevents hallucination through purely extractive approach.
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from typing import Optional, Dict, Any, List
import logging
import json

logger = logging.getLogger(__name__)


class NuExtractClient:
    """Client for NuExtract structured extraction model."""

    def __init__(self, model_name: str = "numind/NuExtract-2.0-4B"):
        """
        Initialize NuExtract client.

        Args:
            model_name: HuggingFace model name or local path
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load model and processor into memory."""
        logger.info(f"Loading NuExtract model from {self.model_name}")

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

            logger.info("NuExtract model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NuExtract model: {e}")
            raise

    async def extract(
        self,
        document: str,
        template: Dict[str, Any],
        examples: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 2048,
    ) -> str:
        """
        Extract structured information from document using template.

        Args:
            document: Text to extract from
            template: JSON schema defining what to extract
            examples: Optional list of example input/output pairs for in-context learning
            max_tokens: Maximum tokens to generate

        Returns:
            JSON string with extracted data
        """
        # Build messages
        messages = [{"role": "user", "content": document}]

        # Apply chat template with extraction schema
        text = self.processor.tokenizer.apply_chat_template(
            messages,
            template=json.dumps(template, indent=2),
            examples=examples if examples else [],
            tokenize=False,
            add_generation_prompt=True,
        )

        # No images for text-only extraction
        inputs = self.processor(
            text=[text],
            images=None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate with temperature=0 for deterministic extraction
        generation_config = {
            "do_sample": False,  # Deterministic
            "num_beams": 1,
            "max_new_tokens": max_tokens,
        }

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_config)

        # Decode only the generated part
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_text[0].strip()

    async def generate(
        self,
        model: str,  # Ignored (using loaded model)
        prompt: str,
        system: Optional[str] = None,
        template: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Generate text using NuExtract (compatibility wrapper).

        This provides a similar interface to TransformersClient for easier integration.

        Args:
            model: Ignored (using self.model_name)
            prompt: Document text to extract from
            system: System message (used to infer template if template not provided)
            template: JSON schema for extraction
            examples: Optional examples for in-context learning

        Returns:
            Generated extraction (JSON string)
        """
        # If no template provided, use a default entity/relationship template
        if template is None:
            template = self._default_extraction_template()

        return await self.extract(
            document=prompt,
            template=template,
            examples=examples,
        )

    def _default_extraction_template(self) -> Dict[str, Any]:
        """
        Default extraction template for entity/relationship extraction.

        Returns:
            JSON template schema
        """
        return {
            "fact_type": "string",
            "entities": [{
                "name": "verbatim-string",
                "type": "string",
                "attributes": {}
            }],
            "relationships": [{
                "subject": "verbatim-string",
                "predicate": "string",
                "object": "verbatim-string",
                "temporal": "string"
            }]
        }

    async def close(self):
        """Cleanup resources."""
        if self.model is not None:
            self.model.to("cpu")
            del self.model
            self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
