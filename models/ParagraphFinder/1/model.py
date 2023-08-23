# model.py
from typing import List, Dict, Optional
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import logging
logging.basicConfig(level=logging.INFO)


class TritonPythonModel():
    def initialize(self, args):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def generate_text(self, input_text, max_length=1000):
        # Encode the input text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        attention_mask = torch.ones(input_ids.shape)

        # Generate text with proper punctuation and capitalization
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=max_length, num_return_sequences=1, top_k=50, top_p=0.95,
                                         pad_token_id = self.tokenizer.eos_token_id, attention_mask = attention_mask,
                                         no_repeat_ngram_size=2)  # Prevent repeating 2-grams and repeat sentences

        # Decode the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text


    def paragraph_finder(self,input_text):
        # Generate a more structured version of the input text
        structured_text = self.generate_text(input_text)
        # Split the structured text into paragraphs based on empty lines
        paragraphs = structured_text.strip().split('\n\n')
        return paragraphs
    
    def execute(self, requests):
        responses = []

        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            text = str(in_0.as_numpy()[0])
            try:
                with torch.inference_mode():
                    paragraphs = self.paragraph_finder(text)
                    # Here change np.object to object
                    out_tensor = pb_utils.Tensor("OUTPUT0", np.array([paragraphs]), dtype=object)
                    inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                    responses.append(inference_response)
                    print(responses)

            except Exception as e:
                error_response = pb_utils.InferenceResponse(output_tensors=[], error=str(e))
                responses.append(error_response)

        return responses