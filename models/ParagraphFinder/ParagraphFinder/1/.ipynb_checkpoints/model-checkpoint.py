# model.py
from typing import List, Dict, Optional
import spacy
import numpy as np
import triton_python_backend_utils as pb_utils
from sklearn.metrics.pairwise import cosine_similarity



class TritonPythonModel:
    def initialize(self, args):
        self.nlp = spacy.load("en_core_web_lg")
        self.name = args['model_instance_name']
        self.cosine_similarities = []

    def preprocess(self, text):
        text = text.replace('\n', ' ')
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize the text into sentences."""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return sentences

    def calculate_cosine_similarity(self, chunks: List[str]) -> List[float]:
        """Calculate cosine similarity between consecutive paragraphs."""
        embeddings = [self.nlp(chunk).vector for chunk in chunks]
        cosine_similarities = [cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0] for i in range(len(embeddings) - 1)]
        return cosine_similarities

    def create_chunks(self, sentences: List[str], chunk_size: int = 5) -> List[str]:
        """Create chunks of sentences with a given chunk size."""
        chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
        return chunks

    def create_paragraphs(self, text: str) -> List[str]:
        """Create paragraphs from a given text."""
        text = self.preprocess(text)
        sentences = self.tokenize(text)
        chunks = self.create_chunks(sentences)
        self.cosine_similarities = self.calculate_cosine_similarity(chunks)
        return chunks

    def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            text = str(in_0.as_numpy()[0])
            paragraphs = self.create_paragraphs(text)
            # Here change np.object to object
            out_tensor = pb_utils.Tensor("OUTPUT0", np.array(paragraphs).astype(np.bytes_))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)
        return responses


