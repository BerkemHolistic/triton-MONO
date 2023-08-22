import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering,pipeline
import triton_python_backend_utils as pb_utils
from typing import List

class TritonPythonModel:
    def initialize(self, args):
        model_name = "deepset/roberta-base-squad2"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)


    def execute(self, requests):
        responses = []
        for request in requests:
            questions = pb_utils.get_input_tensor_by_name(request, "question").as_numpy()
            questions = [str(question[0]) for question in questions]  # Convert to string before passing to tokenizer

            contexts = pb_utils.get_input_tensor_by_name(request, "context").as_numpy()
            contexts = [str(context[0]) for context in contexts]  # Convert to string before passing to tokenizer


            try:
#                 with torch.inference_mode():
#                     inputs = self.tokenizer(questions, contexts, return_tensors='pt', padding=True, truncation=True).to(self.device)

#                     outputs = self.model(**inputs)

#                     # Get the most probable answer
#                     answer_start_scores = outputs.start_logits
#                     answer_end_scores = outputs.end_logits
#                     answer_start = torch.argmax(answer_start_scores)  
#                     answer_end = torch.argmax(answer_end_scores) + 1 
#                     answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
                with torch.inference_mode():
                    answer = self.nlp({'question':questions,'context':contexts})
                    response = pb_utils.Tensor("answer", np.array([answer[0]["answer"]], dtype=object))
                    inference_response = pb_utils.InferenceResponse(output_tensors=[response])
                    responses.append(inference_response)

            except Exception as e:
                error_response = pb_utils.InferenceResponse(output_tensors=[], error=str(e))
                responses.append(error_response)

        return responses


