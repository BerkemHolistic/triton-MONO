from tritonclient.utils import *
import tritonclient.http as httpclient
import numpy as np
import ast
import re
import codecs




class InferenceLoader:
    def __init__(self, inference_type,url_address="localhost:8000"):
        self.url_address = url_address
        self.client = httpclient.InferenceServerClient(url=self.url_address)
        self.inference_type = inference_type
    
    def create_infer_input(self,name, data):
        data_np = np.array(data, dtype=object).reshape([-1,1])
        infer_input = httpclient.InferInput(name, data_np.shape, np_to_triton_dtype(data_np.dtype))
        infer_input.set_data_from_numpy(data_np)
        return infer_input

    def process_inference(self, inputs, output_names, model_name):
        outputs = [httpclient.InferRequestedOutput(name) for name in output_names]
        try:
            return self.client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        except Exception as e:
            print(f"Error during inference: {e}")
            return None

    def sentence_embedding(self,sentence):
        if self.inference_type == "SentenceTransformer":
            input_data = self.create_infer_input("text", [sentence])
            response = self.process_inference([input_data], ["embedding"], self.inference_type)
            if response:
                return response.as_numpy("embedding")
            
    def predict_SM(self, question,context=None,option=None):
        if self.inference_type == "ExtractiveQA":
            input_question = self.create_infer_input("question", [question])
            input_context = self.create_infer_input("context", [context])
            response = self.process_inference([input_question, input_context], ["answer"], self.inference_type)
            if response:
                return [response.as_numpy("answer")[0].decode('utf-8')]
        
        elif self.inference_type == "AbstractiveQA":
            input_question = self.create_infer_input("question", [question])
            input_context = self.create_infer_input("context", [context])
            response = self.process_inference([input_question, input_context], ["answer"], self.inference_type)
            if response:
                return [response.as_numpy("answer")[0].decode('utf-8')]
            
        elif self.inference_type == "MultiQA":
            questions = [question]*len(option)
            contexts = [context]*len(option)
            input_question = self.create_infer_input("question", questions)
            input_context = self.create_infer_input("context", contexts)
            input_options = self.create_infer_input("options", option)
            response = self.process_inference([input_question, input_context, input_options], ["options", "probs"], self.inference_type)
            if response:
                original_options = np.array(response.as_numpy("options")).flatten()
                original_probs = np.array(response.as_numpy("probs")).flatten()
                options = [ast.literal_eval(item.decode('utf-8'))[0].decode('utf-8') for item in original_options]
                probs = list(original_probs.astype(float))
                return options, probs

        

    def paragraph_creator(self, text):
        if self.inference_type == "ParagraphFinder":
            input_data = self.create_infer_input("INPUT0", [text])
            response = self.process_inference([input_data], ["OUTPUT0"], self.inference_type)
            if response:
                result = np.array(response.as_numpy("OUTPUT0")).flatten()

                decoded_result = []
                for x in result:
                    # Convert string representation of bytes to actual bytes
                    x = codecs.escape_decode(x)[0]

                    # Decode with utf-8, ignore undecodable bytes
                    decoded_string = x.decode('utf-8', 'ignore')

                    # Remove non-ASCII characters
                    decoded_string = re.sub(r'[^\x00-\x7F]+', '', decoded_string)

                    decoded_result.append(decoded_string)

                return decoded_result





        
        
            
        
        
        
        
