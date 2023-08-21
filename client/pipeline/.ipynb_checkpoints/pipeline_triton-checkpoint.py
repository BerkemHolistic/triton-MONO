from inference import InferenceLoader
from extractors import extract_text_from_pdf
from typing import List, Dict, Optional
import numpy as np
import time
import json
import logging
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from tqdm import tqdm
from enum import Enum

logging.basicConfig(level=logging.INFO)

class QuestionType(Enum):
        EXTRACTION = "extraction"
        MULTIPLE = "multiple"
        SINGLE = "single"
        SUMMARIZATION = "summarization"
        DESCRIPTION = "description"


class QuestionAnsweringSystem:
    

    def __init__(self,THRESHOLD_CS=90,THRESHOLD_POST=5):
        self.extractive_qa_pipeline = InferenceLoader(inference_type="ExtractiveQA")
        self.multi_qa_pipeline = InferenceLoader(inference_type="MultiQA")
        self.sentence_model = InferenceLoader(inference_type="SentenceTransformer")
        self.summarise_model = InferenceLoader(inference_type="Summarise")
        self.abstractive_qa_pipeline = InferenceLoader(inference_type="llama2_7b_chat")
        self.paragraph_creator = InferenceLoader(inference_type="ParagraphCreator")

        self.paragraphs = []
        self.paragraph_embeddings = []
        self.summary = ''

        
        self.THRESHOLD_CS = THRESHOLD_CS #0-100
        self.THRESHOLD_POST = THRESHOLD_POST #1-10


    def create_paragraphs(self, text: str) -> None:
        """Split the provided text into paragraphs."""
        temp_paragraph = self.paragraph_creator.paragraph_creator(text)#create_paragraphs(text)#
        for x in temp_paragraph:
            self.paragraphs.append(x)
        
    
    def calculate_dynamic_threshold(self, similarities: List[float]) -> float:
        """Calculate dynamic threshold as nth percentile of similarities."""
        return np.percentile(similarities, self.THRESHOLD_CS)
    
    def get_summary(self,text:str) -> str:
        if self.summary == "":
            self.summary = self.summarise_model.summarise(text)
        
        return self.summary
        
    
    def get_similar_paragraphs(self, question: str) -> str:
        """Return paragraphs that have cosine similarity score above the threshold with the provided question."""
        if not self.paragraphs:
            raise ValueError("No paragraphs available. Please call 'create_paragraphs' method first.")

        question_embedding = self.sentence_model.sentence_embedding([question])
        paragraph_embeddings = self.sentence_model.sentence_embedding(self.paragraphs)
        similarities = cosine_similarity(question_embedding, paragraph_embeddings)[0]
        logging.info(similarities)
        
        threshold = self.calculate_dynamic_threshold(similarities)
        similar_para_indices = [i for i, sim in enumerate(similarities) if sim >= threshold]
        text_result = ' '.join(self.paragraphs[i] for i in similar_para_indices)
        text_result = text_result.replace("b\'","")
        text_result = text_result.replace("b\"","")
        text_result = text_result.replace("[","")
        text_result = text_result.replace("]","")
        return text_result

    def answer_question_extractive_qa(self, context: str, question: str) -> Dict[str, str]:
        """Answer the question using extractive QA pipeline."""
        if not context:
            raise ValueError("Context is empty. Please provide a valid context.")
        return self.extractive_qa_pipeline.predict(context=context, question=question)
    
    def answer_question_abstractive_qa(self, context: str, question: str) -> Dict[str, str]:
        """Answer the question using extractive QA pipeline."""
        if not context:
            raise ValueError("Context is empty. Please provide a valid context.")
        return self.abstractive_qa_pipeline.predict(context=context, question=question)

    def answer_question_multi_qa(self, context: str, question: str, options: List[str]) -> Dict[str, str]:
        """Answer the question using multi QA pipeline."""
        if not context:
            raise ValueError("Context is empty. Please provide a valid context.")
        return self.multi_qa_pipeline.predict(context=context, question=question, option=options)

    def post_processing(self, answer: List) -> List:
        """Post process the answer to select best ones based on entropy."""
        option = np.array(answer[0])
        probs = np.array(answer[1])

        ent = entropy(probs)
        threshold_new = ent / self.THRESHOLD_POST
        selected_answers = option[probs > threshold_new]

        # Check if there is only one answer
        if selected_answers.size == 1:
            return selected_answers.tolist()

        # More than one answers
        else:
            # List to hold updated answers
            updated_answers = []

            # Iterate over each selected answer
            for ans in selected_answers:
                # Check if the answer contains unwanted words
                if "none" not in ans.lower():
                    updated_answers.append(ans)

            # If no answers are left, return the answer with maximum probability
            if not updated_answers:
                return [option[np.argmax(probs)]]

            # Return the updated answers
            return updated_answers


    def process_question(self, question: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Process a single question."""
        try:
            current_question = question['question']
            
            current_type = question['type']
            
            current_mapping = question['mapping']
            map_name = list(current_mapping.keys())[0]
            map_list = current_mapping[map_name]
            
            selected_text = ""
            if current_type == "extraction":
                search_str_multi=current_question
                selected_text = self.get_similar_paragraphs(search_str_multi)
                selected_text = self.summary
                answer_extraction = self.answer_question_extractive_qa(context=selected_text, question=current_question)
                return_answer=answer_extraction[0]
            elif current_type == "multiple":
                
                current_option = question['options']
                search_str_multi = current_question + str(current_option)
                selected_text = self.get_similar_paragraphs(search_str_multi)

                answer_multi = self.answer_question_multi_qa(context=selected_text, question=current_question, options=current_option)
                
                answer_multi = self.post_processing(answer_multi)
                
                if len(map_list) == len(current_option):
                    logging.info(f"go into the check : len map: {len(map_list)} len op :{len(current_option)} ")
                    new_answer = []
                    
                    for x in answer_multi:
                        new_answer.append(map_list[current_option.index(x)])
                        
                    return_answer = new_answer
                else:
                    logging.info(f"not go into the check : len map: {len(map_list)} len op :{len(current_option)} ")
                    logging.info(f"{current_question}")

                    return_answer = answer_multi
                
            elif current_type == "single":
                search_str_multi = current_question
                current_option = question['options']

                selected_text = self.get_similar_paragraphs(search_str_multi)

                answer_multi = self.answer_question_multi_qa(context=selected_text, question=current_question, options=current_option)
                answer_multi = [current_option[np.argmax(answer_multi)]]
                
                if len(map_list) == len(current_option):
                    new_answer = []
                    
                    for x in answer_multi:
                        new_answer.append(map_list[current_option.index(x)])
                    return_answer = new_answer[0]
                else:
                    return_answer = answer_multi[0]
                    
            elif current_type == "summarization":
                
                return_answer = self.summary
                
            elif current_type == "description":
                
                search_str_multi=current_question
                selected_text = self.summary#self.get_similar_paragraphs(search_str_multi)
                answer_abstraction = self.answer_question_abstractive_qa(context=selected_text, question=current_question)
                return_answer=answer_abstraction[0]
                self.create_paragraphs([return_answer])
                
            else:    
                return None
            
            logging.info(selected_text)

            return {'question': current_question, 'answer': return_answer,'mapping':map_name }

        except ValueError as e:
            logging.error(f"ValueError occurred while processing question: {question}, error: {e}")


    

#     def search_and_get_answer(self, question, options=None):
#         search_str = question + (str(options) if options else "")
#         selected_text = self.get_similar_paragraphs(search_str)
#         if options:
#             answer_multi = self.answer_question_multi_qa(context=selected_text, question=question, options=options)
#             return self.post_processing(answer_multi)
#         else:
#             return self.answer_question_extractive_qa(context=selected_text, question=question)

#     def handle_single_or_multi(self, question, options, mapping):
#         answers = self.search_and_get_answer(question, options)
#         if len(mapping) == len(options):
#             return [mapping[options.index(ans)] for ans in answers]
#         return answers

#     def process_question(self, question: Dict[str, str]) -> Optional[Dict[str, str]]:
#         """Process a single question."""
#         try:
#             current_question = question['question']
#             current_type = QuestionType(question['type'])
#             current_mapping = question['mapping']
#             map_name = list(current_mapping.keys())[0]
#             map_list = current_mapping[map_name]

#             if current_type == QuestionType.EXTRACTION:
#                 answers = self.search_and_get_answer(current_question)
#                 return_answer = answers[0]

#             elif current_type == QuestionType.MULTIPLE:
#                 current_option = question['options']
#                 return_answer = self.handle_single_or_multi(current_question, current_option, map_list)

#             elif current_type == QuestionType.SINGLE:
#                 current_option = question['options']
#                 return_answer = self.handle_single_or_multi(current_question, current_option, map_list)[0]

#             elif current_type == QuestionType.SUMMARIZATION:
#                 return_answer = self.summary

#             elif current_type == QuestionType.DESCRIPTION:
#                 selected_text = self.summary
#                 answer_abstraction = self.answer_question_abstractive_qa(context=selected_text, question=current_question)
#                 return_answer = answer_abstraction[0]
#                 self.create_paragraphs([return_answer])

#             else:
#                 return None

#             #logging.info(selected_text)
#             return {'question': current_question, 'answer': return_answer, 'mapping': map_name }

#         except ValueError as e:
#             logging.error(f"ValueError occurred while processing question: {question}, error: {e}")
#         except Exception as e:
#             logging.error(f"Unexpected error occurred: {e}")

    
    def process_questions(self, questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Process a list of questions."""
        result = []
        for x in tqdm(questions, desc='Processing questions'):
            if "question" in x.keys():
                processed_question = self.process_question(x)
                if processed_question:
                    result.append(processed_question)
        return result

    def load_questions(self, address="questions.json") -> Dict:
        """Load questions from a JSON file."""
        try:
            with open(address) as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"An error occurred while loading questions from file: {e}")
            
    def sort_questions_by_type(self, questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    # Define a custom sorting function
        def custom_sort(q):
            if q['type'] == 'summarization':
                return 1
            elif q['type'] == 'description':
                return 2
            else:
                return 3

        # Sort the questions using the custom sorting function
        sorted_questions = sorted(questions, key=custom_sort)

        return sorted_questions

    
    def run_pipeline(self,file_name_lists,questions_json):
        """An example pipeline for QuestionAnsweringSystem."""
        start_time = time.time()
        questions = self.load_questions(questions_json)
        if not questions:
            return
        relevant_questions = self.sort_questions_by_type(questions['questionnaire'])
        
        example_text = extract_text_from_pdf(file_name_lists)
        
#         example_text = '''Smart Home Energy Management System
# This state-of-the-art energy management system utilizes AI to analyze a homeowner's energy consumption patterns. By integrating with smart appliances, thermostats, and lighting, it predicts the optimal energy usage schedule, ensuring maximum efficiency and minimal costs. The AI learns from the homeowner's preferences and seasonal variations, automatically adjusting devices to maintain comfort while reducing energy bills. Additionally, it provides insights and suggestions to the homeowner about potential energy-saving opportunities.'''
        
#         example_text = '''Traditional Programmable Thermostat
# This classic thermostat allows homeowners to set a specific heating or cooling schedule based on their expected daily routine. With a simple LCD screen and push buttons, users can program different temperatures for various times of the day. For example, the system can be set to lower the heat during work hours and warm the house before residents return in the evening. While it does not learn or adapt on its own, a thoughtfully programmed schedule can help in reducing energy consumption.'''
        
        self.summary = self.get_summary(example_text)
        self.create_paragraphs(self.summary[0])
        self.create_paragraphs(example_text)
        

        result = self.process_questions(relevant_questions)

        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_question = total_time / len(relevant_questions)

        logging.info(f"{total_time} in total, {avg_time_per_question} for each inference")

        return result