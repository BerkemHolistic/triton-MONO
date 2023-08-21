import json
import logging
import time

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from typing import List, Dict, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from transformers import LongformerTokenizer, LongformerForMultipleChoice
logging.basicConfig(level=logging.INFO)
from scipy.stats import entropy


class MultiQA:

    def __init__(self):
        secret_key = "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"
        self.tokenizer = LongformerTokenizer.from_pretrained("holistic-ai/multiple_choice_QA",use_auth_token=secret_key)
        self.model = LongformerForMultipleChoice.from_pretrained("holistic-ai/multiple_choice_QA",use_auth_token=secret_key)

    def prepare_answering_input(
            self,
            question,  # str
            options,   # List[str]
            context,   # str
            max_seq_length=4096,
        ):
        c_plus_q   = context + ' ' + self.tokenizer.bos_token + ' ' + question
        c_plus_q_4 = [c_plus_q] * len(options)
        tokenized_examples = self.tokenizer(
            c_plus_q_4, options,
            max_length=max_seq_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized_examples['input_ids'].unsqueeze(0)
        attention_mask = tokenized_examples['attention_mask'].unsqueeze(0)
        example_encoded = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return example_encoded


    def predict(self,question,option,context):
        inputs = self.prepare_answering_input(
            question=question,
            options=option, context=context,
        )

        outputs = self.model(**inputs)
        prob = torch.softmax(outputs.logits, dim=-1)[0].tolist()
        #selected_answer = option[np.argmax(prob)]
        return [option,prob]


logging.basicConfig(level=logging.INFO)

class QuestionAnsweringSystem:
    def __init__(self, extractive_qa_model_name="deepset/roberta-base-squad2"):
        self.extractive_qa_pipeline = pipeline("question-answering", model=extractive_qa_model_name)
        self.multi_qa_pipeline = MultiQA()
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.paragraphs = []
        self.paragraph_embeddings = []

    def create_paragraphs(self, text: str) -> None:
        """Split the provided text into paragraphs."""
        doc = self.nlp(text)
        self.paragraphs = [sent.text for sent in doc.sents]
        return self.paragraphs

    def get_similar_paragraphs(self, question: str, threshold: float = 0.1, paragraphs: Optional[List[str]] = None) -> str:
        """Return paragraphs that have cosine similarity score above the threshold with the provided question."""
        paragraphs = self.paragraphs if paragraphs is None else paragraphs
        if not paragraphs:
            raise ValueError("No paragraphs available. Please call 'create_paragraphs' method first.")

        question_embedding = self.sentence_model.encode([question])
        paragraph_embeddings = self.sentence_model.encode(paragraphs)
        similarities = cosine_similarity(question_embedding, paragraph_embeddings)[0]
        print(similarities)
        similar_para_indices = [i for i, sim in enumerate(similarities) if sim >= threshold]

        return ' '.join(paragraphs[i] for i in similar_para_indices)

    def answer_question_extractive_qa(self, context: str, question: str) -> Dict[str, str]:
        """Answer the question using extractive QA pipeline."""
        if not context:
            raise ValueError("Context is empty. Please provide a valid context.")
        answer = self.extractive_qa_pipeline(context=context, question=question)
        return answer

    def answer_question_multi_qa(self, context: str, question: str, options: List[str]) -> Dict[str, str]:
        """Answer the question using multi QA pipeline."""
        if not context:
            raise ValueError("Context is empty. Please provide a valid context.")
        answer = self.multi_qa_pipeline.predict(context=context, question=question, option=options)
        return answer

    def post_processing(self, answer: List, threshold: int = 5) -> List:
        """Post process the answer to select best ones based on entropy."""
        option = np.array(answer[0])
        probs = np.array(answer[1])

        ent = entropy(probs)
        threshold_new = ent / threshold
        selected_answers = option[probs > threshold_new]
        return selected_answers if selected_answers.size else [option[np.argmax(probs)]]

    def process_question(self, question: Dict[str, str]) -> Dict[str, str]:
        """Process a single question."""
        try:
            current_question = question['question']
            current_option = question['options']
            current_type = question['type']

            search_str_multi = current_question + str(current_option)
            selected_text = self.get_similar_paragraphs(search_str_multi)
            answer_multi = self.answer_question_multi_qa(context=selected_text, question=current_question, options=current_option)
            if current_type == "multiple":
                return_answer = self.post_processing(answer_multi)
            else:
                return_answer = [current_option[np.argmax(answer_multi[1])]]

            return {'question': current_question, 'answer': list(return_answer)}

        except ValueError as e:
            logging.error(f"ValueError occurred while processing question: {question}, error: {e}")
            return None

    def process_questions(self, questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Process a list of questions."""
        result = [self.process_question(x) for x in questions if "options" in x.keys() and "question" in x.keys()]
        return [res for res in result if res is not None]


    def load_questions(self,address= "questions.json"):
        with open(address) as f:
            questions = json.load(f)
        return questions
    def example_pipeline(self):
        questions = self.load_questions()

        example_text = '''
        The world of Human Resources (HR) is a domain of perpetual dynamics, characterized by continuous interactions, decision-making, and evaluation. Within this multifaceted milieu, a prominent embodiment of Artificial Intelligence (AI) has emerged as a game-changer: meet "Emplify", a state-of-the-art AI-powered employee engagement and performance management platform.

        Emplify has been engineered to masterfully merge AI capabilities with HR processes, focusing on enhancing employee engagement, facilitating performance management, and informing data-driven HR decisions. By leveraging natural language processing (NLP), machine learning, and predictive analytics, Emplify enables a nuanced understanding of employee sentiments, aspirations, and behavior patterns, thereby enriching the HR management landscape.

        At its core, Emplify performs comprehensive analysis of both qualitative and quantitative employee data. The AI system swiftly scans through a plethora of data sources - including surveys, performance evaluations, and digital communications. It goes beyond just number crunching and recognizes patterns, gauges sentiment, and interprets tacit knowledge concealed within unstructured data.

        In the sphere of performance management, Emplify's AI has been harnessed to create a fair, objective, and consistent evaluation system. Its predictive analytics capability aids in identifying future performance trends and potential employee churn, enabling HR teams to proactively strategize and respond.

        Moreover, Emplify learns continuously, adapting and improving its analysis and recommendations based on user interactions and feedback. This continual learning embodies the concept of AI’s autodidactic capabilities, allowing for a highly personalized and evolving user experience.

        However, in the midst of this technological breakthrough, it is crucial to consider the ethical implications of integrating AI in HR, particularly in regards to data privacy and algorithmic fairness. Unbiased algorithms, data security, and respectful employee privacy are paramount considerations that must be diligently maintained.

        In summary, AI systems such as Emplify represent the dawn of a new era in Human Resources, one that harmonizes human decision-making with AI's analytical prowess. These systems pose significant potential, ushering in transformative changes in the HR landscape, while simultaneously presenting new challenges to be navigated responsibly and ethically.
        '''

        qa_system = QuestionAnsweringSystem()
        qa_system.create_paragraphs(example_text)
        relevant_questions = questions['questionnaire'][0:2]

        start_time = time.time()

        result = qa_system.process_questions(relevant_questions)

        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_question = total_time / len(relevant_questions)

        logging.info(f"{total_time} in total, {avg_time_per_question} for each inference")

        print(result)

if __name__ == '__main__':
    qa = QuestionAnsweringSystem()
    qa.example_pipeline()