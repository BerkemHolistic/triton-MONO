from inference import InferenceLoader
from pipeline import QuestionAnsweringSystem
from extractors import extract_text_from_pdf
from postprocessing import answer_encapsulation,riskmapping_creation
import time

    


if __name__ == "__main__":
    

    THRESHOLD_CS = 85 #0-100
    THRESHOLD_POST = 5 #0-10
    qa = QuestionAnsweringSystem(THRESHOLD_CS=THRESHOLD_CS,THRESHOLD_POST=THRESHOLD_POST)
    
    file_list = ["Sova Overview Unilever.pdf","Sova Pilot.pdf"]
    question_json = 'questions.json'
    result = qa.run_pipeline(file_list ,question_json)
    print(f"len of {len(result)}")
    print(result)
    project_result,risk_mapping_result = answer_encapsulation(result)
    
    print(riskmapping_creation(project_result,risk_mapping_result,False))
    
    # test_model = InferenceLoader("llama2_7b_chat")
    # print(test_model.predict(question="What is my name?",context = "I am Zekun Wu"))