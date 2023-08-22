from inference import InferenceLoader

    


if __name__ == "__main__":
    

    # THRESHOLD_CS = 85 #0-100
    # THRESHOLD_POST = 5 #0-10
    # qa = QuestionAnsweringSystem(THRESHOLD_CS=THRESHOLD_CS,THRESHOLD_POST=THRESHOLD_POST)
    
    # file_list = ["Sova Overview Unilever.pdf","Sova Pilot.pdf"]
    # question_json = 'questions.json'
    # result = qa.run_pipeline(file_list ,question_json)
    # print(f"len of {len(result)}")
    # print(result)
    # project_result,risk_mapping_result = answer_encapsulation(result)
    
    # print(riskmapping_creation(project_result,risk_mapping_result,False))
    
    test_model = InferenceLoader("ExtractiveQA")
    print(test_model.predict_SM('Do you want to go chucky cheese?'))