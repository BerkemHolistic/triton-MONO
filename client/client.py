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

    context ='''
    Sova Assessments is a new, innovative, and cost-effective tool that helps employers identify and hire the right people for the right jobs. It is based on a combination of science, technology, and artificial intelligence. It allows employers to conduct a wide variety of tests, including live two-way and virtual interviews, and it supports a wide range of candidate types. The Sova assessment platform is designed to be accessible to a wide array of people with different needs, including people with learning disabilities, blind people, deaf people, and people with limited movement. The assessment platform also supports a number of other features, such as real-time validation of the results, machine learning, and intelligent analytics. Sova\'s four principles for diversity and inclusion are outlined. The four principles are: 1) Diversity and Inclusion, 2) Fair assessment methods, 3) Assessment format blended approach, 4) Assessment design, and 5) Ongoing optimisation. The project will be a pilot of a new assessment platform called Sova. It will provide candidates with access to the assessment platform and will be used in the selection process for new hires. The personal information collected will be anonymized and will only be used for the purpose of this project. The privacy questions asked in this document are designed to help you understand the privacy implications of this new project. You are asked to demonstrate accountability and governance. The questions ask you to provide more information about your project and activity. The first question asks you to describe the purpose and objectives of the project. Next, the questions ask about the personal data that you will collect. The answer is that it will be collected directly from the candidates. The candidates will be asked to sign up for the platform and to provide their personal information. The process will be conducted via an assessment tool called "Sova Assessment." The personal data collected is stored in the AWS/Cloud computing environment. The company will use Sova to conduct the assessment and will then use the results to develop its own assessment tools. Personal data will be retained for as long and as necessary to ensure that the process is fair and accurate. The candidate will be notified of the process and the company will obtain their consent to use their personal data for the purposes outlined in the project\'s privacy notice.
    '''
    question_insert = "How does this system benefit society?"
    question = f"The following is a multiple choice question (with answers) about {question_insert} based on context: \n{context} "



    print(test_model.predict_SM(context=context, question=question))