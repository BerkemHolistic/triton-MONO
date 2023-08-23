from inference import InferenceLoader    


if __name__ == "__main__":
    

    # # successful Extractive QA
    # test_model = InferenceLoader("ExtractiveQA")
    # print(test_model.predict_SM(context='the weather is great today', question='what will the weather be like today?'))

    test_model = InferenceLoader("ParagraphFinder")
    print(test_model.paragraph_creator('If my grandma had wheels she would have been a bike.\nBut shes doesnt have wheels'))
    

    