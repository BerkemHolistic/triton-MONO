from inference import InferenceLoader    


if __name__ == "__main__":
    

    # # successful Extractive QA
    # test_model = InferenceLoader("ExtractiveQA")
    # print(test_model.predict_SM(context='the weather is great today', question='what will the weather be like today?'))

    test_model = InferenceLoader("ParagraphFinder")

    example_text =''' "The world of Human Resources (HR) is a domain of perpetual dynamics, characterized by continuous interactions, decision-making, and evaluation. Within this multifaceted milieu, a prominent embodiment of Artificial Intelligence (AI) has emerged as a game-changer: meet "Emplify", a state-of-the-art AI-powered employee engagement and performance management platform.

            Emplify has been engineered to masterfully merge AI capabilities with HR processes, focusing on enhancing employee engagement, facilitating performance management, and informing data-driven HR decisions. By leveraging natural language processing (NLP), machine learning, and predictive analytics, Emplify enables a nuanced understanding of employee sentiments, aspirations, and behavior patterns, thereby enriching the HR management landscape.

            At its core, Emplify performs comprehensive analysis of both qualitative and quantitative employee data. The AI system swiftly scans through a plethora of data sources - including surveys, performance evaluations, and digital communications. It goes beyond just number crunching and recognizes patterns, gauges sentiment, and interprets tacit knowledge concealed within unstructured data.

            In the sphere of performance management, Emplify's AI has been harnessed to create a fair, objective, and consistent evaluation system. Its predictive analytics capability aids in identifying future performance trends and potential employee churn, enabling HR teams to proactively strategize and respond.

            Moreover, Emplify learns continuously, adapting and improving its analysis and recommendations based on user interactions and feedback. This continual learning embodies the concept of AI’s autodidactic capabilities, allowing for a highly personalized and evolving user experience.

            However, in the midst of this technological breakthrough, it is crucial to consider the ethical implications of integrating AI in HR, particularly in regards to data privacy and algorithmic fairness. Unbiased algorithms, data security, and respectful employee privacy are paramount considerations that must be diligently maintained.

            In summary, AI systems such as Emplify represent the dawn of a new era in Human Resources, one that harmonizes human decision-making with AI's analytical prowess. These systems pose significant potential, ushering in transformative changes in the HR landscape, while simultaneously presenting new challenges to be navigated responsibly and ethically. '''
                
    print(test_model.paragraph_creator(str(example_text)))