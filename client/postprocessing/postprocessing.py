from haisdk.assessment import run_risk_mapping
from haisdk.config import Config

def print_report(data):
    for key, value in data.items():
        if isinstance(value, list):
            value = ', '.join(value)
        print(f"{key}: {value}\n")

def answer_encapsulation(answer):
    # Define default values
    project = {
        "title": "",
        "description": "",
        "markets": ["ANZ"],
        "divisions": [],
        "businessGroups": [],
        "subdivisions": [],
        "brands": [],
        "operationalDetailsAnswers": {
            "developmentAndDeliveryHeader": "",
            "projectComponentsAccessLevel": ["pcalidk"],
            "externalComponentsCommissionType": [],
            "supplierNames": [],
            "estimatedUpfrontCostCurrency": "gbp",
            "estimatedUpfrontCost": "noUpfrontCost",
            "ongoingAnnualCostCurrency": "gbp",
            "ongoingAnnualCost": "noOngoingAnnual",
            "expectedPeriodProjectDeploymentIdk": ["epdidk"],
            "expectedBenifitsProjectToDeliver": ["ebpdidk"],
            "projectStage": {
                "stageStatus": "unknown",
                "expectedDevelopmentStart": "",
                "expectedDeployment": "",
            },
            "projectDevelopmentStatus": ["Internal vs. External Development Unknown"],
        },
        "formCorrespondentName": "Umar Mohammed",
        "formCorrespondentEmail": "umar.mohammed@holisticai.com",
        "formCorrespondentRole": "TBD",
        "formCorrespondentRights": "edit",
        "projectOwnerName": "Umar Mohammed",
        "projectOwnerEmail": "umar.mohammed@holisticai.com",
        "projectOwnerRole": "TBD",
        "projectOwnerRights": "edit",
        "copyProjectOwnerFromFormCorrespondent": True,
        "impactedProjects": [],
        "team": [],
        "customerExperience": False,
        "employeeExperience": False,
        "commercialExperience": False,
        "productManagementExperience": False,
        "dataAiExperience": False,
        "technologyExperience": False,
        "auditFunction": False,
        "communicationsFunction": False,
        "customerDevelopmentFunction": False,
        "dataAnalyticsFunction": False,
        "generalManagementFunction": False,
        "financeFunction": False,
        "informationTechnologyFunction": False,
        "humanResourcesFunction": False,
        "operationsFunction": False,
        "legalFunction": False,
        "supplyChainFunction": False,
        "researchDevelopmentFunction": False,
        "marketingCmiFunction": False
    }

    risk_mapping = {
        "enterpriseUseCases": [],
        "soceityLevel": [],
        "detailSystemUseCase": "",
        "externalParties": "",
        "whoAreExternalParties": [],
        "describeOtherExternalParties": "",
        "whoAreInternalParties": [],
        "describeOtherInternalParties": "",
        "clearCommunicatedAutomatedSystem": "",
        "howIsUserInformedAboutAutomatedSystem": [],
        "explainUserInformedAboutAutomedSystemOther": "",
        "mechanismsIfUserChallengesDecisions": [],
        "explainMechanismsIfUserChallengesDecisionsOther": "",
        "trainingMethods": [],
        "explainTrainingMethodsOther": "",
        "howWidelyDeployed": "",
        "systemScopeDetails": "",
        "dataType": [],
        "dataSource": [],
        "explainDataSourceOther": "",
        "dataCategory": [],
        "explainDataCategoryOther": "",
        "dataSubjectsAbleToOptOut": "",
        "informedOfOptOutMechanism": [],
        "explainInformedOfOptOutMechanismOther": "",
        "deletionOfPersonalData": "",
        "detailDataUsed": "",
        "autonomy": "",
        "howEssentialAutonomous": "",
        "howEssentialHumanInTheLoop": "",
        "damageCausedIfSubstantialFailure": "",
        "damageCausedIfMarginalFailure": "",
        "explainImpactOfUnderPerformance": "",
        "capabilities": [],
        "explainCapabilitiesOther": "",
        "tasksPerformedBySystem": [],
        "explainTasksPerformedBySystemOther": "",
        "outputGenerated": [],
        "explainOutputGeneratedOther": "",
        "selfAdapting": "",
        "canUsersOptOut": "",
        "explainTechnologicalCapabilities": "",
    }

    for y in answer:
        question = y['question']
        mapping = y['mapping']
        answer_small = y['answer']

        if mapping == "function" or mapping == "experience":
            for j in answer_small:
                if j in list(project.keys()):
                    project[j] = True
        elif mapping == "title" or mapping == "description":
            project[mapping] = answer_small
        else:
            if mapping in risk_mapping:  # Check if the mapping exists in the risk_mapping
                risk_mapping[mapping] = answer_small

    return project, risk_mapping
    

def riskmapping_creation(project_info,risk_mapping_info,return_api):
    config = {
    "clientId": "unilever",
    "key": "PWhFuKJb2SQ5Wn9dDzrlBwaf4CiPQx09",
    "api": "api-sdk-staging-unilever-v1.holisticai.io",
    }
    
    print(f"config dict: {config}")
    #print_report(config)
    print(f"project dict: {project_info}")
    #print_report(project_info)
    print(f"risk mapping dict: {risk_mapping_info}")
    #print_report(risk_mapping_info)
    
    if return_api == True:
        run_risk_mapping(config=config, project=project_info, risk_mapping=risk_mapping_info)