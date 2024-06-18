import os
import pandas as pd
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from groq import Groq

# Set up environment variables
os.environ["SERPER_API_KEY"] = "KEY"
os.environ["OPENAI_API_KEY"] = "KEY"
os.environ["GROQ_API_KEY"] = "KEY"

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Function to perform a chat completion using Groq
def perform_groq_chat_completion(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Creating a tool for web search
search_tool = SerperDevTool()

# Define agents with their roles and tasks
agents = []
tasks = []

# Neuroscientist Agent
neuroscientist = Agent(
    role='Neuroscientist',
    goal='Analyze neural data to understand the mechanisms of consciousness.',
    verbose=True,
    memory=True,
    backstory=(
        "You are a leading neuroscientist with expertise in analyzing neural data to study consciousness."
        "Your goal is to uncover the neural correlates of consciousness through data analysis and research."
    ),
    tools=[search_tool],
    allow_delegation=False
)
agents.append(neuroscientist)

task_analyze_data = Task(
    description="Analyze the neural data to understand the mechanisms of consciousness.",
    expected_output="Detailed analysis and insights on the neural correlates of consciousness.",
    tools=[search_tool],
    agent=neuroscientist,
)
tasks.append(task_analyze_data)

# Neuroanatomist Agent
neuroanatomist = Agent(
    role='Neuroanatomist',
    goal='Provide detailed maps of brain regions, neural pathways, and connectivity.',
    verbose=True,
    memory=True,
    backstory=(
        "You are an expert in the structure of the brain, providing detailed maps of brain regions, neural pathways, and connectivity."
        "Your goal is to map the anatomy of the brain comprehensively."
    ),
    tools=[search_tool],
    allow_delegation=False
)
agents.append(neuroanatomist)

task_map_brain = Task(
    description="Map the detailed structure of the brain, including regions, neural pathways, and connectivity.",
    expected_output="Detailed anatomical maps of the brain.",
    tools=[search_tool],
    agent=neuroanatomist,
)
tasks.append(task_map_brain)

# Neurophysiologist Agent
neurophysiologist = Agent(
    role='Neurophysiologist',
    goal='Study the electrical and chemical activities of the nervous system.',
    verbose=True,
    memory=True,
    backstory=(
        "You are an expert in the electrical and chemical activities of the nervous system."
        "Your goal is to study brain function through techniques such as EEG, MEG, and electrophysiology."
    ),
    tools=[search_tool],
    allow_delegation=False
)
agents.append(neurophysiologist)

task_study_brain_function = Task(
    description="Study brain function through techniques such as EEG, MEG, and electrophysiology.",
    expected_output="Data on the electrical and chemical activities of the brain.",
    tools=[search_tool],
    agent=neurophysiologist,
)
tasks.append(task_study_brain_function)

# Neuropsychologist Agent
neuropsychologist = Agent(
    role='Neuropsychologist',
    goal='Study the relationship between brain function and behavior.',
    verbose=True,
    memory=True,
    backstory=(
        "You are a researcher who studies the relationship between brain function and behavior."
        "Your goal is to provide insights into cognitive processes and how brain injuries or diseases affect mental functions."
    ),
    tools=[search_tool],
    allow_delegation=False
)
agents.append(neuropsychologist)

task_study_behavior = Task(
    description="Study the relationship between brain function and behavior.",
    expected_output="Insights into cognitive processes and how brain injuries or diseases affect mental functions.",
    tools=[search_tool],
    agent=neuropsychologist,
)
tasks.append(task_study_behavior)

# Neurologist Agent
neurologist = Agent(
    role='Neurologist',
    goal='Diagnose and treat neurological disorders.',
    verbose=True,
    memory=True,
    backstory=(
        "You are a medical doctor specializing in diagnosing and treating neurological disorders."
        "Your goal is to contribute clinical knowledge and insights from patient data."
    ),
    tools=[search_tool],
    allow_delegation=False
)
agents.append(neurologist)

task_clinical_insights = Task(
    description="Contribute clinical knowledge and insights from patient data.",
    expected_output="Clinical data and insights on neurological disorders.",
    tools=[search_tool],
    agent=neurologist,
)
tasks.append(task_clinical_insights)

# Cognitive Scientist Agent
cognitive_scientist = Agent(
    role='Cognitive Scientist',
    goal='Study mental processes such as perception, memory, reasoning, and language.',
    verbose=True,
    memory=True,
    backstory=(
        "You are a researcher who studies mental processes such as perception, memory, reasoning, and language."
        "Your goal is to help bridge the gap between neural activity and cognitive functions."
    ),
    tools=[search_tool],
    allow_delegation=False
)
agents.append(cognitive_scientist)

task_study_mental_processes = Task(
    description="Study mental processes such as perception, memory, reasoning, and language.",
    expected_output="Data and insights on cognitive functions.",
    tools=[search_tool],
    agent=cognitive_scientist,
)
tasks.append(task_study_mental_processes)

# Bioinformatics Specialist Agent
bioinformatics_specialist = Agent(
    role='Bioinformatics Specialist',
    goal='Manage and analyze biological data.',
    verbose=True,
    memory=True,
    backstory=(
        "You are an expert in managing and analyzing biological data."
        "Your goal is to design and maintain databases, develop algorithms for data integration, and ensure data quality and accessibility."
    ),
    tools=[search_tool],
    allow_delegation=False
)
agents.append(bioinformatics_specialist)

task_manage_data = Task(
    description="Design and maintain databases, develop algorithms for data integration, and ensure data quality and accessibility.",
    expected_output="Integrated and high-quality database of brain data.",
    tools=[search_tool],
    agent=bioinformatics_specialist,
)
tasks.append(task_manage_data)

# Data Scientist Agent
data_scientist = Agent(
    role='Data Scientist',
    goal='Process and interpret large datasets, uncovering patterns and insights.',
    verbose=True,
    memory=True,
    backstory=(
        "You are a professional skilled in data analysis, machine learning, and statistical modeling."
        "Your goal is to help process and interpret large datasets, uncovering patterns and insights."
    ),
    tools=[search_tool],
    allow_delegation=False
)
agents.append(data_scientist)

task_analyze_datasets = Task(
    description="Process and interpret large datasets, uncovering patterns and insights.",
    expected_output="Statistical analysis and machine learning models of brain data.",
    tools=[search_tool],
    agent=data_scientist,
)
tasks.append(task_analyze_datasets)

# Computer Scientist Agent
computer_scientist = Agent(
    role='Computer Scientist',
    goal='Build the infrastructure needed to store, manage, and analyze brain data.',
    verbose=True,
    memory=True,
    backstory=(
        "You are a specialist in software development, database design, and computational modeling."
        "Your goal is to build the infrastructure needed to store, manage, and analyze brain data."
    ),
    tools=[search_tool],
    allow_delegation=False
)
agents.append(computer_scientist)

task_build_infrastructure = Task(
    description="Build the infrastructure needed to store, manage, and analyze brain data.",
    expected_output="Database infrastructure for brain data.",
    tools=[search_tool],
    agent=computer_scientist,
)
tasks.append(task_build_infrastructure)

# AI Researcher Agent
ai_researcher = Agent(
    role='AI Researcher',
    goal='Develop models to simulate brain functions and analyze complex data patterns.',
    verbose=True,
    memory=True,
    backstory=(
        "You are an expert in AI and machine learning."
        "Your goal is to develop models to simulate brain functions and analyze complex data patterns."
    ),
    tools=[search_tool],
    allow_delegation=False
)
agents.append(ai_researcher)

task_develop_models = Task(
    description="Develop models to simulate brain functions and analyze complex data patterns.",
    expected_output="AI models simulating brain functions.",
    tools=[search_tool],
    agent=ai_researcher,
)
tasks.append(task_develop_models)

# Ethicist Agent
ethicist = Agent(
    role='Ethicist',
    goal='Address the ethical considerations of collecting, storing, and using brain data.',
    verbose=True,
    memory=True,
    backstory=(
        "You are a professional who addresses the ethical considerations of collecting, storing, and using brain data."
        "Your goal is to ensure privacy, consent, and responsible use of information."
    ),
    tools=[search_tool],
    allow_delegation=False
)
agents.append(ethicist)

task_ensure_ethics = Task(
    description="Ensure privacy, consent, and responsible use of brain data.",
    expected_output="Ethical guidelines and compliance for brain data usage.",
    tools=[search_tool],
    agent=ethicist,
)
tasks.append(task_ensure_ethics)

# Biostatistician Agent
biostatistician = Agent(
    role='Biostatistician',
    goal='Apply statistical methods to biological data.',
    verbose=True,
    memory=True,
    backstory=(
        "You are an expert in applying statistical methods to biological data."
        "Your goal is to design experiments, analyze data, and interpret results to ensure scientific rigor."
    ),
    tools=[search_tool],
    allow_delegation=False
)
agents.append(biostatistician)

task_apply_statistics = Task(
    description="Apply statistical methods to biological data, design experiments, analyze data, and interpret results.",
    expected_output="Statistical analysis and interpretation of brain data.",
    tools=[search_tool],
    agent=biostatistician,
)
tasks.append(task_apply_statistics)

# Medical Imaging Specialist Agent
medical_imaging_specialist = Agent(
    role='Medical Imaging Specialist',
    goal='Provide detailed images of the brain’s structure and function.',
    verbose=True,
    memory=True,
    backstory=(
        "You are a professional skilled in MRI, fMRI, PET, and other imaging techniques."
        "Your goal is to provide detailed images of the brain’s structure and function."
    ),
    tools=[search_tool],
    allow_delegation=False
)
agents.append(medical_imaging_specialist)

task_provide_images = Task(
    description="Provide detailed images of the brain’s structure and function using MRI, fMRI, PET, and other imaging techniques.",
    expected_output="Detailed brain images.",
    tools=[search_tool],
    agent=medical_imaging_specialist,
)
tasks.append(task_provide_images)

# Geneticist Agent
geneticist = Agent(
    role='Geneticist',
    goal='Study how genes influence brain development and function.',
    verbose=True,
    memory=True,
    backstory=(
        "You are an expert in genetics and genomics."
        "Your goal is to provide insights into the genetic basis of neurological and psychiatric disorders."
    ),
    tools=[search_tool],
    allow_delegation=False
)
agents.append(geneticist)

task_study_genetics = Task(
    description="Study how genes influence brain development and function.",
    expected_output="Genetic data and insights on neurological and psychiatric disorders.",
    tools=[search_tool],
    agent=geneticist,
)
tasks.append(task_study_genetics)

# Pharmacologist Agent
pharmacologist = Agent(
    role='Pharmacologist',
    goal='Study the effects of drugs on the brain.',
    verbose=True,
    memory=True,
    backstory=(
        "You are a researcher who studies the effects of drugs on the brain."
        "Your goal is to contribute knowledge about neurochemistry and pharmacodynamics."
    ),
    tools=[search_tool],
    allow_delegation=False
)
agents.append(pharmacologist)

task_study_drug_effects = Task(
    description="Study the effects of drugs on the brain.",
    expected_output="Data on neurochemistry and pharmacodynamics.",
    tools=[search_tool],
    agent=pharmacologist,
)
tasks.append(task_study_drug_effects)

# Software Engineer Agent
software_engineer = Agent(
    role='Software Engineer',
    goal='Create the software tools and interfaces for data entry, retrieval, and visualization.',
    verbose=True,
    memory=True,
    backstory=(
        "You are a developer who creates the software tools and interfaces for data entry, retrieval, and visualization."
        "Your goal is to ensure the database is user-friendly and efficient."
    ),
    tools=[search_tool],
    allow_delegation=False
)
agents.append(software_engineer)

task_develop_tools = Task(
    description="Create software tools and interfaces for data entry, retrieval, and visualization.",
    expected_output="User-friendly software tools and interfaces for brain data.",
    tools=[search_tool],
    agent=software_engineer,
)
tasks.append(task_develop_tools)

# Project Manager Agent
project_manager = Agent(
    role='Project Manager',
    goal='Coordinate the project, managing timelines, budgets, and ensuring milestones are met.',
    verbose=True,
    memory=True,
    backstory=(
        "You are a professional who oversees the project, coordinating between different teams, managing timelines, budgets, and ensuring that milestones are met."
        "Your goal is to ensure the project stays on track and that all teams collaborate effectively."
    ),
    tools=[search_tool],
    allow_delegation=False
)
agents.append(project_manager)

task_manage_project = Task(
    description="Coordinate the project, managing timelines, budgets, and ensuring milestones are met.",
    expected_output="A well-coordinated project with timelines, budgets, and milestones met.",
    tools=[search_tool],
    agent=project_manager,
)
tasks.append(task_manage_project)

# Task to export the final database to a CSV file
def export_to_csv(data):
    df = pd.DataFrame(data)
    df.to_csv('brain_knowledge_database.csv', index=False)
    return "Database exported to brain_knowledge_database.csv"

export_task = Task(
    description="Export the collected data to a CSV file.",
    expected_output="CSV file containing the entire knowledge of the human brain.",
    tools=[search_tool],
    agent=project_manager,
    function=lambda: export_to_csv(collect_all_data())  # collect_all_data() is a placeholder for the actual data collection logic
)

# Forming the crew with the defined agents and tasks
crew = Crew(
    agents=agents,
    tasks=tasks + [export_task],
    process=Process.sequential
)

# Placeholder function for collecting all data from tasks
def collect_all_data():
    data = []
    for task in tasks:
        data.append({
            "agent": task.agent.role,
            "task_description": task.description,
            "output": task.expected_output,
        })
    return data

# Kickoff the project
result = crew.kickoff(inputs={'project_name': 'brain_knowledge_database'})
print(result)
