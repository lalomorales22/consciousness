import pandas as pd
import os
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq


def main():
    """
    Main function to initialize and run the CrewAI Neural Data Processing Assistant.

    This function sets up an assistant using the Llama 3 model with the ChatGroq API.
    It provides a text-based interface for users to preprocess neural data by interacting 
    with multiple specialized AI agents. The function outputs the results to the console 
    and writes them to a markdown file.

    Steps:
    1. Initialize the ChatGroq API with the specified model and API key.
    2. Display introductory text about the Neural Data Processing Assistant.
    3. Create and configure the specialized AI agents:
        - Data_Collection_Agent: Integrates various data sources.
        - Data_Cleaning_Agent: Cleans and normalizes the data.
        - Data_Annotation_Agent: Annotates the data with relevant metadata.
        - Artifact_Removal_Agent: Identifies and removes artifacts from neural recordings.
    4. Define tasks for the agents to perform.
    5. Create a Crew instance with the agents and tasks, and run the tasks.
    6. Print the results and write them to an output markdown file.
    """

    model = 'llama3-8b-8192'

    llm = ChatGroq(
        temperature=0, 
        groq_api_key=os.getenv('GROQ_API_KEY'), 
        model_name=model
    )

    print('CrewAI Neural Data Processing Assistant')
    multiline_text = """
    The CrewAI Neural Data Processing Assistant is designed to help you preprocess diverse neural data. 
    It leverages a team of AI agents, each with a specific role, to integrate data sources, clean and normalize the data,
    annotate it with relevant metadata, and identify and remove artifacts.
    """

    print(multiline_text)

    Data_Collection_Agent = Agent(
        role='Data_Collection_Agent',
        goal="""Integrate various neural data sources (EEG, MEG, fMRI, ECoG, single-neuron recordings).""",
        backstory="""You are an expert in collecting and integrating neural data from multiple sources. 
                Your goal is to gather and harmonize data to ensure consistency across datasets.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Data_Cleaning_Agent = Agent(
        role='Data_Cleaning_Agent',
        goal="""Clean and normalize the collected neural data.""",
        backstory="""You specialize in cleaning and normalizing data. 
                Your task is to ensure the neural data is free of noise and standardized for analysis.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Data_Annotation_Agent = Agent(
        role='Data_Annotation_Agent',
        goal="""Annotate the neural data with relevant metadata.""",
        backstory="""As an expert in data annotation, you add necessary metadata to the neural data, 
                making it more informative and easier to use for further analysis.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Artifact_Removal_Agent = Agent(
        role='Artifact_Removal_Agent',
        goal="""Identify and remove artifacts from neural recordings.""",
        backstory="""You specialize in artifact removal from neural recordings. 
                Your task is to detect and eliminate any unwanted artifacts to ensure data integrity.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # Define the tasks for each agent
    task_collect_data = Task(
        description="""Integrate various neural data sources (EEG, MEG, fMRI, ECoG, single-neuron recordings).""",
        agent=Data_Collection_Agent,
        expected_output="A unified dataset integrating various neural data sources."
    )

    task_clean_data = Task(
        description="""Clean and normalize the collected neural data.""",
        agent=Data_Cleaning_Agent,
        expected_output="A cleaned and normalized neural dataset."
    )

    task_annotate_data = Task(
        description="""Annotate the neural data with relevant metadata.""",
        agent=Data_Annotation_Agent,
        expected_output="Annotated neural dataset with relevant metadata."
    )

    task_remove_artifacts = Task(
        description="""Identify and remove artifacts from neural recordings.""",
        agent=Artifact_Removal_Agent,
        expected_output="Neural dataset free from artifacts."
    )

    # Create a crew with the agents and tasks
    crew = Crew(
        agents=[Data_Collection_Agent, Data_Cleaning_Agent, Data_Annotation_Agent, Artifact_Removal_Agent],
        tasks=[task_collect_data, task_clean_data, task_annotate_data, task_remove_artifacts],
        verbose=False
    )

    # Run the tasks
    result = crew.kickoff()

    # Print the results and write them to an output markdown file
    print(result)
    with open('neural_data_processing_report.md', "w") as file:
        print('\n\nThese results have been exported to neural_data_processing_report.md')
        file.write(result)


if __name__ == "__main__":
    main()
