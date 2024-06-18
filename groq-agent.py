import pandas as pd
import os
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq


def main():
    """
    Main function to initialize and run the CrewAI AI Newsletter Assistant.

    This function sets up an assistant using the Llama 3 model with the ChatGroq API.
    It provides a text-based interface for users to compile a weekly AI newsletter by interacting 
    with multiple specialized AI agents. The function outputs the results to the console 
    and writes them to a markdown file.

    Steps:
    1. Initialize the ChatGroq API with the specified model and API key.
    2. Display introductory text about the AI Newsletter Assistant.
    3. Create and configure four AI agents:
        - News_Collector_Agent: Gathers the latest AI news.
        - News_Summarizer_Agent: Summarizes the collected news.
        - Analysis_Agent: Provides analysis and insights on the news.
        - Newsletter_Compiler_Agent: Compiles the summaries and analysis into a newsletter format.
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

    print('CrewAI AI Newsletter Assistant')
    multiline_text = """
    The CrewAI AI Newsletter Assistant is designed to help you compile a weekly AI newsletter. 
    It leverages a team of AI agents, each with a specific role, to gather the latest AI news, 
    summarize the key points, provide insights, and compile everything into a newsletter format.
    """

    print(multiline_text)

    News_Collector_Agent = Agent(
        role='News_Collector_Agent',
        goal="""Gather the latest AI news from various sources, ensuring coverage of major events, 
                breakthroughs, and noteworthy research.""",
        backstory="""You are an expert in tracking down the latest and most relevant AI news. 
                Your goal is to collect up-to-date information on AI developments from reliable sources.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    News_Summarizer_Agent = Agent(
        role='News_Summarizer_Agent',
        goal="""Summarize the collected AI news, highlighting the key points and main takeaways.""",
        backstory="""You specialize in distilling large amounts of information into concise and readable summaries. 
                Your task is to make the news accessible and understandable for the newsletter readers.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Analysis_Agent = Agent(
        role='Analysis_Agent',
        goal="""Provide analysis and insights on the summarized news, offering context and expert opinions.""",
        backstory="""As an expert in AI, you provide in-depth analysis and context to help readers understand 
                the significance of the news and its potential impact on the field.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Newsletter_Compiler_Agent = Agent(
        role='Newsletter_Compiler_Agent',
        goal="""Compile the summaries and analysis into a well-structured newsletter format, ready for publication.""",
        backstory="""You are skilled in creating engaging and visually appealing newsletters. 
                Your goal is to ensure the final product is both informative and enjoyable to read.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # Define the tasks for each agent
    task_collect_news = Task(
        description="""Collect the latest AI news from various sources, ensuring comprehensive coverage 
            of major events, breakthroughs, and noteworthy research.""",
        agent=News_Collector_Agent,
        expected_output="A list of the latest AI news articles with source links."
    )

    task_summarize_news = Task(
        description="""Summarize the collected AI news, highlighting the key points and main takeaways.""",
        agent=News_Summarizer_Agent,
        expected_output="A summarized list of the latest AI news articles."
    )

    task_analyze_news = Task(
        description="""Provide analysis and insights on the summarized news, offering context and expert opinions.""",
        agent=Analysis_Agent,
        expected_output="An analysis of the summarized AI news articles with context and insights."
    )

    task_compile_newsletter = Task(
        description="""Compile the summaries and analysis into a well-structured newsletter format, ready for publication.""",
        agent=Newsletter_Compiler_Agent,
        expected_output="A compiled AI newsletter with news summaries and analysis."
    )

    # Create a crew with the agents and tasks
    crew = Crew(
        agents=[News_Collector_Agent, News_Summarizer_Agent, Analysis_Agent, Newsletter_Compiler_Agent],
        tasks=[task_collect_news, task_summarize_news, task_analyze_news, task_compile_newsletter],
        verbose=False
    )

    # Run the tasks
    result = crew.kickoff()

    # Print the results and write them to an output markdown file
    print(result)
    with open('weekly_ai_newsletter.md', "w") as file:
        print('\n\nThese results have been exported to weekly_ai_newsletter.md')
        file.write(result)


if __name__ == "__main__":
    main()
