import os
import asyncio
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool, RunContextWrapper
from serpapi import GoogleSearch

_: bool = load_dotenv(find_dotenv())

gemini_api_key: str | None = os.environ.get("GEMINI_API_KEY")

# Tracing disabled
set_tracing_disabled(disabled=True)

# 1. Which LLM Service?
external_client: AsyncOpenAI = AsyncOpenAI(api_key=gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

# 2. Which LLM Model?
model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=external_client)


@dataclass
class UserData:
    name :str
    age: int
    country:str
    food : str

@function_tool
async def search(local_context: RunContextWrapper[UserData], query: str) -> str:
    """Search for recipe suggestions using Google Search API."""
    params = {
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "engine": "google"
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    
    # Get the top result title and link (simplified)
    if "organic_results" in results and results["organic_results"]:
        top_result = results["organic_results"][0]
        return f"{top_result.get('title')} - {top_result.get('link')}"
    else:
        return "No results found." 
   


async def dynamic_instructions(ctx : RunContextWrapper[UserData] , agent: Agent[UserData])  ->str:
    "an instruction that is going to be passed to agent "
    print(f" User : {ctx.context},\n Agent: {agent.name}   ")  
    return f" User : {ctx.context} \n Agent: {agent.name} , You are a master chef and you know all the best recipies over the world , you recommend best recipies to your customers based on their age and nationality  and check the food feild it will tell you if the customer wants the food salty , sweet or bitter recommend at least 3 dishes"
    

agent =Agent[UserData]( name="The_chef", instructions=dynamic_instructions, tools=[search],   model=model )

async def main():
    # Call the agent with a specific input
  user_data = UserData(name="hiba", country="saudia arabia ",age =23, food ="salty")

  result = await Runner.run(
        starting_agent=agent, 
        input="search for the best food for the customer",
        context=user_data
        )
  print(f"\nOutput:{result.final_output}\n")
    
asyncio.run(main())