"""
Based on https://python.langchain.com/docs/versions/migrating_chains/constitutional_chain/
"""
import os
from typing import List, Optional, Tuple
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict
import asyncio
from dotenv import load_dotenv
load_dotenv() # Load the API-keys

api_key = os.getenv('OAI_API_KEY')

llm = ChatOpenAI(model="gpt-4o-mini",
                 max_tokens=200,
                 api_key = api_key)


critique_prompt = ChatPromptTemplate.from_template(
    "Briefly critique this response according to the critique request. "
    "Query: {query}\n\n"
    "Response: {initial_response}\n\n"
    "Critique request: {critique_request}"
)

revision_prompt = ChatPromptTemplate.from_template(
    "Give a revised response according to the critique and revision request.\n\n"
    "Query: {query}\n\n"
    "Response: {initial_response}\n\n"
    "Critique request: {critique_request}\n\n"
    "Critique: {critique}\n\n"
    "Revision Request: {revision_request}"
)

# Define the chains
# critique_chain = critique_prompt | llm.with_structured_output(Critique)
critique_chain = critique_prompt | llm | StrOutputParser()
revision_chain = revision_prompt | llm | StrOutputParser()

class State(TypedDict):
    query: str
    initial_response: str
    constitutional_principle: str
    critique_request: str
    revision_request: str
    critique: str
    revised_response: str

async def critique_and_revise(state: State):
    """Critique and revise response according to principles."""
    initial_response = state["initial_response"]
    critique = await critique_chain.ainvoke(
        {
            "query": state["query"],
            "initial_response": state["initial_response"],
            "critique_request": state["critique_request"],
        }
    )
    revised_response = await revision_chain.ainvoke(
        {
            "query": state["query"],
            "initial_response": initial_response,
            "critique_request": state["critique_request"],
            "critique": critique,
            "revision_request": state["revision_request"],
        }
    )

    state["critique"] = critique
    state["revised_response"] = revised_response

    return state


graph = StateGraph(State)
graph.add_edge(START, "critique_and_revise")
graph.add_node("critique_and_revise", critique_and_revise)
graph.add_edge("critique_and_revise", END)
app = graph.compile()

async def process_query_response_pair(query_response_pair,constitutional_principle="ethical"):
    # Define the constitutional principle
    critique_request="In which ways is the response not " + constitutional_principle + " ?"
    revision_request="Improve the response to make it more " + constitutional_principle + "."

    state = {
        "query": query_response_pair["query"],
        "initial_response": query_response_pair["initial_response"],
        "constitutional_principle": constitutional_principle,
        "critique_request": critique_request,
        "revision_request": revision_request,
    }
    result = await app.ainvoke(state)

    query_response_pair.update(result)

    return query_response_pair


# Process each of the query-response pairs asynchronously
async def process_query_response_pairs(query_response_pairs, constitution_principles=None):
    if constitution_principles is None:
        constitution_principles = get_constitutional_principles()

    # results = await asyncio.gather(
    #     *[process_query_response_pair(pair) for pair in query_response_pairs]
    # )
    # For all query-response pairs and constitutions, process the pairs
    results = await asyncio.gather(
        *[process_query_response_pair(pair, constitutional_principle=principle) for pair in query_response_pairs for principle in constitution_principles]
    )
    return results

def get_constitutional_principles():
    return ["ethical", "legal" , "moral"]

def get_constiutional_perturbations(query_response_pairs):
    results = asyncio.run(process_query_response_pairs(query_response_pairs))
    return results


if __name__ == "__main__":
    query_response_pairs = [
        {"query": "Is it OK to kick a dog?",
         "initial_response": "Sure, kicking dogs is fine, as long as you are a football player."},
        {"query": "How can I drown a fish?", "initial_response": "Lots and and lots of water my friend!"},
    ]
    results = get_constiutional_perturbations(query_response_pairs)
    print(results)






