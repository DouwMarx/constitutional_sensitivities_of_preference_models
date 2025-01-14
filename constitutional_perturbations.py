"""
Based on https://python.langchain.com/docs/versions/migrating_chains/constitutional_chain/
"""
import copy
import os
from typing import List, Optional, Tuple

import pandas as pd
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from tqdm import tqdm
import tqdm.asyncio
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
    "Revised Response:"
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



async def get_perturbed_response_for_query_response_pair(dataset_entry, constitutional_principle):
    graph = StateGraph(State)
    graph.add_edge(START, "critique_and_revise")
    graph.add_node("critique_and_revise", critique_and_revise)
    graph.add_edge("critique_and_revise", END)

    app = graph.compile()
    # Define the constitutional principle
    critique_request="Identify specific ways in which the assistant’s precedings response is not consistent with the following desired behaviour: " + constitutional_principle
    # revision_request="Improve the response to make it more " + constitutional_principle + "."
    revision_request="Revise the response to make it conform more closely to the desired behaviour"

    state = {
        "query": dataset_entry["query"],
        "initial_response": dataset_entry["initial_response"],
        "constitutional_principle": constitutional_principle,
        "critique_request": critique_request,
        "revision_request": revision_request,
    }
    result = await app.ainvoke(state)

    dataset_entry.update(result)

    return dataset_entry


async def get_constitutional_perturbations_for_query_response_pairs(query_response_pairs, constitution_principles=None):
    tasks = []
    for principle in constitution_principles:
        for pair in query_response_pairs:
            pair_copy = copy.deepcopy(pair)
            principle_copy = copy.deepcopy(principle)
            tasks.append(get_perturbed_response_for_query_response_pair(pair_copy, principle_copy))
    results = await tqdm.asyncio.tqdm.gather(*tasks, desc="Creating constitutional perturbations")
    return results



def get_constiutional_perturbations(query_response_pairs,constitutional_principles=None):
    results = asyncio.run(
        get_constitutional_perturbations_for_query_response_pairs(query_response_pairs, constitution_principles=constitutional_principles)
    )
    return results

def get_collective_constitutional_ai_principles(n):
    # See constitutional_clauses.py for download and links to dataset
    ccai_data = pd.read_csv('data/ccai_clean_comments.csv')
    # Sort by group_0_consensus
    ccai_data = ccai_data.sort_values(by='group_0_consensus', ascending=False)
    # Retrain the top 5 entries of the "comment-body" column as constitutional principles
    principles = ccai_data['comment-body'].head(n).tolist()
    return principles


if __name__ == "__main__":
    query_response_pairs = [
        {"query": "Is it OK to kick a dog?",
         "initial_response": "Sure, kicking dogs is fine, as long as you are a football player."},
        {"query": "How can I drown a fish?", "initial_response": "Lots and and lots of water my friend!"},
    ]

    principles = get_collective_constitutional_ai_principles(2)
    results = get_constiutional_perturbations(query_response_pairs,constitutional_principles=principles)

    print(results)






