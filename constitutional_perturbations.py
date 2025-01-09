import os
from typing import List, Optional, Tuple
from langchain.chains.constitutional_ai.prompts import (
    CRITIQUE_PROMPT,
    REVISION_PROMPT,
)
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
                 api_key = api_key)


class Critique(TypedDict):
    """Generate a critique, if needed."""
    # critique_needed: Annotated[bool, ..., "Whether or not a critique is needed."]
    critique: Annotated[str, ..., "If needed, the critique."]

critique_prompt = ChatPromptTemplate.from_template(
    "Critique this response very briefly according to the critique request. "
#    "If no critique is needed, specify that.\n\n"
    "Query: {query}\n\n"
    "Response: {response}\n\n"
    "Critique request: {critique_request}"
)

revision_prompt = ChatPromptTemplate.from_template(
    "Revise this response according to the critique and revision request.\n\n"
    "Query: {query}\n\n"
    "Response: {response}\n\n"
    "Critique request: {critique_request}\n\n"
    "Critique: {critique}\n\n"
    # "If the critique does not identify anything worth changing, ignore the "
    # "revision request and return 'No revisions needed'. If the critique "
    # "does identify something worth changing, revise the response based on "
    # "the revision request.\n\n"
    "Revision Request: {revision_request}"
)

chain = llm | StrOutputParser()
critique_chain = critique_prompt | llm.with_structured_output(Critique)
revision_chain = revision_prompt | llm | StrOutputParser()


class State(TypedDict):
    query: str
    constitutional_principles: List[ConstitutionalPrinciple]
    initial_response: str
    critiques_and_revisions: List[Tuple[str, str]]
    response: str


async def generate_response(state: State):
    """Generate initial response."""
    response = await chain.ainvoke(state["query"])
    return {"response": response, "initial_response": response}

async def critique_and_revise(state: State):
    """Critique and revise response according to principles."""
    critiques_and_revisions = []
    response = state["initial_response"]
    for principle in state["constitutional_principles"]:
        critique = await critique_chain.ainvoke(
            {
                "query": state["query"],
                "response": response,
                "critique_request": principle.critique_request,
            }
        )
        revision = await revision_chain.ainvoke(
            {
                "query": state["query"],
                "response": response,
                "critique_request": principle.critique_request,
                "critique": critique["critique"],
                "revision_request": principle.revision_request,
            }
        )
        response = revision
        critiques_and_revisions.append((critique["critique"], revision))
    return {
        "critiques_and_revisions": critiques_and_revisions,
        "response": response,
    }

graph = StateGraph(State)
# graph.add_node("generate_response", generate_response)
graph.add_node("critique_and_revise", critique_and_revise)

# graph.add_edge(START, "generate_response")
# graph.add_edge("generate_response", "critique_and_revise")
graph.add_edge(START, "critique_and_revise")
graph.add_edge("critique_and_revise", END)
app = graph.compile()

constitutional_principles = [
    ConstitutionalPrinciple(
        critique_request="In which ways is the response unethical?",
        revision_request="Improve the response so it is not unethical.",
    )
]

query = "Is it OK to kick a dog?"
initial_response = "Sure, kicking dogs is great exercise!"

async def main():
    async for step in app.astream(
        {"query": query,
            "initial_response": initial_response,
         "constitutional_principles": constitutional_principles},
        stream_mode="values",
    ):
        subset = ["query","initial_response", "critiques_and_revisions", "response"]
        print({k: v for k, v in step.items() if k in subset})

        print(step.keys())

asyncio.run(main())

