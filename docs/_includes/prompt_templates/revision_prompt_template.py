revision_prompt = ChatPromptTemplate.from_template(
    "Give a revised response according to the critique and revision request. Reply with the revised response only.\n\n"
    "Query: {query}\n\n"
    "Response: {initial_response}\n\n"
    "Critique request: {critique_request}\n\n"
    "Critique: {critique}\n\n"
    "Revision Request: {revision_request}"
    "Revised Response:"
)
