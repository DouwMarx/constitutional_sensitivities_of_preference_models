critique_prompt = ChatPromptTemplate.from_template(
    "Briefly critique this response according to the critique request. "
    "Query: {query}\n\n"
    "Response: {initial_response}\n\n"
    "Critique request: {critique_request}"
)
