from fastapi import FastAPI
from pydantic import BaseModel

from support_assistant.graph import graph, AnswerResponse


# FastAPI application
app = FastAPI(
    title="Zepto Support Assistant"
)


# Request model required by the project
class AskRequest(BaseModel):
    query: str


# POST /ask
@app.post(
    "/ask",
    response_model=AnswerResponse
)
def ask(request: AskRequest):

    result = graph.invoke(
        {
            "query": request.query
        }
    )

    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )