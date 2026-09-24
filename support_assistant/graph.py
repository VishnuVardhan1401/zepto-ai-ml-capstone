import json
import os
from typing import Literal

from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict, NotRequired

from langgraph.graph import StateGraph, START, END

from support_assistant.retrieval import retrieve
from support_assistant.prompt import build_prompt


MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"


POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


class AssistantState(TypedDict):
    query: str

    intent: NotRequired[
        Literal[
            "policy_question",
            "general_question"
        ]
    ]

    retrieved: NotRequired[list]

    answer: NotRequired[str]

    sources: NotRequired[list[str]]

    confidence: NotRequired[float]


def classify_intent(
    state: AssistantState
):

    query = state["query"].lower()

    if MOCK_LLM:

        if any(
            keyword in query
            for keyword in POLICY_KEYWORDS
        ):
            return {
                "intent": "policy_question"
            }

        return {
            "intent": "general_question"
        }

    prompt = f"""
Classify the following query as exactly one:

policy_question
general_question

Return only JSON.

Example:
{{"intent": "policy_question"}}

Query:
{state["query"]}
""".strip()

    raw = call_real_llm(prompt)

    data = json.loads(raw)

    intent = data.get("intent")

    if intent not in [
        "policy_question",
        "general_question"
    ]:
        raise ValueError(
            "Invalid intent returned by LLM."
        )

    return {
        "intent": intent
    }


def retrieve_and_answer(
    state: AssistantState
):

    results = retrieve(
        state["query"],
        k=3
    )

    source_ids = [
        item["id"]
        for item in results
    ]

    if MOCK_LLM:

        top_chunk_snippet = (
            results[0]["document"][:200].strip()
        )

        response = AnswerResponse(
            answer=(
                "Based on the retrieved context: "
                + top_chunk_snippet
            ),
            sources=source_ids,
            confidence=1.0
        )

        return {
            "retrieved": results,
            "answer": response.answer,
            "sources": response.sources,
            "confidence": response.confidence
        }

    context = "\n\n".join(
        f"[{item['id']}] {item['document']}"
        for item in results
    )

    response = generate_real_answer(
        query=state["query"],
        context=context,
        source_ids=source_ids
    )

    return {
        "retrieved": results,
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence
    }


def direct_answer(
    state: AssistantState
):

    if MOCK_LLM:

        response = AnswerResponse(
            answer=(
                "I can only answer questions "
                "about Zepto policies right now."
            ),
            sources=[],
            confidence=1.0
        )

        return {
            "answer": response.answer,
            "sources": response.sources,
            "confidence": response.confidence
        }

    response = generate_real_answer(
        query=state["query"],
        context="No Zepto policy context was retrieved.",
        source_ids=[]
    )

    return {
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence
    }


def route_by_intent(
    state: AssistantState
) -> Literal[
    "retrieve_and_answer",
    "direct_answer"
]:

    if state["intent"] == "policy_question":
        print("ROUTE: retrieve_and_answer")
        return "retrieve_and_answer"

    print("ROUTE: direct_answer")
    return "direct_answer"

def call_real_llm(
    prompt: str
):

    from groq import Groq

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    model_name = os.getenv(
        "GROQ_MODEL"
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required when MOCK_LLM=0."
        )

    if not model_name:
        raise RuntimeError(
            "GROQ_MODEL is required when MOCK_LLM=0."
        )

    client = Groq(
        api_key=api_key
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return (
        response.choices[0]
        .message
        .content
        or ""
    )


def generate_real_answer(
    query: str,
    context: str,
    source_ids: list[str]
):

    correction = ""

    for attempt in range(3):

        prompt = build_prompt(
            query=query,
            context=context,
            correction=correction
        )

        raw = call_real_llm(
            prompt
        )

        try:

            return AnswerResponse.model_validate_json(
                raw
            )

        except (
            ValidationError,
            ValueError,
            json.JSONDecodeError
        ) as error:

            correction = (
                "Your previous output failed validation. "
                "Return only valid JSON with exactly "
                "answer, sources, and confidence. "
                f"Validation error: {error}"
            )

    return AnswerResponse(
        answer=(
            "ERROR: The real LLM response "
            "could not be validated after "
            "3 attempts."
        ),
        sources=source_ids,
        confidence=0.0
    )


builder = StateGraph(
    AssistantState
)


builder.add_node(
    "classify_intent",
    classify_intent
)

builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

builder.add_node(
    "direct_answer",
    direct_answer
)


builder.add_edge(
    START,
    "classify_intent"
)


builder.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "retrieve_and_answer":
            "retrieve_and_answer",

        "direct_answer":
            "direct_answer"
    }
)


builder.add_edge(
    "retrieve_and_answer",
    END
)

builder.add_edge(
    "direct_answer",
    END
)


graph = builder.compile()


if __name__ == "__main__":

    print("POLICY QUESTION")

    print(
        graph.invoke(
            {
                "query":
                "What is the delivery fee for orders below INR 149?"
            }
        )
    )

    print("\nGENERAL QUESTION")

    print(
        graph.invoke(
            {
                "query":
                "What is the capital of India?"
            }
        )
    )