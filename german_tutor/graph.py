"""The tutor as a LangGraph ``StateGraph``.

    START ──► router ──┬──► conversation ──┐
                       ├──► quiz ──────────┼──► END
                       └──► grammar ───────┘

* ``router`` classifies the latest learner message with structured output.
* Each specialist is a ``create_agent`` ReAct sub-agent with its own prompt and tool set.
  Its system prompt is rendered per call by middleware, so it always includes the
  current learner profile (vocabulary stats and recurring mistakes) from SQLite.
* A checkpointer persists the whole conversation per ``thread_id``.
"""

from __future__ import annotations

from typing import Literal

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt, wrap_model_call
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, trim_messages
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Checkpointer
from pydantic import BaseModel, Field

from .prompts import CONVERSATION_PROMPT, GRAMMAR_PROMPT, QUIZ_PROMPT, ROUTER_PROMPT
from .store import VocabStore
from .tools import make_tools

Route = Literal["conversation", "quiz", "grammar"]
ROUTES: tuple[Route, ...] = ("conversation", "quiz", "grammar")


class RouteDecision(BaseModel):
    """Which specialist should handle the learner's latest message."""

    route: Route = Field(description="The specialist to hand the message to.")
    reason: str = Field(description="One short sentence explaining the choice.")


class TutorState(MessagesState):
    route: Route


def _transcript(messages: list[AnyMessage], max_turns: int = 6) -> str:
    """Render recent learner/tutor text as plain lines for the router.

    Tool calls and tool results are skipped; plain text keeps the router call valid for any
    provider regardless of where the history window is cut.
    """
    lines: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"Learner: {m.text}")
        elif isinstance(m, AIMessage) and m.text.strip():
            lines.append(f"Tutor: {m.text}")
    return "\n".join(lines[-max_turns:])


def _system_prompt(template: str, store: VocabStore, level: str):
    @dynamic_prompt
    def render(request: ModelRequest) -> str:
        return template.format(level=level, profile=store.profile_summary())

    return render


def _history_window(max_messages: int):
    """Send only the most recent messages to the model; the full history stays checkpointed.

    ``start_on="human"`` guarantees the window never begins with an orphaned tool result.
    """

    @wrap_model_call
    def trim(request: ModelRequest, handler):
        window = trim_messages(
            request.messages,
            strategy="last",
            token_counter=len,
            max_tokens=max_messages,
            start_on="human",
        )
        return handler(request.override(messages=window))

    return trim


def build_graph(
    model: BaseChatModel,
    store: VocabStore,
    *,
    level: str = "A2",
    checkpointer: Checkpointer | None = None,
    router_model: BaseChatModel | None = None,
    history_window: int = 30,
):
    tools = make_tools(store)

    def specialist(name: str, prompt: str, tool_list: list):
        return create_agent(
            model,
            tool_list,
            middleware=[_system_prompt(prompt, store, level), _history_window(history_window)],
            name=name,
        )

    conversation = specialist(
        "conversation",
        CONVERSATION_PROMPT,
        [tools.save_word, tools.list_saved_words, tools.log_mistake, tools.get_learner_profile],
    )
    quiz = specialist("quiz", QUIZ_PROMPT, [tools.get_due_words, tools.record_review])
    grammar = specialist("grammar", GRAMMAR_PROMPT, [tools.save_word, tools.log_mistake])

    classifier = (router_model or model).with_structured_output(RouteDecision)

    def router(state: TutorState) -> dict:
        transcript = _transcript(state["messages"])
        try:
            decision = classifier.invoke(
                [SystemMessage(ROUTER_PROMPT), HumanMessage(f"Recent transcript:\n{transcript}")]
            )
            route = decision.route if decision and decision.route in ROUTES else "conversation"
        except Exception:  # a failed classification should never break the chat
            route = "conversation"
        return {"route": route}

    builder = StateGraph(TutorState)
    builder.add_node("router", router)
    builder.add_node("conversation", conversation)
    builder.add_node("quiz", quiz)
    builder.add_node("grammar", grammar)

    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", lambda s: s["route"], list(ROUTES))
    for name in ROUTES:
        builder.add_edge(name, END)

    return builder.compile(checkpointer=checkpointer, name="german_tutor")
