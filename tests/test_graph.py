import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from german_tutor.cli import rollback_turn
from german_tutor.graph import _transcript, build_graph
from german_tutor.store import VocabStore

from .conftest import router, scripted

CONFIG = {"configurable": {"thread_id": "t1"}}


def tool_call(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_conversation_agent_calls_tools_and_persists_word(store: VocabStore):
    model = scripted(
        tool_call("save_word", {"german": "der Bahnhof (-e)", "english": "train station",
                                "example": "Der Bahnhof ist groß."}),
        AIMessage("Gespeichert! 📚 der Bahnhof – train station. Wohin fährst du?"),
    )
    graph = build_graph(model, store, router_model=router("conversation"))

    result = graph.invoke({"messages": [HumanMessage("Was heißt Bahnhof?")]})

    assert result["route"] == "conversation"
    assert result["messages"][-1].text.startswith("Gespeichert!")
    assert any(isinstance(m, ToolMessage) and "Saved" in m.text for m in result["messages"])
    assert [w.german for w in store.all_words()] == ["der Bahnhof (-e)"]


def test_quiz_route_records_review(store: VocabStore):
    store.add_word("die Tür", "door")
    word_id = store.all_words()[0].id
    model = scripted(
        tool_call("record_review", {"word_id": word_id, "correct": True}),
        AIMessage("Richtig! Nächstes Wort..."),
    )
    graph = build_graph(model, store, router_model=router("quiz"))

    result = graph.invoke({"messages": [HumanMessage("die Tür")]})

    assert result["route"] == "quiz"
    assert store.get_word(word_id).box == 2


def test_checkpointer_keeps_history_across_turns(store: VocabStore):
    model = scripted(AIMessage("Hallo! Wie geht's?"), AIMessage("Das Dativ ist..."))
    graph = build_graph(model, store, checkpointer=InMemorySaver(),
                        router_model=router("conversation", "grammar"))

    graph.invoke({"messages": [HumanMessage("Hallo")]}, CONFIG)
    result = graph.invoke({"messages": [HumanMessage("How does the dative work?")]}, CONFIG)

    assert result["route"] == "grammar"
    assert [m.text for m in result["messages"]] == [
        "Hallo", "Hallo! Wie geht's?", "How does the dative work?", "Das Dativ ist...",
    ]


def test_router_failure_falls_back_to_conversation(store: VocabStore):
    model = scripted(AIMessage("Okay!"))
    graph = build_graph(model, store, router_model=router())  # no routes -> StopIteration

    result = graph.invoke({"messages": [HumanMessage("Hi")]})

    assert result["route"] == "conversation"


def test_transcript_skips_tool_traffic():
    messages = [
        HumanMessage("Quiz me"),
        tool_call("get_due_words", {}),
        ToolMessage("id=1 | german=die Tür", tool_call_id="call-1"),
        AIMessage("Was heißt 'door'?"),
        HumanMessage("die Tür"),
    ]
    assert _transcript(messages) == "Learner: Quiz me\nTutor: Was heißt 'door'?\nLearner: die Tür"


def test_failed_turn_is_rolled_back(store: VocabStore):
    model = scripted(AIMessage("Hallo!"))  # only one scripted reply: turn 2 will fail
    graph = build_graph(model, store, checkpointer=InMemorySaver(),
                        router_model=router("conversation", "conversation"))
    graph.invoke({"messages": [HumanMessage("Hallo")]}, CONFIG)

    with pytest.raises(Exception):
        graph.invoke({"messages": [HumanMessage("Noch einmal")]}, CONFIG)
    rollback_turn(graph, CONFIG, keep=2)

    state = graph.get_state(CONFIG)
    assert [m.text for m in state.values["messages"]] == ["Hallo", "Hallo!"]
    assert state.next == ()
