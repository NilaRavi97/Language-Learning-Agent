from __future__ import annotations

from collections.abc import Iterator

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from german_tutor.graph import RouteDecision
from german_tutor.store import VocabStore


class ScriptedChatModel(GenericFakeChatModel):
    """Fake model that replays scripted AIMessages and supports tool binding."""

    def bind_tools(self, tools, **kwargs):
        return self


class FakeRouter(GenericFakeChatModel):
    """Fake model whose structured output always returns the next scripted route."""

    routes: list[str] = []

    def with_structured_output(self, schema, **kwargs):
        routes = iter(self.routes)
        return RunnableLambda(lambda _: RouteDecision(route=next(routes), reason="test"))


def scripted(*messages: AIMessage) -> ScriptedChatModel:
    it: Iterator[AIMessage] = iter(messages)
    return ScriptedChatModel(messages=it)


def router(*routes: str) -> FakeRouter:
    return FakeRouter(messages=iter([]), routes=list(routes))


@pytest.fixture
def store() -> Iterator[VocabStore]:
    s = VocabStore(":memory:")
    yield s
    s.close()
