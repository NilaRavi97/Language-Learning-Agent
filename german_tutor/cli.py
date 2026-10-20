"""Command-line entry point: ``python -m german_tutor``."""

from __future__ import annotations

import argparse
import sqlite3
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from .config import CEFR_LEVELS, Settings
from .store import VocabStore

ROUTE_ICONS = {"conversation": "💬", "quiz": "🧠", "grammar": "📐"}
QUIT_WORDS = {"quit", "exit", ":q"}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="german_tutor", description="Lena, a German tutor agent.")
    p.add_argument("--level", choices=CEFR_LEVELS, help="CEFR level (default: $TUTOR_LEVEL or A2)")
    p.add_argument("--thread", default="default",
                   help="conversation id; history is persisted per thread (default: 'default')")
    p.add_argument("--new", action="store_true", help="start a fresh, randomly named thread")
    p.add_argument("--import-json", metavar="PATH", help="import vocabulary from a JSON file and exit")
    p.add_argument("--stats", action="store_true", help="print learning statistics and exit")
    p.add_argument("--quiet", action="store_true", help="hide routing and tool-call traces")
    return p.parse_args(argv)


def _print_turn(new_messages: list, route: str | None, quiet: bool) -> None:
    if not quiet:
        print(f"   {ROUTE_ICONS.get(route, '•')} routed to: {route}")
        for m in new_messages:
            if isinstance(m, AIMessage):
                for call in m.tool_calls:
                    args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                    print(f"   🔧 {call['name']}({args})")
            elif isinstance(m, ToolMessage) and m.status == "error":
                print(f"   ⚠️  tool error: {m.text}")
    reply = next((m for m in reversed(new_messages) if isinstance(m, AIMessage)), None)
    print(f"\nLena: {reply.text if reply else '(no reply)'}\n")


def rollback_turn(graph, config: dict, keep: int) -> None:
    """Drop messages a failed turn left in the checkpoint so the thread stays consistent.

    The update is applied as if it came from a terminal node, so the graph has nothing
    pending afterwards.
    """
    messages = graph.get_state(config).values.get("messages", [])
    if len(messages) > keep:
        graph.update_state(config, {"messages": [RemoveMessage(id=m.id) for m in messages[keep:]]},
                           as_node="conversation")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parse_args(argv)
    settings = Settings.from_env()
    level = args.level or settings.level
    store = VocabStore(settings.vocab_db)

    if args.import_json:
        added, skipped = store.import_json(args.import_json)
        print(f"Imported {added} words ({skipped} already present).")
        return 0
    if args.stats:
        print(store.profile_summary())
        return 0

    # Imported here so --stats/--import stay fast and don't need an API key.
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.checkpoint.sqlite import SqliteSaver

    from .graph import build_graph

    rate_limiter = None
    if settings.requests_per_minute:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=settings.requests_per_minute / 60, max_bucket_size=1
        )
    model = ChatGoogleGenerativeAI(model=settings.model, max_retries=4, rate_limiter=rate_limiter)
    settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = SqliteSaver(sqlite3.connect(settings.checkpoint_db, check_same_thread=False))
    graph = build_graph(model, store, level=level, checkpointer=checkpointer,
                        history_window=settings.history_window)

    thread_id = uuid.uuid4().hex[:8] if args.new else args.thread
    config = {"configurable": {"thread_id": thread_id}}
    previous = len(graph.get_state(config).values.get("messages", []))

    print(f"Chat with Lena ({level}, thread '{thread_id}'"
          f"{', resuming ' + str(previous) + ' messages' if previous else ''}).")
    print("Try: chatting in German, 'quiz me', or a grammar question. Type 'quit' to exit.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in QUIT_WORDS:
            break

        try:
            result = graph.invoke({"messages": [HumanMessage(user_text)]}, config)
        except Exception as exc:  # keep the session alive on API/network errors
            detail = str(exc).splitlines()[0][:200]
            hint = " (set TUTOR_RPM=5 on the free tier)" if "RESOURCE_EXHAUSTED" in detail else ""
            print(f"\n⚠️  {type(exc).__name__}: {detail}{hint}\n", file=sys.stderr)
            rollback_turn(graph, config, keep=previous)
            continue

        messages = result["messages"]
        _print_turn(messages[previous + 1:], result.get("route"), args.quiet)
        previous = len(messages)

    print(store.profile_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
