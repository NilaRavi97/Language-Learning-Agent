"""LangChain tools that give the tutor agents access to the learner's data.

Tools are built by a factory bound to a ``VocabStore`` so they are easy to test with an
in-memory database and never touch global state.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain.tools import tool
from langchain_core.tools import BaseTool

from .store import VocabStore


@dataclass(frozen=True)
class TutorTools:
    save_word: BaseTool
    list_saved_words: BaseTool
    get_due_words: BaseTool
    record_review: BaseTool
    log_mistake: BaseTool
    get_learner_profile: BaseTool


def make_tools(store: VocabStore) -> TutorTools:
    @tool
    def save_word(german: str, english: str, example: str) -> str:
        """Save a German word or phrase to the learner's vocabulary list.

        Args:
            german: The German word. For nouns include article and plural, e.g. "der Bahnhof (-e)".
            english: The English meaning.
            example: A short, simple German example sentence using the word.
        """
        if store.add_word(german, english, example):
            return f"Saved '{german}'. The learner now has {store.stats()['total']} words."
        return f"'{german}' is already in the vocabulary list."

    @tool
    def list_saved_words() -> str:
        """List every word the learner has saved, with its review box (1 = new, 5 = mastered)."""
        words = store.all_words()
        if not words:
            return "No words saved yet."
        return "\n".join(f"{w.german} – {w.english} (box {w.box})" for w in words)

    @tool
    def get_due_words(limit: int = 5) -> str:
        """Get vocabulary words that are due for a spaced-repetition review.

        Args:
            limit: Maximum number of words to return.
        """
        words = store.due_words(limit=max(1, min(limit, 20)))
        if not words:
            return "No words are due for review today."
        return "\n".join(
            f"id={w.id} | german={w.german} | english={w.english} | box={w.box}" for w in words
        )

    @tool
    def record_review(word_id: int, correct: bool) -> str:
        """Record the result of quizzing the learner on one word. Call once per answered question.

        Args:
            word_id: The id returned by get_due_words.
            correct: True if the learner's answer was correct (minor spelling slips are OK).
        """
        result = store.record_review(word_id, correct)
        if result is None:
            return f"No word with id={word_id}."
        verdict = "correct" if correct else "wrong"
        return (f"Recorded {verdict} for '{result.word.german}'. "
                f"Now in box {result.new_box}, next review {result.next_due.isoformat()}.")

    @tool
    def log_mistake(category: str, wrong: str, corrected: str) -> str:
        """Log a grammar or vocabulary mistake the learner made, to track recurring weaknesses.

        Args:
            category: Short, reusable label such as "perfekt with sein", "dative case",
                "verb position", "noun gender", "adjective endings".
            wrong: The learner's incorrect phrase.
            corrected: The corrected phrase.
        """
        store.log_mistake(category, wrong, corrected)
        return f"Logged mistake under '{category.strip().lower()}'."

    @tool
    def get_learner_profile() -> str:
        """Get the learner's progress statistics and most frequent mistake categories."""
        return store.profile_summary()

    return TutorTools(
        save_word=save_word,
        list_saved_words=list_saved_words,
        get_due_words=get_due_words,
        record_review=record_review,
        log_mistake=log_mistake,
        get_learner_profile=get_learner_profile,
    )
