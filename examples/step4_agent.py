import json
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=4)

WORDS_FILE = Path("words.json") 


# ---------- TOOLS ----------

@tool
def save_word(german: str, english: str, example: str) -> str:
    """Save a German word or phrase to the learner's vocabulary list.

    Args:
        german: The German word. For nouns include the article and plural, e.g. "der Bahnhof (-e)".
        english: The English meaning.
        example: A short, simple German example sentence using the word.
    """
    words = json.loads(WORDS_FILE.read_text(encoding="utf-8")) if WORDS_FILE.exists() else []

    if any(w["german"].lower() == german.lower() for w in words):
        return f"'{german}' is already saved."

    words.append({"german": german, "english": english, "example": example})
    WORDS_FILE.write_text(json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   🔧 [tool] save_word('{german}')")   # so you can SEE the tool run
    return f"Saved '{german}'. The learner now has {len(words)} words."


@tool
def list_saved_words() -> str:
    """Return all words the learner has saved so far."""
    print("   🔧 [tool] list_saved_words()")
    if not WORDS_FILE.exists():
        return "No words saved yet."
    words = json.loads(WORDS_FILE.read_text(encoding="utf-8"))
    return "\n".join(f"{w['german']} – {w['english']}" for w in words)


# ---------- AGENT ----------

LEVEL = "A2"

SYSTEM_PROMPT = f"""You are Lena, a friendly and patient German tutor.
The learner is an English speaker at CEFR level {LEVEL}.

## Conversation
- Reply mostly in simple German suitable for {LEVEL}. Keep replies to 2-4 sentences and end with a question.

## Corrections
If the learner's German has mistakes, start with:
✏️ Correction: <corrected sentence>
💡 Why: <one short explanation in English>
Never invent mistakes that aren't there.

## Vocabulary (use your tools)
- When the learner asks what a word means, or uses a word incorrectly, call save_word for it.
- When the learner asks to save a word, call save_word.
- When the learner asks what words they have learned, call list_saved_words.
- After saving, briefly tell the learner which word you saved.
"""

agent = create_agent(
    model=model,
    tools=[save_word, list_saved_words],
    system_prompt=SYSTEM_PROMPT,
)

# ---------- CHAT LOOP ----------

messages = []

print("Chat with Lena (agent). Type 'quit' to exit.\n")
while True:
    user_text = input("You: ")
    if user_text.lower() == "quit":
        break

    messages.append({"role": "user", "content": user_text})
    result = agent.invoke({"messages": messages})
    messages = result["messages"]        
    print("\nLena:", messages[-1].text, "\n")
