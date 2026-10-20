from dotenv import load_dotenv
from google import genai

load_dotenv()             
client = genai.Client()

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input="Say hello in German and explain the greeting in one sentence.",
)

print(interaction.output_text)
