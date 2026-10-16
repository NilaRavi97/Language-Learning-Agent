from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=4)

history = []  

print("Chat with Gemini (LangChain). Type 'quit' to exit.\n")
while True:
    user_text = input("You: ")
    if user_text.lower() == "quit":
        break

    history.append(HumanMessage(content=user_text))
    response = model.invoke(history)     
    history.append(response)            

    print("Gemini:", response.content, "\n")
