import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def test_gemini():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    print("API KEY PRESENT:", bool(api_key))
    if api_key:
        print("API KEY START:", api_key[:10] + "...")
        
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        msg = HumanMessage(content="Hello Gemini, reply with 'Connection OK'")
        response = llm.invoke([msg])
        print("Response:", response.content)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_gemini()
