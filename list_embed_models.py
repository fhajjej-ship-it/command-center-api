import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
if "GOOGLE_API_KEY" in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
if "GEMINI_API_KEY" not in os.environ:
    print("No API key found in GOOGLE_API_KEY or GEMINI_API_KEY")
    exit(1)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(f"Model Name: {m.name}")
        print(f"Supported methods: {m.supported_generation_methods}")
        print("---")
