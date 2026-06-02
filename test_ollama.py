import ollama

def test_ollama_connection():
    try:
        response = ollama.chat(
            model="deepseek-r1:7b",
            messages=[
                {
                    "role": "user",
                    "content": "Hello! Can you tell me a brief introduction about natural language processing?"
                }
            ]
        )
        print("✅ Ollama API connection successful!")
        print("Response:", response['message']['content'][:200], "...")
        return True
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("Please make sure Ollama is installed and running, and the model is downloaded.")
        print("Download command: ollama pull deepseek-r1:7b")
        return False

def list_available_models():
    try:
        models = ollama.list()
        print("\nAvailable models:")
        for model in models['models']:
            print(f"- {model['name']}")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    print("Testing Ollama connection...")
    test_ollama_connection()
    list_available_models()