from langchain_community.llms import Ollama

def invoke_gemma():
    llm = Ollama(model='gemma')
    response = llm.invoke("Give me 5 questions about the Indian Prime Minister Lal Bahadur Shastri and its answers in 4 options format")
    print("Response:{}".format(response))

invoke_gemma()