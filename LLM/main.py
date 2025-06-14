API = "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"
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key= API,
)
res = llm.invoke("Explain the")
print(res.content)