import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(
    system_prompt: str,
    user_prompt: str,
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """
    Generate answer using Groq's free API.
    llama-3.1-8b-instant is fast, free, and good enough
    for grounded clinical Q&A where the LLM is just
    reasoning over provided context — not recalling from memory.
    """
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )

    return response.choices[0].message.content