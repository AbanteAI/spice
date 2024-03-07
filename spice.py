import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4-0125-preview"


def call_llm(messages, stream=False):
    def internal_gen():
        for chunk in chat_completion:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content

    chat_completion = OPENAI_CLIENT.chat.completions.create(
        messages=messages,
        model=MODEL,
        temperature=0.3,
        stream=stream,
    )

    if stream:
        return internal_gen()
    else:
        response = chat_completion.choices[0].message.content
        return response
