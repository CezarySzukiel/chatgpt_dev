import os
from dotenv import load_dotenv
from openai import OpenAI

import utils

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gpt_model = os.getenv("GPT_MODEL")


def chat_completions_request(messages, model=gpt_model, json_mode=True):
    api_params = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }

    if json_mode:
        api_params["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**api_params)
    return response.choices[0].message


def process_transcript(transcript):
    prompts_path = os.getenv("PROMPTS_PATH")
    messages = [
        {"role": "system", "content": utils.open_file(os.path.join(prompts_path, "system_prompt.txt"))},
        {"role": "user", "content": utils.open_file(os.path.join(prompts_path, "user_prompt_01.txt")) + transcript},
    ]

    firts_response = chat_completions_request(messages)
    print("Information extracted... ⏰")

    print(firts_response)


if __name__ == "__main__":
    directory_path = os.getenv("TRANSCRIPTS_PATH")
    files = os.listdir(directory_path) # List all files in the directory

    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(directory_path, file)

            process_transcript(utils.open_file(file_path))