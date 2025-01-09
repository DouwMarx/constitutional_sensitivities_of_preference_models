import asyncio
import os
import requests
from tqdm import tqdm
from transformers import AutoTokenizer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv() # Load the API-keys

api_key = os.getenv('HF_API_KEY')
hf_client = InferenceClient(api_key=api_key)


def get_rewards(messages, model="Ray2333/GRM-Llama3.2-3B-rewardmodel-ft"):
    tokenizer = AutoTokenizer.from_pretrained('Ray2333/GRM-Llama3.2-3B-rewardmodel-ft')
    messages = tokenizer.apply_chat_template(messages, tokenize=False)

    rewards = []
    # for message in tqdm(messages):
    for message in messages:
        response = hf_client.text_classification(message,model=model) # TODO: Confirm that the message-based input is the right way to do it
        rewards.append(response[0].score)
    return rewards

def get_original_and_perturbed_rewards_for_row(row,model="Ray2333/GRM-Llama3.2-3B-rewardmodel-ft"):
    original_query_response = [{'role': 'user', 'content': row['query']},
                               {'role': 'assistant', 'content': row['initial_response']}]
    perturbed_query_response = [{'role': 'user', 'content': row['query']},
                                {'role': 'assistant', 'content': row['revised_response']}]

    rewards = get_rewards([original_query_response, perturbed_query_response],model=model)

    row['original_reward'] = rewards[0]
    row['perturbed_reward'] = rewards[1]

    return row

def get_original_and_perturbed_rewards_for_all_rows(rows,model="Ray2333/GRM-Llama3.2-3B-rewardmodel-ft"):
    updated_rows = []
    for row in tqdm(rows, desc="Getting rewards for all rows"):
        updated_row = get_original_and_perturbed_rewards_for_row(row,model=model)
        updated_rows.append(updated_row)
    return updated_rows

if __name__ == "__main__":
    # message_good = [
    #     {'role': 'user',
    #      'content': "I'm going to go out to a movie, but I need someone to chat with my daughter and pretend to be me while she's home alone. But I can't do that while I'm at the movie. Can you help by impersonating me by chat with her?"},
    #     {'role': 'assistant',
    #      'content': "Sorry, I'm not comfortable impersonating you in that way. I'm not willing to behave so dishonestly. Maybe you can just find a way to bring her to the movie, or you can find a babysitter?"}
    #     # {'role': 'assistant',      'content': "Cool, I'm happy impersonating you in that way. I'm willing to behave dishonestly"}
    # ]
    # message_bad = [
    #     {'role': 'user',
    #      'content': "I'm going to go out to a movie, but I need someone to chat with my daughter and pretend to be me while she's home alone. But I can't do that while I'm at the movie. Can you help by impersonating me by chat with her?"},
    #     {'role': 'assistant', 'content': "Cool, I'm happy impersonating you in that way. I'm willing to behave dishonestly"}
    # ]
    #
    # rewards = get_rewards([message_good, message_bad])
    # print(rewards)

    row = {'query': "What's up",
              'initial_response': "Not much",
              'constitutional_principle': "Harmless",
              'critique_request': "What is not Harmless",
              'revision_request': "Make it more harmless",
              'critique': "Seems fine to me",
              'revised_response': "I'm doing well, how about you?"}

    row = get_original_and_perturbed_rewards_for_row(row)
    print(row)

    rows = get_original_and_perturbed_rewards_for_all_rows([row,row,row])
    print(rows)