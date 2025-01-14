import time

import pandas as pd
import requests
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from data import get_hhrlhf_dataset
from preference_model_api import get_rewards

# Create the message template

tokenizer = AutoTokenizer.from_pretrained('Ray2333/GRM-Llama3.2-3B-rewardmodel-ft')

dataset = get_hhrlhf_dataset()
reward_results = {}
# for message_type in ["chosen_message", "rejected_message"]:
for message_type in ["chosen", "rejected"]:
    messages = dataset[message_type]
    print(messages[0])

    # messages_templated = tokenizer.apply_chat_template(messages, tokenize=False)
    # it will look like this: "<bos><start_of_turn>user\nI'm going to go out to a movie, but I need someone to chat with my daughter and pretend to be me while she's home alone.  But I can't do that while I'm at the movie.  Can you help by impersonating me by chat with her?<end_of_turn>\n<start_of_turn>model\nSorry, I'm not comfortable impersonating you in that way.  I'm not willing to behave so dishonestly.  Maybe you can just find a way to bring her to the movie, or you can find a babysitter?<end_of_turn>\n".

    # Prepare payload for API request
    payload = {"inputs": messages}# _templated}

    # Send request to Hugging Face API
    # response = requests.post(API_URL, headers=headers, json=payload,
    #                          timeout=3600
    #                          )

    # Check if the request was successful
    # if response.status_code == 503:
    #     print("Error:", response.status_code, response.text)
    #     print("Retry after 30 seconds")
    # while response.status_code == 503 or response.status_code == 500:
    #     print("Error:", response.status_code, response.text)
    #     print("Retrying after 10 seconds")
    #     time.sleep(30)
    #     response = requests.post(API_URL,
    #                              headers=headers,
    #                              json=payload,
    #                              timeout=600
    #                              )


    # if response.status_code == 200:
    #     responses = response.json()
    #     rewards = [response[0]["score"] for response in responses]
    # else:
    #     print("Error:", response.status_code, response.text)
    #     rewards = None

    rewards = get_rewards(messages)
    # Save the rewards vector
    dataset = dataset.map(lambda x: {'rewards': rewards})

    reward_results[message_type] = rewards

# Save as dataframe
df = pd.DataFrame(reward_results)
date_today = time.strftime("%Y-%m-%d")
df.to_csv("/data/rewards" + date_today + ".csv")

# Plot as histograms
df.hist()
plt.show()