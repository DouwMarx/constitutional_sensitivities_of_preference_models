from transformers import AutoTokenizer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()  # Load the API-keys

api_key = os.getenv('HF_API_KEY')
hf_client = InferenceClient(api_key=api_key)

model="Ray2333/GRM-Llama3.2-3B-rewardmodel-ft"

message1= [{'role': 'user', 'content': 'Why do squares have 3 corners'},
                  {'role': 'assistant', 'content': "They only have two corners you bright spark"}]  # Compute for the perturbed query

message2= [{'role': 'user', 'content': 'Why do squares have 3 corners'},
                  {'role': 'assistant', 'content': 'Squares have four corners, not three.'}]  # Compute for the original query

messages = [message1, message2]

tokenizer = AutoTokenizer.from_pretrained(model)
tokenized_messages = tokenizer.apply_chat_template(messages, tokenize=False)

# https://huggingface.co/Ray2333/GRM-Llama3.2-3B-rewardmodel-ft
# According to model author is should look like this: "<bos><start_of_turn>user\nI'm going to go out to a movie, but I need someone to chat with my daughter and pretend to be me while she's home alone.  But I can't do that while I'm at the movie.  Can you help by impersonating me by chat with her?<end_of_turn>\n<start_of_turn>model\nSorry, I'm not comfortable impersonating you in that way.  I'm not willing to behave so dishonestly.  Maybe you can just find a way to bring her to the movie, or you can find a babysitter?<end_of_turn>\n".


print("Original message: ", messages[0])
print("Tokenized message: ", tokenized_messages[0])

reward = hf_client.text_classification(tokenized_messages[0], model=model)
print("Reward for message 1: ", reward[0].score)

reward = hf_client.text_classification(tokenized_messages[1], model=model)
print("Reward for message 2: ", reward[0].score)


# Returns:
# <|eot_id|><|start_header_id|>user<|end_header_id|>
#
# Why do squares have 3 corners<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#
# Actually, they only have two corners bright spark<|eot_id|>
# Reward for message 1:  0.0004608786548487842
# Reward for message 2:  0.08374433219432831


# 1) What does message based mean vs tokenized
# 2) Does this tokenizer actually do tokenization?
# 3) Recommendations for inference of more samples: Should I setup an inference endpoint? How if not automatically available?
