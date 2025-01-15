import re
import datasets.config
from datasets import load_dataset, load_from_disk
from datasets import Dataset

datasets.config.IN_MEMORY_MAX_SIZE = 1024 * 1024 * 1024  * 6# 1GB

def convert_to_message_format(text,label=None):
    # Define the role mapping
    role_map = {
        'Human': 'user',
        'Assistant': 'assistant'
    }

    # Split the input text into relevant lines, preserving order
    # Using regex to capture both roles and their content
    lines = re.findall(r'(Human|Assistant): (.*?)\n', text, re.DOTALL)

    # Create the message representation
    messages = [{'role': role_map[role], 'content': content.strip(), 'label':label} for role, content in lines]

    return messages

def extract_query_response_pairs(text,label=None):
    # Split text into pairs of (Human, Assistant) interactions
    interactions = re.findall(r'Human: (.*?)\nAssistant: (.*?)(?=(?:\nHuman:|\Z))', text, re.DOTALL)
    # Convert to {query, response} format and keep the label of the preference
    query_response_pairs = [{'query': user.strip(), 'initial_response': assistant.strip(),'label':label} for user, assistant in interactions]

    return query_response_pairs


def get_hhrlhf_dataset(split="test", size="100"):
    # Get the harmfulness subset
    dataset = load_dataset("Anthropic/hh-rlhf",
                           data_dir="harmless-base",
                           # split="test[:5%]",
                           split=f"{split}[:{size}%]",
                           keep_in_memory=True,
                           cache_dir="cache")
    return dataset

def get_hhrlhf_query_response_dataset():
    dataset = get_hhrlhf_dataset()

    # Use both chosen and rejected
    # all_response_pairs = []
    # for label in ["chosen", "rejected"]:
    #     for example in dataset:
    #         # Extract query-response pairs from the example
    #         query_response_pairs = extract_query_response_pairs(example[label],label =label)
    #         # Add the pairs to the list
    #         all_response_pairs += query_response_pairs

    # Use only rejected
    all_response_pairs = []
    for example in dataset:
        # Extract query-response pairs from the example
        query_response_pairs = extract_query_response_pairs(example["rejected"],label ="rejected")
        # Add the pairs to the list
        all_response_pairs += query_response_pairs

    # Create a new dataset from the extracted pairs
    new_dataset = Dataset.from_list(all_response_pairs)

    new_dataset = new_dataset.shuffle(seed=0) # Shuffle the dataset entries
    return new_dataset


def get_local_dataset():
    dataset = load_from_disk("hhrlhf_test_5p")
    return dataset

if __name__ == "__main__":
    dataset = get_hhrlhf_query_response_dataset()


    # dataset = get_hhrlhf_dataset()
    # Stack the chosen and rejected messages, but add a new colum  with 0 for chosen and 1 for rejected

    # Stack all the dialogs, we are only intestersted in question:response, and not the full dialogue
    all_messages = []
    # Process each example
    counter = 0
    for example in dataset:
        # Convert chosen and rejected to message format
        chosen_messages = convert_to_message_format(example['chosen'],label="chosen")
        rejected_messages = convert_to_message_format(example['rejected'],label="rejected")

        # Combine both sets of messages
        all_messages += chosen_messages
        all_messages += rejected_messages

    # Create a new dataset from the stacked dialog list
    new_dataset = Dataset.from_list(all_messages)

    # Display the first example from the new dataset
    print(new_dataset[0])

    dataset.save_to_disk("hhrlhf_test_5p")