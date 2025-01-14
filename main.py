from datasets import Dataset, DatasetInfo

from constitutional_perturbations import get_constiutional_perturbations, get_collective_constitutional_ai_principles
from data import get_hhrlhf_query_response_dataset
from preference_model_api import get_original_rewards_for_all_rows, get_perturbed_rewards_for_all_rows
import pickle

models = [
    "Ray2333/GRM-Llama3.2-3B-rewardmodel-ft",
    "Ray2333/GRM-gemma2-2B-rewardmodel-ft",
    # "Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback"
]

constitutional_principles = get_collective_constitutional_ai_principles(5)

# Load the query-response dataset
dataset = get_hhrlhf_query_response_dataset()
dataset = list(dataset)
dataset = dataset[0:10]

# # Write the list of dataset to a temporary pickle file for quick reloading
with open("data/dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)

# Load the dataset from the pickle file
with open("data/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# Compute the rewards for the original responses
data_with_original_rewards = get_original_rewards_for_all_rows(dataset, models=models)

# Get the perturbed the responses
data_with_perturbed_responses = get_constiutional_perturbations(data_with_original_rewards, constitutional_principles=constitutional_principles)

# Compute the rewards for the perturbed responses
data_with_perturbed_rewards = get_perturbed_rewards_for_all_rows(data_with_perturbed_responses)

# Convert the augmented dataset to a Dataset object

info = DatasetInfo(description="A dataset of query-response pairs from the Antropic hhrlfh dataset with original and perturbed responses and the rewards for each response.")
augmented_dataset = Dataset.from_list(data_with_perturbed_rewards, info=info)

# Upload the dataset to the Hugging Face Hub
augmented_dataset.push_to_hub("douwmarx/hh-rlhf-pm-constitutional-sensitivities")

# Save as pandas dataframe
augmented_dataset = augmented_dataset.to_pandas()
augmented_dataset.to_csv("data/augmented_dataset.csv")