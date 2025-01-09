from datasets import Dataset

from constitutional_perturbations import get_constiutional_perturbations
from data import get_hhrlhf_query_response_dataset
from scale_api_based_preference_model import get_original_and_perturbed_rewards_for_all_rows

# Load the query-response dataset
dataset = get_hhrlhf_query_response_dataset()
dataset = list(dataset)
dataset = dataset[0:2]

# Augment the dataset by perturbing the responses according the the constitution
augmented_dataset_list = get_constiutional_perturbations(dataset)

# Compute the rewards for the original and perturbed responses
augmented_dataset_list = get_original_and_perturbed_rewards_for_all_rows(augmented_dataset_list)

# Convert the augmented dataset to a Dataset object
augmented_dataset = Dataset.from_list(augmented_dataset_list)

# Save as pandas dataframe
augmented_dataset = augmented_dataset.to_pandas()
augmented_dataset.to_csv("data/augmented_dataset.csv")