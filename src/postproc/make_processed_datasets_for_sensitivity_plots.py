import pandas as pd
import numpy as np
from datasets import Dataset, DatasetInfo
from scipy.stats import wilcoxon, mannwhitneyu
def soft_wrap_words(text, width=25):
    words = text.split()
    lines = []
    line = ""
    for word in words:
        if len(line) + len(word) < width:
            line += word + " "
        else:
            lines.append(line)
            line = word + " "
    lines.append(line)
    return "<br>".join(lines)

def compute_sensitivies(groupless_df, df_query_response_rewards):
    """
    Compute sensitivity indexes
    :param groupless_df: Dataset where duplicate query response pairs for the same consitutional principle are removed (Shared between groups)
    :param df_query_response_rewards: Set of query response pairs with the original and perturbed rewards: Only used for computing percentile-based sensitivity index
    :return:
    """

    models = groupless_df["model"].unique()
    constitutional_principles = groupless_df["constitutional_principle"].unique()

    # Compute the elementary effects for each of the constitutional principles using the original and perturbed rewards
    sensitivity_data = []
    for model in models:
        all_rewards_by_model = df_query_response_rewards[df_query_response_rewards["model"] == model]
        sorted_rewards = all_rewards_by_model["reward"].sort_values().values
        for principle in constitutional_principles:
            # print(f"Model: {model}, Principle: {principle}")
            original_rewards = \
            groupless_df[(groupless_df["model"] == model) & (groupless_df["constitutional_principle"] == principle)][
                "original_reward"]
            perturbed_rewards = \
            groupless_df[(groupless_df["model"] == model) & (groupless_df["constitutional_principle"] == principle)][
                "perturbed_reward"]
            original_percentiles = np.searchsorted(sorted_rewards, original_rewards) / len(sorted_rewards)
            perturbed_percentiles = np.searchsorted(sorted_rewards, perturbed_rewards) / len(sorted_rewards)

            elementary_effects = perturbed_rewards - original_rewards  # Note that this assumes that the perturbed rewards are always higher than the original rewards
            # Compute indexes
            mean_effect = np.mean(elementary_effects)
            std_effect = np.std(elementary_effects)
            median_effect = np.median(elementary_effects)

            # Compute percentile change elementary effects relative to the query_response_dataset
            mean_percentile_effect = np.mean(perturbed_percentiles - original_percentiles)
            std_percentile_effect = np.std(perturbed_percentiles - original_percentiles)
            median_percentile_effect = np.median(perturbed_percentiles - original_percentiles)

            # Compute the wilcoxon signed rank test and report the U-statistic (Like ROC-AUC)
            wilcoxon_statistic, wilcoxon_p_value = wilcoxon(perturbed_rewards, original_rewards, alternative="greater")
            # Notice order, Docs: alternative= ‘greater’: the distribution underlying x (perturbed_rewards) is stochastically greater than the distribution underlying y (original_rewards)
            # Wilcoxon signed rank test is for paired samples, which we have here

            # Compute the Mann-Whitney U test
            manwhiteneyu_statistic, mannwhitneyu_p_value = mannwhitneyu(perturbed_rewards, original_rewards, alternative="greater")
            # My understanding ist that mannwhitneyu requires that the samples are independent, which is not strictly the case here

            # ALso save the ccai groups that is associated with this principle
            ccai_groups = groupless_df[groupless_df["constitutional_principle"] == principle]["ccai_group"].unique()

            row = {
                "model": model,
                "principle": principle,
                "ccai_groups": ccai_groups,
                "mean_effect": mean_effect,
                "median_effect": median_effect,
                "std_effect": std_effect,
                "mean_percentile_effect": mean_percentile_effect,
                "median_percentile_effect": median_percentile_effect,
                "std_percentile_effect": std_percentile_effect,
                "wilcoxon_statistic": wilcoxon_statistic,
                "wilcoxon_p_value": wilcoxon_p_value,
                "mannwhitneyu_statistic": manwhiteneyu_statistic,
                 "mannwhitneyu_p_value": mannwhitneyu_p_value
            }
            sensitivity_data.append(row)
    sensitivity_df = pd.DataFrame(sensitivity_data)
    # Sort the dataframe by the mean effect
    sensitivity_df = sensitivity_df.sort_values(by="mean_effect", ascending=False)

    # Add a new columns with the soft-wrapped principles
    sensitivity_df["principle_wrapped"] = sensitivity_df["principle"].apply(lambda x: soft_wrap_words(x, width=25))
    return sensitivity_df

if __name__ == "__main__":
    # Load the datasets for different collective constitutional AI groups
    df_ccai_0 = pd.read_csv("../../data/augmented_dataset_with_rewards_ccai_group_0.csv", index_col=0)
    df_ccai_1 = pd.read_csv("../../data/augmented_dataset_with_rewards_ccai_group_1.csv", index_col=0)

    # Remove the small number of rows with nan values where HF API failed
    df_ccai_0 = df_ccai_0.dropna()
    df_ccai_1 = df_ccai_1.dropna()

    # Upload the raw data (with a few nans removed) to HF
    # hf_dataset_0 = Dataset.from_pandas(df_ccai_0)
    # hf_dataset_0.push_to_hub("douwmarx/hh-rlhf-constitutional-sensitivities-of-pms", split="ccai_group_0")
    # hf_dataset_1 = Dataset.from_pandas(df_ccai_1)
    # hf_dataset_1.push_to_hub("douwmarx/hh-rlhf-constitutional-sensitivities-of-pms", split="ccai_group_1")


    # Combine both datasets
    df = pd.concat([df_ccai_0, df_ccai_1])
    df["original_reward"] = np.log(df["original_reward"])
    df["perturbed_reward"] = np.log(df["perturbed_reward"])

    # There is some overlap in the constitutional principles of the groups: Dropping the duplicates  between group 0 and 1 to avoid unrepresentative percentiles
    groupless_df = df.drop_duplicates(
        subset=["query", "initial_response", "label", "model", "constitutional_principle", "critique_request",
                "revision_request"])
    # Save the dataset without duplicates
    groupless_df.to_csv("../../data/groupless_dataset_with_rewards.csv")

    # Identify the constitutional principles that are unique to each group
    unique_principles_0 = set(df_ccai_0["constitutional_principle"].unique()) - set(df_ccai_1["constitutional_principle"].unique())
    unique_principles_1 = set(df_ccai_1["constitutional_principle"].unique()) - set(df_ccai_0["constitutional_principle"].unique())
    print(f"{len(unique_principles_0)} in group 0")
    print(f"{len(unique_principles_1)} in group 1")
    overlaping_principles = set(df_ccai_0["constitutional_principle"].unique()) & set(df_ccai_1["constitutional_principle"].unique())
    print(f"{len(overlaping_principles)} overlapping principles: {overlaping_principles}")

    df_with_no_overlapping_principles = df[~df["constitutional_principle"].isin(overlaping_principles)]
    df_with_no_overlapping_principles.to_csv("../../data/dataset_with_no_overlapping_principles_between_groups.csv")

    # Stack the query response pairs for initial and perturbed so that the percentiles relative to all prompts can be computed
    new_cols = ["query", "response", "label", "model", "constitutional_principle",
                "reward"]  # No more original and perturbed
    df_intial = groupless_df[["query", "initial_response", "label", "model", "constitutional_principle", "original_reward"]]
    df_intial.columns = new_cols
    df_intial["perturbed"] = False
    df_perturbed = groupless_df[
        ["query", "initial_response", "label", "model", "constitutional_principle", "perturbed_reward"]]
    df_perturbed.columns = new_cols
    df_perturbed["perturbed"] = True

    df_query_response_rewards = pd.concat(
        [df_intial, df_perturbed])  # Dataset with all the query response pairs that were evaluated by the reward models

    # Save the dataset with the query response pairs and the rewards
    df_query_response_rewards.to_csv("../../data/query_response_rewards.csv")

    # Compute the sensitivity indexes for different models
    sensitivity_df_for_different_models = compute_sensitivies(groupless_df, df_query_response_rewards)
    sensitivity_df_for_different_models.to_csv("../../data/sensitivity_data_for_different_models.csv")

    # Compute the sensitivity indexes for the different groups
    sensitivity_df_for_different_groups = compute_sensitivies(df_with_no_overlapping_principles, df_query_response_rewards)
    # Confirm that there is only one ccai group relating each principle and if so take this value out of the list
    if sensitivity_df_for_different_groups["ccai_groups"].apply(lambda x: len(x) > 1).any():
        raise ValueError("There is more than one ccai group for a principle in the df that compares groups")
    sensitivity_df_for_different_groups["ccai_group"] = sensitivity_df_for_different_groups["ccai_groups"].apply(lambda x: x[0])
    # Drop the ccai_groups column
    sensitivity_df_for_different_groups = sensitivity_df_for_different_groups.drop(columns=["ccai_groups"])
    sensitivity_df_for_different_groups.to_csv("../../data/sensitivity_data_for_different_groups.csv")


