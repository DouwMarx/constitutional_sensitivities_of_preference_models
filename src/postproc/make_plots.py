import numpy as np
import pandas as pd
import plotly
from plotly import graph_objects as go
from src.postproc.make_processed_datasets_for_sensitivity_plots import soft_wrap_words

def underscore_to_sentence_case(text):
    """
    Makes the metric names prettier
    :param text:
    :return:
    """
    return text.replace("_", " ").capitalize()

def make_orginal_and_perturbed_rewards_plot(groupless_df, query_response_rewards_df, model="Ray2333/GRM-Llama3.2-3B-rewardmodel-ft"):
    # Sample 30 examples to show
    examples = query_response_rewards_df.sample(30, random_state=0)

    # Soft wrap the query and response so they can be displayed in the hover text
    examples["query_response"] = examples["query"] + "<br><br>" + examples["response"]
    examples["query_response"] = examples["query_response"].apply(soft_wrap_words)

    model_specific_groupless_df = groupless_df[groupless_df["model"] == model]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=model_specific_groupless_df["original_reward"], name="Original", histnorm='probability'))
    fig.add_trace(
        go.Histogram(x=model_specific_groupless_df["perturbed_reward"], name="Perturbed", histnorm='probability'))

    fig.add_trace(go.Scatter(x=examples["reward"],
                             y=-0.001 * np.ones_like(examples["reward"]),
                             mode="markers",
                             hovertext=examples["query_response"],
                             hoverinfo="text",
                             marker=dict(size=10, color="black"), showlegend=False))

    fig.update_layout(title="Original and Perturbed Rewards",
                      xaxis_title="log(Reward)",
                      yaxis_title="Probability Density")
    # Export as html
    fig.write_html("docs/_includes/original_and_perturbed_rewards.html")
    fig.show()


def make_sensitivity_plot_comparing_models(sensitivity_data_for_different_models, metric):
    fig = go.Figure()
    models = sensitivity_data_for_different_models["model"].unique()
    # Sort the data by the metric
    sensitivity_data_for_different_models = sensitivity_data_for_different_models.sort_values(by=metric, ascending=False)

    plot_min = -np.inf
    plot_max = np.inf
    for model in models:
        model_data = sensitivity_data_for_different_models[sensitivity_data_for_different_models["model"] == model]
        y = model_data[metric]
        # Sum normalize for the model
        y_norm = y / y.sum()

        plot_min = max(plot_min, y_norm.min())
        plot_max = min(plot_max, y_norm.max())

        # Min-max normalize for the model
        # y_norm = (y-y.min())/(y.max()-y.min())

        # log transform
        # y_norm = np.log(y)

        # If the metric contains "mean", also add error bars for the standard deviation
        # if "mean" in metric:
        #     fig.add_trace(go.Bar(x=model_data["principle"], y=y_norm, name=model, error_y=dict(type='data', array=model_data["std_effect"])))
        # else:
        fig.add_trace(go.Bar(x=model_data["principle_wrapped"], y=y_norm, name=model))

    fig.update_layout(title=f"Normalised sensitivity index: {underscore_to_sentence_case(metric)}",
                      xaxis_title="Constitutional Principle",
                      yaxis_title=f"Normalised sensitivity index: {underscore_to_sentence_case(metric)}",
                        )
    fig.update_yaxes(
        range=[0.9*plot_min, 1.05*plot_max],
        # range=[0.5, 0.6],
    )

    # Export as html
    fig.write_html(f"docs/_includes/compare_model_sensitivity_{metric}.html")
    fig.show()


def make_sensitivity_plot_comparing_groups(sensitivities_for_groups, metric, model ="Ray2333/GRM-Llama3.2-3B-rewardmodel-ft"):
    fig = go.Figure()


    model_specific_sensitivities_for_groups = sensitivities_for_groups[sensitivities_for_groups["model"] == model]

    # Sort data by the metric
    model_specific_sensitivities_for_groups = model_specific_sensitivities_for_groups.sort_values(by=metric, ascending=False)
    # Sum normalize the metric
    model_specific_sensitivities_for_groups[metric] = model_specific_sensitivities_for_groups[metric] / \
                                                      model_specific_sensitivities_for_groups[metric].sum()

    fig.add_trace(go.Bar(x=model_specific_sensitivities_for_groups["ccai_group"],
                         y=model_specific_sensitivities_for_groups[metric],
                         name="Group 0: CCAI",
                         hovertext=model_specific_sensitivities_for_groups["principle_wrapped"],
                         hoverinfo="text",
                         # Set the color on the size of the metric
                         marker=dict(color=model_specific_sensitivities_for_groups[metric],
                                     colorscale='Viridis')
                         )
                  )
    fig.update_layout(title=f"Normalised sensitivity index: {underscore_to_sentence_case(metric)}",
                      xaxis_title="CCAI Group",
                      yaxis_title=f"Normalised sensitivity index: {underscore_to_sentence_case(metric)}",
                      barmode="stack"
                      )

    # Export as html
    fig.write_html(f"docs/_includes/compare_group_sensitivity_{metric}.html")
    fig.show()

if __name__ == "__main__":
    # Load the data
    # groupless_df = pd.read_csv("data/groupless_dataset_with_rewards.csv", index_col=0)
    # query_response_rewards_df = pd.read_csv("data/query_response_rewards.csv", index_col=0)
    sensitivity_data_for_different_models = pd.read_csv("data/sensitivity_data_for_different_models.csv", index_col=0)
    sensitivity_data_for_different_groups = pd.read_csv("data/sensitivity_data_for_different_groups.csv", index_col=0)

    # Make the histogram plot showing the effect of the perturbation
    # make_orginal_and_perturbed_rewards_plot(groupless_df, query_response_rewards_df)

    # Show plots of the sensitivity indexes for the different models
    for metric in [
        "median_effect",
        "mean_effect",
        # "std_effect",
        # "mean_percentile_effect",
        "median_percentile_effect",
        # "std_percentile_effect",
        "wilcoxon_statistic",
        # "mannwhitneyu_statistic",
        # "mannwhitneyu_p_value",
        # "wilcoxon_p_value"
    ]:
        make_sensitivity_plot_comparing_models(sensitivity_data_for_different_models, metric)


        make_sensitivity_plot_comparing_groups(sensitivity_data_for_different_groups, metric)







