import numpy as np
import pandas as pd
import plotly
from plotly import graph_objects as go

# Make a box plot from the saved results

df = pd.read_csv("data/augmented_dataset.csv", index_col=0)


# Make a plotly box plot with the reward on the y
fig = go.Figure()
fig.add_trace(go.Box(y=np.log(df["original_reward"]), name="Original",jitter=0.3))
fig.add_trace(go.Box(y=np.log(df["perturbed_reward"]), name="Perturbed",jitter=0.3))
fig.update_layout(title="Original and Perturbed Rewards",
                    xaxis_title="Response Type",
                    yaxis_title="Log(Reward)")

# Export as html
fig.write_html("docs/_includes/original_and_perturbed_rewards.html")

fig.show()


# Compute the sobol indices for each of the constitutional principles using the original and perturbed rewards
models = df["model"].unique()
constitutional_principles = df["constitutional_principle"].unique()

# Compute the sobol indices
sobol_data = []
for model in models:
    for principle in constitutional_principles:
        print(f"Model: {model}, Principle: {principle}")
        original_rewards = df[(df["model"] == model) & (df["constitutional_principle"] == principle)]["original_reward"]
        perturbed_rewards = df[(df["model"] == model) & (df["constitutional_principle"] == principle)]["perturbed_reward"]

        original_rewards=np.log(original_rewards)
        perturbed_rewards= np.log(perturbed_rewards)

        elementary_effects = perturbed_rewards - original_rewards

        mean_effect = np.mean(elementary_effects)
        std_effect = np.var(elementary_effects)

        # Compute the sobol indices
        total_variance = np.var(original_rewards)
        mean_of_differences = np.mean(perturbed_rewards - original_rewards)
        first_order_variance = np.var(np.mean(perturbed_rewards) - np.mean(original_rewards))
        second_order_variance = np.var(perturbed_rewards - original_rewards)

        first_order_sobol = first_order_variance / total_variance
        second_order_sobol = second_order_variance / total_variance

        row = {
            "model": model,
            "principle": principle,
            "total_variance": total_variance,
            "first_order_sobol": first_order_sobol,
            "second_order_sobol": second_order_sobol,
            "mean_effect": mean_effect,
            "std_effect":std_effect
        }
        sobol_data.append(row)

        print(f"First order Sobol index: {first_order_sobol}")
        print(f"Second order Sobol index: {second_order_sobol}")

sobol_df = pd.DataFrame(sobol_data)

# Make plotly plot comparing the second order sobol indices for each principle for different models
fig = go.Figure()
for model in models:
    # Sum normalize
    model_data = sobol_df[sobol_df["model"] == model]
    y = model_data["mean_effect"]
    y_norm = y/y.sum()
    # y_norm = (y-y.min())/(y.max()-y.min())
    fig.add_trace(go.Bar(x=model_data["principle"], y=y_norm, name=model))
fig.update_layout(title="Normalized Mean effect",
                    xaxis_title="Constitutional Principle",
                    yaxis_title="Normalized Mean Effect")
# Export as html
fig.write_html("docs/_includes/mean_effect.html")
fig.show()

# Make a plot of the mean differences
fig = go.Figure()
for model in models:
    model_data = sobol_df[sobol_df["model"] == model]
    y = model_data["std_effect"]
    y_norm = y/y.sum()
    # y_norm = (y-y.min())/(y.max()-y.min())
    fig.add_trace(go.Bar(x=model_data["principle"], y=y_norm, name=model))
fig.update_layout(title="Normalised standard deviation of effect",
                    xaxis_title="Constitutional Principle",
                    yaxis_title="Normalized std effect")
# Export as html
fig.write_html("docs/_includes/std_effect.html")
fig.show()


