import numpy as np
import pandas as pd
import plotly
from plotly import graph_objects as go

# Make a box plot from the saved results

df = pd.read_csv("data/augmented_dataset.csv", index_col=0)


# Make a plotly box plot with the reward on the y
fig = go.Figure()
fig.add_trace(go.Box(y=np.log(df["original_reward"]), name="Original",boxpoints='all',jitter=0.3))
fig.add_trace(go.Box(y=np.log(df["perturbed_reward"]), name="Perturbed",boxpoints='all',jitter=0.3))
fig.update_layout(title="Original and Perturbed Rewards",
                    xaxis_title="Response Type",
                    yaxis_title="Log(Reward)")

# Export as html
fig.write_html("docs/images/plots/original_and_perturbed_rewards.html")

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

        original_rewards =np.log(original_rewards)
        perturbed_rewards = np.log(perturbed_rewards)

        # Compute the sobol indices
        total_variance = np.var(original_rewards)
        first_order_variance = np.var(np.mean(perturbed_rewards) - np.mean(original_rewards))
        second_order_variance = np.var(perturbed_rewards - original_rewards)

        first_order_sobol = first_order_variance / total_variance
        second_order_sobol = second_order_variance / total_variance

        row = {
            "model": model,
            "principle": principle,
            "total_variance": total_variance,
            "first_order_sobol": first_order_sobol,
            "second_order_sobol": second_order_sobol
        }
        sobol_data.append(row)

        print(f"First order Sobol index: {first_order_sobol}")
        print(f"Second order Sobol index: {second_order_sobol}")

sobol_df = pd.DataFrame(sobol_data)

# Make plotly plot comparing the second order sobol indices for each principle for different models
fig = go.Figure()
for model in models:
    model_data = sobol_df[sobol_df["model"] == model]
    fig.add_trace(go.Bar(x=model_data["principle"], y=model_data["second_order_sobol"], name=model))

fig.update_layout(title="Second Order Sobol Indices for each Constitutional Principle",
                    xaxis_title="Constitutional Principle",
                    yaxis_title="Second Order Sobol Index")

# Export as html
fig.write_html("docs/images/plots/second_order_sobol_indices.html")
fig.show()


