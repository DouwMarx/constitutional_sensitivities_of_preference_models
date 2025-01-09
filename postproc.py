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
fig.to_html("docs/images/plots/reward_box_plot.html")

fig.show()
