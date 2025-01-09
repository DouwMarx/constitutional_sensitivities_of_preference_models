import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Make a box plot from the saved results

df = pd.read_csv("data/rewards_new.csv", index_col=0)

# Make a seaborn boxplot
# sns.boxplot(data=df, palette="Set2",log_scale=True)
sns.stripplot(data=df, log_scale= True)
# sns.swarmplot(data=df,log_scale=True)
# sns.violinplot(data=df, palette="Set2",log_scale=True,inner="stick")
# sns.swarmplot(data=df, log_scale=True)
# sns.violinplot(data=df, palette="Set2",log_scale=True)
# Ylabel is reward


plt.ylabel("Log reward (Ray2333/GRM-Llama3.2-3B-rewardmodel-ft)")
plt.xlabel("Message label: (Anthropic/hh-rlhf, harmless-base)")
plt.show()
plt.savefig("reports/rewards_on_hhrlhf.png")