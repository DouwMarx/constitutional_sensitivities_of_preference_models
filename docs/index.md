### TLDR

|Constitutional   |sensitivities |of preference models |
|---|---|---|
| Principles that an AI system could adhere to. As in [Constitutional AI](https://www.anthropic.com/news/claudes-constitution). | How much does the output change when the input is changed according to a certain principle? | A model used to align an AI system with human preferences. Text goes in, one number comes out.|

A method to identify the principles in a constitution that a preference model is most sensitive to.

### Overview
<iframe src="images/diagrams/overview.drawio.html" width="100%" height="600" frameborder="0"></iframe>



### Introduction

[//]: # (*What question did you try to answer? You probably wrote this for Q4 of your project planning template.*)

[//]: # (*Why would it be useful to know the answer? &#40;Or: How is this relevant to the safety of advanced AI systems? Or: What motivated you to try to find an answer?&#41;*)

[//]: # (*What existing or related answers did you find to this question? Where can people find those answers? &#40;provide links or references&#41;. You probably wrote this for Q5 of your project planning template.*)

|   |
| :---- |

Similar to [1 @goCompositionalPreferenceModels2024] a global preference assesment is decomposed into a series of human interpretable features.
However, contraty to [1 @goCompositionalPreferenceModels2024] the global  preference is not decomposed on the basis of rankings of an helpful lmm based on a series of questions, but is instead based on the latent features of an llm.

Mention the same advantages that CPM's have and the additional advantages that OCPM's might have.

Ways in which the method is different from compositional preference models
 - You do not have to choose a fixed set of features, you can pick and choose which ones of them you would like to retain.
 - Here we choose to speak of a clause in a constritution rather than a feature like in the CPM framework


cite:LouyangTrainingLanguageModels2022  The instuct-gpt paper that explain the rlhf proses and they also have a nice diagram that I might borrow


### Methods

The code is available at [this repository](https://github.com/DouwMarx/constitutional_sensitivities_of_preference_models)


#### Reward models applied
  Three sequence-classifier-based reward models were selected from [Reward Bench](https://huggingface.co/spaces/allenai/reward-bench)
  Models are selected in the performance range of 70-80%, 80-90% and 90-100%.
The models and their respective rankings on reward  bench as of 2025-01-13 are given in the following table.

| Model                              | Reward Bench Score | Reward Bench Ranking | Safety Score | Reasoning Score |
|:-----------------------------------| :----------------- | :------------------- | :----------- | :-------------- |
| `some-person/some-model-hyperlink` | 0.75 | 1 | 0.8 | 0.7 |

Here I need to link to the paper and verify that the models are actually trained on the same dataset.

#### Constitutional perturbations through critique and revisions
The prompt templates use to perturb the inputs to the preference models are given below.
Based on https://python.langchain.com/docs/versions/migrating_chains/constitutional_chain/
I call this a constitutional perturbation, following terminology from sensitivity analysis.

##### **Critique Prompt Template**
```python
   {% include prompt_templates/critique_prompt_template.py %}
```

##### **Revision Prompt Template**
```python
   {% include prompt_templates/revision_prompt_template.py %}
```

##### Example of a "Constitutional perturbation"

{% include prompt_templates/example_prompt.md %}

#### Datasets


##### RLHF prompt dataset
<iframe src="https://huggingface.co/datasets/Anthropic/hh-rlhf/embed/viewer/default/test" width="100%" height="300px" frameborder="0"></iframe>
I used a part of the test set that I randomly shuffled
Dataset
Anthropic/hh-rlhf, "harmless-base" test data. I used the rejected samples, thinking that this would lead to the largest possible range over which the sensitivities can be measured. 

##### Collective constitutional AI dataset  
<iframe src="https://huggingface.co/datasets/douwmarx/ccai-dataset/embed/viewer/default/train" width="100%" height="600px" frameborder="0"></iframe>

They used PCA and then k-means to form the opinion groups
Consensus is the proportion of the group that agrees with the opinion 

I used the 10 constitutional principles with the highest consensus[^consensus] to form the opinion groups.
One of the principles is overlapping and removed
Normalized the sensitivities over all of the 18 principles.

[^consensus]: [Proportion of the group that agrees with the opinion](https://github.com/saffronh/ccai/blob/3ff5dce9a1299d6035f1dd9e2f95be995311cb6e/ccai_data_processing.ipynb#L874)


###### Source datasets
<table>
  <tr>
    <td width="50%">
      <strong>Anthropic hh-rlhf</strong><br>
      <iframe src="https://huggingface.co/datasets/Anthropic/hh-rlhf/embed/viewer/default/test" width="100%" height="600px" frameborder="0"></iframe>
    </td>
    <td width="50%">
      <strong>Collective constitutional AI</strong><br>
      <iframe src="https://huggingface.co/datasets/douwmarx/ccai-dataset/embed/viewer/default/train" width="100%" height="600px" frameborder="0"></iframe>
    </td>
  </tr>
</table>

###### Created Dataset
<iframe src="https://huggingface.co/datasets/douwmarx/hh-rlhf-constitutional-sensitivities-of-pms/embed/viewer/default/ccai_group_0" width="100%" height="600px"></iframe>

#### Sensitivity Study
[Global sensitivity analysis](https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis)
https://en.wikipedia.org/wiki/Morris_method MOris method as opposed to sobol method when  varying one effect at a time.
The Morris Method, also known as the "Morris Screening" or "Elementary Effects Method," is a type of global sensitivity analysis designed to identify inputs that have significant effects on the output, while also being relatively computationally inexpensive compared to full variance-based methods like Sobol'.
- The Morris method uses an efficient design of experiments to evaluate the "elementary effects" of input variables. An elementary effect is computed by perturbing one input variable at a time while holding the others constant.

- The ROC AUC is mathematically equivalent to the Wilcoxon-Mann-Whitney U statistic, scaled and averaged over all possible threshold values.
- Specifically, the value of the AUC represents the probability that a randomly chosen positive instance will be ranked higher than a randomly chosen negative instance.

##### Sensitivity metrics
Wilcoxon signed-rank statistic for the test
The Wilcoxon statistic itself is the sum of the ranks of differences that are positive. 
It's a measure of the tendency for one condition (say, post-treatment) to yield higher values than the other.
A larger Wilcoxon statistic suggests that there are more and/or larger positive differences, indicating a possible systematic increase from one condition to another.
Imagine you have a group of people who took a memory test before and after a training program. You want to know if the training had an effect:

- Calculate the difference in scores for each individual.
- Rank these differences, considering only positive ones for the statistic.
- A significantly large statistic compared to a critical value indicates a likely improvement thanks to the training.

By focusing on ranks rather than raw scores, the Wilcoxon signed-rank test avoids assumptions about data distribution, making it robust for small sample sizes or when normality cannot be assumed.

This test provides a practical way to understand how one condition might consistently lead to better (or worse) outcomes compared to another, particularly when your data come in naturally matched pairs.


The sensitivity indices for the different principles are sum-normalized such that the sum of the indices for all principles is equal to 1.
This way we can see the relative importance of the different principles according to the model. 

        "median_effect",
        "mean_effect",
        "std_effect",
        "mean_percentile_effect",
        "median_percentile_effect",
        "std_percentile_effect",
        "wilcoxon_statistic",
        "mannwhitneyu_statistic",
        "mannwhitneyu_p_value",
        "wilcoxon_p_value"

The sensitivity indexes that considered include the mean, median and standard deviation of the effects, the mean, median and standard deviation of the percentile effect over all evaluated prompts, the Wilcoxon signed-rank statistic and the Mann-Whitney U statistic.
The results for three of these indexes are shown below.


### Results

## Overall effect of critique revision perturbations

The reward values for the `Ray2333/GRM-Llama3.2-3B-rewardmodel-ft` model across all the evaluated query response pairs are shown below.
{% include original_and_perturbed_rewards.html %}
Mention that this could be due to the fact that the perturbed response is ultimately produced using an aligned model in this case.
Mention that the full axis is not shown and that the differences between the different constitutions are minor although they do seem to be shared amongst models. 


## Sensitivities of different models to constitutional perturbations
The results for mean effects sensitivity metric is shown below. 

### Mean effect sensitivity metric
Although there are clear differences in the sensitivity indexes across different principles, the variations of sensitivities for different models for a given principle is not very large.[^model_sensitivity]
Notice that the y-axis has been clipped to amplify the differences between the different principles.

[^model_sensitivity]: The similarity in sensitivity metrics is likely due to the fact that both models were trained on the same preference dataset. See the limitations section for more details.

{% include compare_model_sensitivity_mean_effect.html %}

### Other sensitivity metrics

[//]: # (Now we include the other plotly plots, but they will not have the same height. They will be smaller.)

[//]: # ({% include compare_model_sensitivity_median_effect.html %})

[//]: # ()
[//]: # ({% include compare_model_sensitivity_std_effect.html %})

[//]: # ()
[//]: # ({% include compare_model_sensitivity_mean_percentile_effect.html %})

[//]: # ()
[//]: # ({% include compare_model_sensitivity_median_percentile_effect.html %})

[//]: # ()
[//]: # ({% include compare_model_sensitivity_std_percentile_effect.html %})

[//]: # ()
[//]: # ({% include compare_model_sensitivity_wilcoxon_statistic.html %})

[//]: # ()

The results for some of the other sensitivity metrics are shown below.
It is concerning to see that they 

It could be that we are just looking at noise and that the gpt40 prompt is just maxing out on preference.

{% include compare_model_sensitivity_mannwhitneyu_statistic.html %}

<iframe src="compare_model_sensitivity_wilcoxon_statistic.html" width="100%" height="300px" frameborder="0"></iframe>
<iframe src="compare_model_sensitivity_median_effect.html" width="100%" height="300px" frameborder="0"></iframe>
<iframe src="compare_model_sensitivity_median_percentile_effect.html" width="100%" height="300px" frameborder="0"></iframe>

The different sensitivity metrics generally lead to a similar ranking of the "importance" of the different principles.
The mean should be robust to the non-linear nature of the preference model.


I would generally trust the wilcoxon statistic since it is a non-parametric test and does not assume normality of the data and it assummes the kinds of paired data with which we are working (i.e. start, treatment, end).


# Preference model sensitivities for constitutional principles associated with different groups.
For the CCAI dataset we took non-overlapping groups and compared their normalized sensitivities for the mean effects sensitivity metric.
Hover over the bar segments to see which principle they correspond to.

{% include compare_group_sensitivity_mean_effect.html %}

A small differences in the total percentage contribution can be seen between the different groups, suggesting that the preference model might be more sensitive to the principles associated with group 0.

### **Discussion**

#### Findings

#### Limitations
    It is reasonable to expect that there might be some overlap in for example helplessness and harmlessness.
    * The model that was used to perturb the promps was allready alligned, the perturbation is possibly not just usefull, but possibly also has its own ethical principles baked in of which kinds of perturbations are allowable. 
    * A serious limitation is the use of gpt40 for creating the constitutional perturbations. This model is alligned, and this means that an increase in preference could be due to adherence of the model to other aspects like for example style. The hope is that this improvement would happen across all principles and that the perturbation should still be usefull for identifying the principlse a given preference model is most sensitive to.
      * Occationally respons with "sure" here is the revised response
      * There is a big peak in the maxed out reward which makes sense 
   * You would not know if the sensitivity indexes translate to these behaviours actually being adhered to in the RLHF'ed model, and would now know unless it is tested.

   * The sensitivity analysis assumes independence of input features (presence of one constitutional principle does not affect the presence of another). This may not hold in practice, as some principles may be closely related.


### Future work
The biggest improvement to this work can be made by making use of a purely useful but  possibly harmfull model to measure the sensitivities.
The other big improvement would be to apply it with reward models that are trained on significantly different training sets
More rigorous methods of sensitivity analysis.


### Acknowledgements
I want to acknowledge the following people that contributed to this project:

| Person           | How they helped                                |
|:-----------------|:-----------------------------------------------|
| Cara Selvarajah  | Narrowing down topics and facilitating course. |
| Vicente Herrera  | Tokenization, Langchain and inference.         |
| Bluedot          | For the course                                 |

[//]: # (* *Check your article is written in [plain English]&#40;https://www.plainenglish.co.uk/how-to-write-in-plain-english.html&#41;. The [Hemingway Editor]&#40;https://hemingwayapp.com/&#41; and the [ONS’s editing guidance]&#40;https://service-manual.ons.gov.uk/content/writing-for-users/editing-and-proofreading&#41; can be useful for this.*  )
[//]: # ( [submit your project on the course hub]&#40;https://course.aisafetyfundamentals.com/alignment?tab=project&#41;\! 🎉)
