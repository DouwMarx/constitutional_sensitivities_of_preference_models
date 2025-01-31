{% include mermaid.html %}

# Constitutional Sensitivities of Reward Models
## TLDR

[//]: # (This post proposes a method for identifying the principles in a constitution that a large language model &#40;LLM&#41; preference model &#40;PM&#41; is most sensitive to.)
This post proposes a method for measuring the sensitivities of large language model reward models to different principles from a constitution.

| Constitutional                                                                                                                                 | sensitivities                                                                                | of reward models                                                                               |
|------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| Principles that an AI system should adhere to. Constitutional, as in [Constitutional AI](https://www.anthropic.com/news/claudes-constitution). | How much does the output change when the input is changed according to a specific principle? | A model used to align an AI system with human preferences. Text goes in; one number comes out. |

Preliminary results show that PM's have different sensitivities to various constitutional principles, and that one PM might be more sensitive to the constitutional principles charasteristic on one group of people compared to another.

## Introduction

### RLHF
[Reinforcement learning from human feedback (RLHF)](https://huggingface.co/blog/rlhf) is a popular method for [aligning](https://en.wikipedia.org/wiki/AI_alignment) language models (LLMs) to generate text that is consistent with our values and preferences.

An important part of this process involves teaching the LLM to "behave", similar to how you would teach a puppy to behave:
Ask for paw - paw is given - reward with treat.  
Ask not to rip up the couch - proceeds to rips up the couch - give stern look (This is the third time this week Rover!).

The mapping between the dog's behaviour and reward you give it is super important.
If you reward the dog for ripping up the couch and punish it for giving its paw, your dog will end up valuing very different from you.

Now exchange the dog for an LLM - "Hi. How can I help you?" should be rewarded, "Hi. Bugger off!" should not.

### Reward models
During training, untamed LLM's produce an overwhelming load of undesirable slop that cannot manually be judged in a single human lifetime.
So, we use rewards models[^reward_models] to do this on our behalf.
For the purpose of this work, a reward model is basically a function.
Text goes in. 
One number comes out.

[^reward_model]: The reward model is also known as a preference model.

<div class="mermaid" markdown="0" >
graph LR;
    A["''Hi. How can I help you?''"] --> B("Preference Model");
    B --> C(0.8);

    D["''Hi. Bugger off!''"] --> E("Preference Model");
    E --> F(0.1);`
</div>

It is important that the reward model produces big numbers for text we approve of and small values for text we would rather not see again.

Returning to our puppy training metaphor: A reward model is like a dog trainer.
They train the puppy on your behalf.
And you trust your dog trainer to do this well.
Dog trainers should not reward the puppy for ripping apart couches, right?
Similarly, you want to use the right LLM reward model when training an LLM.
That is, reward models that reward generated text that is consistent with things you value.

## Sensitivities of reward models 
In my home, dogs are not allowed on the couch.
We can say that my dog reward protocol is *sensitive* to the principle: *"Dogs are not allowed on the couch"*.

Some dog trainers are, however, shamelessly *insensitive* to this principle, with no couch, bed or bathroom forbidden to their furry friends.

This post is about ensuring you hire the right dog trainer for you.

In LLM terms, this post is about measuring if the reward model you intend on using is sensitive to the principles you value.  
We say a model is sensitive to a principle if it its output changes significantly when the input is changed according a given principle.

This method aims be useful for identifing reward models that are best aligned with someone's values and preferences.
Ultimately, this could lead to LLM's after training that produce text better aligned with someone's personal values and preferences. 

### Constitutional principles of LLM's


We borrow the idea of a constitution from the [Constitutional AI paper](https://arxiv.org/abs/2212.08073).
A constitution is a set of principles that and AI system should adhere to.

Here's a set of principles that Bruce from Finding Nemo might to see in his personalized LLM.

> **Bruce's Constitution:**
> - Principle 1: *"Fish are friends, not food."*
> - Principle 2: *"I am a nice shark, not a mindless eating machine."*

Anthropic speaks about their models constitution [here](https://www.anthropic.com/news/claudes-constitution).

### Constitutional perturbations
To measure the sensitivity of a function (like a reward model) we require a perturbation of the input to the model so that we can measure how much the output changes.

The input in this work is natural language text prompt.
Creating perturbations is therefore not as simple as adding a small number to one of the inputs of the function.

Instead, we use a different LLM to modify an original prompt to create a perturbed prompt.
We do this using a sprocess of critique and revision similar to that used in the [Constitutional AI paper](https://arxiv.org/abs/2212.08073).

## Overview of method

The method we proposed is summarized in the following diagram.

<iframe src="images/diagrams/overview.drawio.html" width="100%" height="600" frameborder="0"></iframe>

The method involves the following steps:
1. Select a set of constitutional principles.
2. For each principle:
  - Perturb a dataset of prompts to create a set of perturbed prompts that adhere to the principle.
  - Evaluate the reward model on the original and perturbed prompts.
  - Calculate the sensitivity of the reward model to the principle using a sensitivity metric.
3. Compare the sensitivities of the reward model to the different principles.

The technical details of the method are given in the following sections.

## Methods

*Code for this project is available [here](https://github.com/DouwMarx/constitutional_sensitivities_of_preference_models)*


### Reward models
The two models[^models] and their respective performance on the [Reward Bench](https://huggingface.co/spaces/allenai/reward-bench)  benchmark as of 2025-01-13 are given in the following table.

[^models]: Reward models were mainly chosen based on low parameter count and convenient inference through the Hugging Face API. See limitations section for more details. 

| Model                                                                                                     | Reward Bench Ranking | Score | Chat | Chat Hard | Safety | Reasoning |
|:----------------------------------------------------------------------------------------------------------|:---------------------|-------|:-----|:----------|:-------|-----------|
| [`Ray2333/GRM-Llama3.2-3B-rewardmodel-ft`](https://huggingface.co/Ray2333/GRM-Llama3.2-3B-rewardmodel-ft) | 20                   | 90.9  | 91.6 | 84.9      | 92.7   | 94.5      |
| [`Ray2333/GRM-gemma2-2B-rewardmodel-ft`](https://huggingface.co/Ray2333/GRM-gemma2-2B-rewardmodel-ft)     | 33                   | 88.4  | 93.0 | 77.2      | 92.2   | 91.2      | 

 
    
[//]: # (Here I need to link to the paper and verify that the models are actually trained on the same dataset.)

### Constitutional perturbations through critique and revisions
The prompt templates use to perturb the initial prompts using a critique and revision step is given below.[^langchain]

[^langchain]: The LangGraph code to do this is based on https://python.langchain.com/docs/versions/migrating_chains/constitutional_chain/.


#### Critique Prompt Template

```python
   {% include prompt_templates/critique_prompt_template.py %}
```

#### Revision Prompt Template

```python
   {% include prompt_templates/revision_prompt_template.py %}
```

#### "Constitutional perturbation" example

An example of a constitutional perturbation is given in the following table.[^perturbation_example]
The constitutional principle is Bruce's first principle: *"Fish are friends, not food."*

[^perturbation_example]: The prompt and initial response is fictional. Bruce's values are [certainly not fictional](https://www.youtube.com/watch?v=kD8dHDpXVcI).

{% include prompt_templates/example_prompt.md %}

### Datasets
Two source datasets were used in this work.
After computing the reward values for the original and perturbed prompts, a third dataset was created to store original prompt, perturbed prompt and the reward values for each prompt for a given model.
You scroll through the datasets below.
All datasets are available for download on Hugging Face.

#### Source 1: RLHF prompt dataset
<iframe src="https://huggingface.co/datasets/Anthropic/hh-rlhf/embed/viewer/default/test" width="100%" height="400px" frameborder="0"></iframe>

Anthropic/hh-rlhf, "harmless-base" test data. I used the rejected samples, thinking that this would lead to the largest possible range over which the sensitivities can be measured. 

#### Source 2: Collective constitutional AI dataset  
<iframe src="https://huggingface.co/datasets/douwmarx/ccai-dataset/embed/viewer/default/train" width="100%" height="400px" frameborder="0"></iframe>

They used PCA and then k-means to form the opinion groups
Consensus is the proportion of the group that agrees with the opinion 

I used the 10 constitutional principles with the highest consensus[^consensus] to form the opinion groups.
One of the principles is overlapping and removed
Normalized the sensitivities over all of the 18 principles.

[^consensus]: [Proportion of the group that agrees with the opinion](https://github.com/saffronh/ccai/blob/3ff5dce9a1299d6035f1dd9e2f95be995311cb6e/ccai_data_processing.ipynb#L874)

#### Created Dataset
<iframe src="https://huggingface.co/datasets/douwmarx/hh-rlhf-constitutional-sensitivities-of-pms/embed/viewer/default/ccai_group_0" width="100%" height="600px"></iframe>

### Sensitivity Study
[Global sensitivity analysis](https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis)
https://en.wikipedia.org/wiki/Morris_method MOris method as opposed to sobol method when  varying one effect at a time.
The Morris Method, also known as the "Morris Screening" or "Elementary Effects Method," is a type of global sensitivity analysis designed to identify inputs that have significant effects on the output, while also being relatively computationally inexpensive compared to full variance-based methods like Sobol'.
- The Morris method uses an efficient design of experiments to evaluate the "elementary effects" of input variables. An elementary effect is computed by perturbing one input variable at a time while holding the others constant.

- The ROC AUC is mathematically equivalent to the Wilcoxon-Mann-Whitney U statistic, scaled and averaged over all possible threshold values.
- Specifically, the value of the AUC represents the probability that a randomly chosen positive instance will be ranked higher than a randomly chosen negative instance.

#### Sensitivity metrics
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


## Results

### Overall effect of critique revision perturbations

The reward values for the `Ray2333/GRM-Llama3.2-3B-rewardmodel-ft` model across all the evaluated query response pairs are shown below.
{% include original_and_perturbed_rewards.html %}
Mention that this could be due to the fact that the perturbed response is ultimately produced using an aligned model in this case.
Mention that the full axis is not shown and that the differences between the different constitutions are minor although they do seem to be shared amongst models. 


### Sensitivities of different models to constitutional perturbations
The results for mean effects sensitivity metric is shown below. 

#### Mean effect sensitivity metric
Although there are clear differences in the sensitivity indexes across different principles, the variations of sensitivities for different models for a given principle is not very large.[^model_sensitivity]
Notice that the y-axis has been clipped to amplify the differences between the different principles.

[^model_sensitivity]: The similarity in sensitivity metrics is likely due to the fact that both models were trained on the same preference dataset. See the limitations section for more details.

{% include compare_model_sensitivity_mean_effect.html %}

#### Other sensitivity metrics
The results for some of the other sensitivity metrics are shown below.
It is concerning to see that they 

It could be that we are just looking at noise and that the gpt40 prompt is just maxing out on preference.

{% include compare_model_sensitivity_wilcoxon_statistic.html %}
{% include compare_model_sensitivity_median_effect.html %}
{% include compare_model_sensitivity_median_percentile_effect.html %}


The different sensitivity metrics generally lead to a similar ranking of the "importance" of the different principles.
The mean should be robust to the non-linear nature of the preference model.


I would generally trust the wilcoxon statistic since it is a non-parametric test and does not assume normality of the data and it assummes the kinds of paired data with which we are working (i.e. start, treatment, end).


### Preference model sensitivities for constitutional principles associated with different groups.
For the CCAI dataset we took non-overlapping groups and compared their normalized sensitivities for the mean effects sensitivity metric.
Hover over the bar segments to see which principle they correspond to.

{% include compare_group_sensitivity_mean_effect.html %}

A small differences in the total percentage contribution can be seen between the different groups, suggesting that the preference model might be more sensitive to the principles associated with group 0.

## Discussion

### Findings

### Limitations
    It is reasonable to expect that there might be some overlap in for example helplessness and harmlessness.
    * The model that was used to perturb the promps was allready alligned, the perturbation is possibly not just usefull, but possibly also has its own ethical principles baked in of which kinds of perturbations are allowable. 
    * A serious limitation is the use of gpt40 for creating the constitutional perturbations. This model is alligned, and this means that an increase in preference could be due to adherence of the model to other aspects like for example style. The hope is that this improvement would happen across all principles and that the perturbation should still be usefull for identifying the principlse a given preference model is most sensitive to.
      * Occationally respons with "sure" here is the revised response
      * There is a big peak in the maxed out reward which makes sense 
   * You would not know if the sensitivity indexes translate to these behaviours actually being adhered to in the RLHF'ed model, and would now know unless it is tested.

   * The sensitivity analysis assumes independence of input features (presence of one constitutional principle does not affect the presence of another). This may not hold in practice, as some principles may be closely related.


## Future work
The biggest improvement to this work can be made by making use of a purely useful but  possibly harmfull model to measure the sensitivities.
The other big improvement would be to apply it with reward models that are trained on significantly different training sets
More rigorous methods of sensitivity analysis.


## Acknowledgements
I want to acknowledge the following people that contributed to this project:

| Person           | How they helped                                |
|:-----------------|:-----------------------------------------------|
| Cara Selvarajah  | Narrowing down topics and facilitating course. |
| Vicente Herrera  | Tokenization, Langchain and inference.         |
| Bluedot          | For the course                                 |

[//]: # (* *Check your article is written in [plain English]&#40;https://www.plainenglish.co.uk/how-to-write-in-plain-english.html&#41;. The [Hemingway Editor]&#40;https://hemingwayapp.com/&#41; and the [ONS’s editing guidance]&#40;https://service-manual.ons.gov.uk/content/writing-for-users/editing-and-proofreading&#41; can be useful for this.*  )
[//]: # ( [submit your project on the course hub]&#40;https://course.aisafetyfundamentals.com/alignment?tab=project&#41;\! 🎉)
