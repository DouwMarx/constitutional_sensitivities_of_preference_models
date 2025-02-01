[//]: # ({% include mermaid.html %})

[//]: # (# Constitutional Sensitivities of Reward Models)
## TLDR

This post proposes a method for measuring the sensitivities of large language model reward models to different principles from a constitution.

| Constitutional                                                                                                                                 | sensitivities                                                                                | of reward models                                                                               |
|------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| Principles that an AI system should adhere to. Constitutional, as in [Constitutional AI](https://www.anthropic.com/news/claudes-constitution). | How much does the output change when the input is changed according to a specific principle? | A model used to align an AI system with human preferences. Text goes in; one number comes out. |

Preliminary results show that PMs have different sensitivities to various constitutional principles and that one PM might be more sensitive to the constitutional principles characteristic of one group of people than another.

## Introduction
### RLHF
[Reinforcement learning from human feedback (RLHF)](https://huggingface.co/blog/rlhf) is a popular method for [aligning](https://en.wikipedia.org/wiki/AI_alignment) language models (LLMs) to generate text that is consistent with our values and preferences.

An essential part of this process involves teaching the LLM to "behave", similar to how you would teach a puppy to behave:
Ask for a paw - paw is given - reward with a treat.  
Ask not to rip the sofa to shreds - proceeds to rip the sofa to shreds - give a stern look.

The mapping between the dog's behaviour and the reward you give it is super important.
If you reward the dog for shredding the sofa and punish it for giving its paw, your dog will end up valuing things very differently from you.

Now exchange the dog for an LLM - "Hi. How can I help you?" should be rewarded, "Hi. Bugger off!" should not.

### Reward models
During training, untamed LLMs produce an overwhelming load of undesirable slop that cannot be judged manually in a single human lifetime.
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
    E --> F(0.1);
</div>

The reward model must produce big numbers for text we approve of and small values for text we would rather not see again.

Returning to our puppy training metaphor, a reward model is like a dog trainer.
They train the puppy on your behalf.
And you trust your dog trainer to do this well.
Dog trainers should not reward the puppy for ripping apart couches, right?
Similarly, you want to use the right LLM reward model when training an LLM.
That is reward models that reward generated text that is consistent with things you value.

## Sensitivities of reward models 
In my home, dogs are not allowed on the couch.
We can say that my dog reward protocol is *sensitive* to the principle: *"Dogs are not allowed on the couch"*.

Some dog trainers, however, are shamelessly *insensitive* to this principle, with no couch, bed or bathroom forbidden to their furry friends.

This post is about ensuring you hire the right dog trainer for you.

In LLM terms, this post is about measuring if the reward model you intend on using is sensitive to the principles you value.  
We say a model is sensitive to a principle if its output changes significantly when the input is changed according to that principle.

This method aims to be useful for identifying reward models that best align with a person's values and preferences.
Ultimately, this could lead to LLMs producing text that is better aligned with someone's personal values and preferences after training. 

### Constitutional principles of LLM's
We borrow the idea of a constitution from the [Constitutional AI paper](https://arxiv.org/abs/2212.08073).
A constitution is a set of principles that an AI system should adhere to.

Here's a set of principles that Bruce from Finding Nemo might see in his personalized LLM.

> **Bruce's Constitution:**
> - Principle 1: *"Fish are friends, not food."*
> - Principle 2: *"I am a nice shark, not a mindless eating machine."*

Anthropic speaks about their model constitution [here](https://www.anthropic.com/news/claudes-constitution).

### Constitutional perturbations
To measure the sensitivity of a function (like a reward model) we require a perturbation of the input to the model so that we can measure how much the output changes.

The input in this work is a natural language text prompt.
Creating perturbations is therefore not as simple as adding a small number to one of the inputs of the function.

Instead, we use a different LLM to modify an original prompt to create a perturbed prompt.
We do this using a critique and revision process similar to that used in the [Constitutional AI paper](https://arxiv.org/abs/2212.08073).

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
The following table gives the two models [models] and their respective performance on the [Reward Bench](https://huggingface.co/spaces/allenai/reward-bench) benchmark as of 2025-01-13.

[^models]: Reward models were chosen based on low parameter count and convenient inference through the Hugging Face API. See the limitations section for more details. 

| Model                                                                                                     | Reward Bench Ranking | Score | Chat | Chat Hard | Safety | Reasoning |
|:----------------------------------------------------------------------------------------------------------|:---------------------|-------|:-----|:----------|:-------|-----------|
| [`Ray2333/GRM-Llama3.2-3B-rewardmodel-ft`](https://huggingface.co/Ray2333/GRM-Llama3.2-3B-rewardmodel-ft) | 20                   | 90.9  | 91.6 | 84.9      | 92.7   | 94.5      |
| [`Ray2333/GRM-gemma2-2B-rewardmodel-ft`](https://huggingface.co/Ray2333/GRM-gemma2-2B-rewardmodel-ft)     | 33                   | 88.4  | 93.0 | 77.2      | 92.2   | 91.2      | 

 
    
[//]: # (Here I need to link to the paper and verify that the models are actually trained on the same dataset.)

### Constitutional perturbations through critique and revisions
The LLM used to perturb the prompts is OpenAi's [gpt-4o-mini](https://platform.openai.com/docs/models#gpt-4o-mini).[^gpt40]
The prompt templates used to perturb the initial prompts using a critique and revision step are given below.[^langchain]

[^gpt40]: Note that GPT-40-mini is an "aligned" model that, by its nature, should produce text that has high reward values. See the limitations section for more details.
[^langchain]: The LangGraph code is based on https://python.langchain.com/docs/versions/migrating_chains/constitutional_chain/.



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

[^perturbation_example]: The prompt and initial response are fictional. Bruce's values are [certainly not fictional](https://www.youtube.com/watch?v=kD8dHDpXVcI).

{% include prompt_templates/example_prompt.md %}

### Datasets
This section describes the two source datasets used in this work and the dataset that was created.
You can scroll through the datasets below using the horizontal scroll bar at the bottom of the dataset viewer.
All datasets are available for download on Hugging Face.

#### Source 1: RLHF prompt dataset

We use the [Anthropic/hh-rlhf](https://github.com/anthropics/hh-rlhf?tab=readme-ov-file), "harmless-base" data as the original prompt dataset.
Specifically, we use 200 samples labelled "rejected" from the test set as our original prompts.

[//]: # (. I used the rejected samples, thinking that this would lead to the largest possible range over which the sensitivities can be measured. )

<iframe src="https://huggingface.co/datasets/Anthropic/hh-rlhf/embed/viewer/default/test" width="100%" height="400px" frameborder="0"></iframe>


#### Source 2: Collective constitutional AI dataset  
We required a set of constitutional principles to perturb the prompts by critiquing and revising them.
We use the [Collective Constitutional AI](https://www.anthropic.com/news/collective-constitutional-ai-aligning-a-language-model-with-public-input) (CCAI) dataset from Anthropic and the Collective Intelligence project.
The CCAI [dataset](https://github.com/saffronh/ccai/) contains [responses](https://pol.is/report/r3rwrinr5udrzwkvxtdkj) from ~1,000 Americans to help draft the principles for a constitution.

The responses are [clustered into two groups by principal component analysis and k-means](https://github.com/saffronh/ccai/blob/main/ccai_data_processing.ipynb).
We measure the sensitivity of the reward models using the ten constitutional principles with the highest consensus[^consensus] scores from each group.

<iframe src="https://huggingface.co/datasets/douwmarx/ccai-dataset/embed/viewer/default/train" width="100%" height="400px" frameborder="0"></iframe>

[^consensus]: [Proportion of the group that agrees with the opinion](https://github.com/saffronh/ccai/blob/3ff5dce9a1299d6035f1dd9e2f95be995311cb6e/ccai_data_processing.ipynb#L874)

#### Dataset created for this work
After computing the reward values for the original and perturbed prompts, we create a dataset that contains the reward values for the different prompts.
The dataset is available at [douwmarx/hh-rlhf-constitutional-sensitivities-of-pms](https://huggingface.co/datasets/douwmarx/hh-rlhf-constitutional-sensitivities-of-pms). 
<iframe src="https://huggingface.co/datasets/douwmarx/hh-rlhf-constitutional-sensitivities-of-pms/embed/viewer/default/ccai_group_0" width="100%" height="600px"></iframe>

### Sensitivity Study
In this work, we attempt to apply a form of [global sensitivity analysis](https://en.wikipedia.org/wiki/Sensitivity_analysis) by changing One factor at a time (OAT) and measuring the effect on the output.
In this case, the "factors" are the different constitutional principles and the "output" is the reward value of the preference model.

We measure the  ["elementary effects"](https://en.wikipedia.org/wiki/Elementary_effects_method) of the different principles by calculating the difference in the reward values for the original and perturbed prompts.
An elementary effect should technically be computed by independently perturbing one input variable at a time while holding the others constant, but in our case, we are perturbing the prompts according to a single principle at a time.
Ideally, the inputs would be independent (the presence of one constitutional principle does not affect the presence of another), but this is unlikely to be the case for closely related principles.


#### Sensitivity metrics

A few different sensitivity measures are computed from the differences in the reward values for the original and perturbed prompts.
The simplest sensitivity metric is the mean effect, which is the average difference in reward values for the original and perturbed prompts.
Other sensitivity metrics include the median effect, the standard deviation of the effects, and the mean and median of the percentile effects.
We also computed the Wilcoxon signed-rank statistic, a nonparametric test that compares the differences between two conditions.

## Results
In this section, the results of the sensitivity study are presented.
First, the overall effect of the critique revision perturbations is shown.
Then, the sensitivities of different models to constitutional perturbations are compared.
Finally, the sensitivities of the preference models to constitutional principles associated with different groups are compared.

### Overall effect of critique revision perturbations

The probability density of the rewards for the `Ray2333/GRM-Llama3.2-3B-rewardmodel-ft` model across all the evaluated query-response pairs from the hh-rlhf dataset are shown below.
{% include original_and_perturbed_rewards.html %}

You can hover over the black dots in the plot to see the text of the prompt corresponding to a given reward value.
The results show that the critique revision perturbations generally lead to an increase in the reward values assigned by the preference model.
The general increase in reward values can plausibly be thanks to critique revision perturbations.
However, it is important to recognize that the perturbation is performed using an aligned model, meaning the responses it produces are likely to be high reward values (For example by producing well formatted text).

[//]: # (Mention that this could be due to the fact that the perturbed response is ultimately produced using an aligned model in this case.)

### Sensitivities of different models to constitutional perturbations
#### Mean effect sensitivity metric
The results for the mean effects sensitivity metric are shown below for the two models evaluated.
Larger values indicate that the preference model is more sensitive to the given principle.
Notice that the y-axis has been clipped to amplify the differences between the different principles.
Although there are clear differences in the sensitivity indexes across different principles, the differences in sensitivities for different models for a given principle are not very large.

The similarity in sensitivity metrics is likely due to the fact that both models were trained on the same preference dataset by the same author.
Nonetheless, the results suggest that the models have different sensitivities to the different constitutional principles and that the proposed method could be useful for identifying the principles to which a given preference model is most sensitive.

{% include compare_model_sensitivity_mean_effect.html %}

#### Other sensitivity metrics
The results for some of the other sensitivity metrics calculated are shown below.

{% include compare_model_sensitivity_wilcoxon_statistic.html %}
{% include compare_model_sensitivity_median_effect.html %}
{% include compare_model_sensitivity_median_percentile_effect.html %}

Different sensitivity metrics generally lead to a similar ranking of the "importance" of the different principles.
However, there are clear differences in the sensitivity indexes across different principles.
Ultimately, the most suitable sensitivity metric would have to be identified by using the model to RLHF an LLM.
Human evaluators would then have to evaluate which sensitivity metric best translate to RLHF'ed model behaviour.

### Preference model sensitivities for constitutional principles associated with different groups.
Sensitivity metrics for the non-overlapping principles associated with the two groups in the CCAI dataset are shown below.
The sensitivity metrics are sum-normalized over all principles.
You can hover over the bar segments to see which principle they are associated with.

{% include compare_group_sensitivity_mean_effect.html %}

Small differences in the total sum of sensitivities can be seen when comparing the two groups
This suggests that the preference model evaluated might be more sympathetic to the principles associated with group 0.

## Conclusion
- This work proposes a method for measuring the sensitivities of large language model reward models to different principles from a constitution.
- The method involves perturbing a dataset of prompts according to different principles and measuring the effect on the reward model output.
- The results show that different reward models have different sensitivities to various constitutional principles and that the proposed method could be useful for identifying the principles that a given preference model is most sensitive to.

### Limitations
-  The LLM that was used to perturb prompts is aligned, meaning that increased reward values could be due to aspects unrelated to the constitutional principles like style and formatting. Using a purely useful but possibly harmful model to measure the sensitivities would be better. This way, the perturbations could also be made in a negative direction.
-  Occasionally, the chatbot-like responses of GPT-40 contaminated the perturbed prompts with irrelevant information. This could have affected the reward values of the perturbed prompts.
- High sensitivity to a given principle does not necessarily translate to strict adherence to that principle after RLHF'ing an LLM. Human evaluators would need to evaluate the relationship between the sensitivities and the behaviour of the RLHF'ed LLM.

## Future work
Sensitivities to different constitutions can possibly be used to regularise reward models such that different constitutional principles are orthogonal to each other.
This could allow users to choose the principles they value most and create a reward model that is suited to their values and preferences.
This proposal could serve as an extension to [compositional preference models](https://arxiv.org/abs/2310.13011).

## Acknowledgements

|                 |                                                                                              |
|:----------------|:---------------------------------------------------------------------------------------------|
| Cara Selvarajah | Narrowing down topics and facilitating the course.                                           |
| Vicente Herrera | Advice on tokenization, Langchain and inference.                                             |
| Bluedot         | [For hosting the Technical AI alignment course](https://aisafetyfundamentals.com/alignment/) |

[//]: # (* *Check your article is written in [plain English]&#40;https://www.plainenglish.co.uk/how-to-write-in-plain-english.html&#41;. The [Hemingway Editor]&#40;https://hemingwayapp.com/&#41; and the [ONS’s editing guidance]&#40;https://service-manual.ons.gov.uk/content/writing-for-users/editing-and-proofreading&#41; can be useful for this.*  )
[//]: # ( [submit your project on the course hub]&#40;https://course.aisafetyfundamentals.com/alignment?tab=project&#41;\! 🎉)
