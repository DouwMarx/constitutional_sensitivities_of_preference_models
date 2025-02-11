{% include mermaid.html %}

[//]: # (# Constitutional Sensitivities of Reward Models)
## Summary

This post proposes a method for measuring the sensitivities of large language model (LLM) reward models (RMs) to different principles from a constitution.

| Constitutional                                                                                                                                 | sensitivities                                                                                | of reward models                                                                               |
|------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| Principles that an AI system should adhere to. Constitutional, as in [Constitutional AI](https://www.anthropic.com/news/claudes-constitution). | How much does the output change when the input is changed according to a specific principle? | A model used to align an AI system with human preferences. Text goes in; one number comes out. |

Preliminary results show that RMs have different sensitivities to various constitutional principles and that one RM might be more sensitive to the constitution of one group of people than another.

## Introduction
### RLHF

[Reinforcement learning from human feedback (RLHF)](https://huggingface.co/blog/rlhf) is a popular method for [aligning](https://en.wikipedia.org/wiki/AI_alignment) language models (LLMs) to generate text that reflects our values and preferences.

Part of this process involves teaching the LLM to "behave", similar to how you would teach a puppy to behave:
Ask for a paw - paw is given - reward with a treat.  
Ask not to rip the sofa to shreds - proceeds to rip the sofa to shreds - give a stern look.

The mapping between the dog's behaviour and the reward given is super important.
If you reward the dog for shredding the sofa and punish it for giving its paw, your dog will end up valuing things very differently from you.

Exchange the dog for an LLM - "Hi. How can I help you?" should be rewarded, "Hi. Bugger off!" should not.
Like the puppy, the LLM is learning from "human feedback".

### Reward models

During training by RLHF, untamed LLMs produce an overwhelming load of slop that cannot be judged manually in a single human lifetime.
So, rewards models[^reward_model] are used to give feedback on our behalf.
In this work, a reward model is a function that measures how preferable a chunk of text is.
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
You trust your dog trainer to do this well.
Dog trainers should not reward the puppy for ripping apart couches, right?

Similarly, the right LLM reward model should be used when training an LLM.
People have different values and preferences and should be able to use reward models that reward generated text that is consistent with things they value.

### Sensitivities of reward models 

In my home, dogs are not allowed on the couch.
We can say that my dog rewarding protocol is *sensitive* to the principle: *"Dogs are not allowed on the couch"*.

Some dog trainers, however, are shamelessly *insensitive* to this principle, with no couch, bed or bathroom forbidden to their furry friends.

This work is about hiring the right dog trainer for you.

In LLM terms, this post is about measuring if the reward model you intend to use for RLHF is sensitive to the principles you value.  
We say a model is sensitive to a given principle if its output changes significantly when the input is changed according to that principle.

### Why it is nice to know the sensitivity of a reward model to different principles.
This method aims to help identify a reward model that best aligns with a person's values and preferences.
It could further be useful as a cheap way to measure whether reward models are adhering to the principles we want them to follow or to diagnose possible causes of misalignment. 
Ultimately, this could lead to LLMs producing text that is better aligned with someone's values and preferences after training. 

### Constitutional principles in LLMs

We borrow the idea of a constitution from the [Constitutional AI paper](https://arxiv.org/abs/2212.08073).
A constitution is a set of principles that an AI system should adhere to.
In this work, we measure the sensitivity of a reward model with respect to one of these principles.

Here's a set of principles that Bruce from Finding Nemo might like to see in his personalised LLM.

> **Bruce's Constitution:**
> - Principle 1: *"Fish are friends, not food."*
> - Principle 2: *"I am a nice shark, not a mindless eating machine."*

A real-life example is [Anthropic's constitution](https://www.anthropic.com/news/claudes-constitution) for the Claude model.

### Constitutional perturbations

To measure the sensitivity of a function (like a reward model), we typically require a perturbation (small change or deviation) of the model's input.
This way the extent to which the output changes for a given change in the input can be measured (the sensitivity).

The input to the reward models in this work is a natural language text prompt.
Therefore, perturbing the input is not as simple as adding a small number to it.

Instead, a different LLM is used to modify the original prompt according to the constitutional principle to create a perturbed prompt.
We do this using a critique and revision process similar to that used in the [Constitutional AI paper](https://arxiv.org/abs/2212.08073).

## Methods

*Code for this project is available [here](https://github.com/DouwMarx/constitutional_sensitivities_of_preference_models)*

### Overview of method

The method we proposed is summarised in the following diagram.

<iframe src="images/diagrams/overview.drawio.html" width="100%" height="600" frameborder="0"></iframe>

The method involves the following steps:
1. Select a set of constitutional principles.
2. For each principle:
  - Perturb a dataset of prompts to create a set of perturbed prompts that adhere to the principle.
  - Evaluate the reward model on the original and perturbed prompts.
  - Calculate the sensitivity of the reward model to the principle using a sensitivity metric.
3. Normalise the sensitivity metrics across principles
4. Compare the sensitivities of the reward model to different principles.

The technical details of the method are given in the following sections.


### Reward models

The following table shows the two reward models used and their respective performance on the [Reward Bench](https://huggingface.co/spaces/allenai/reward-bench) benchmark as of 2025-01-31.

[^models]: Reward models were chosen based on low parameter count and convenient inference through the Hugging Face API. See the limitations section for more details. 

| Model                                                                                                     | Reward Bench Ranking | Score | Chat | Chat Hard | Safety | Reasoning |
|:----------------------------------------------------------------------------------------------------------|:---------------------|-------|:-----|:----------|:-------|-----------|
| [`Ray2333/GRM-Llama3.2-3B-rewardmodel-ft`](https://huggingface.co/Ray2333/GRM-Llama3.2-3B-rewardmodel-ft) | 20                   | 90.9  | 91.6 | 84.9      | 92.7   | 94.5      |
| [`Ray2333/GRM-gemma2-2B-rewardmodel-ft`](https://huggingface.co/Ray2333/GRM-gemma2-2B-rewardmodel-ft)     | 33                   | 88.4  | 93.0 | 77.2      | 92.2   | 91.2      | 

 

### Constitutional perturbations through critique and revisions

The LLM used to perturb the prompts is OpenAi's [gpt-4o-mini](https://platform.openai.com/docs/models#gpt-4o-mini).[^gpt40]
The prompt templates used to perturb the initial prompts using a critique and revision step are given below.[^langchain]

[^gpt40]: Note that GPT-40-mini is an "aligned" model that, by its nature, should produce text that has high reward values. See the limitations section for more details.
[^langchain]: The LangGraph code used to do this is based on https://python.langchain.com/docs/versions/migrating_chains/constitutional_chain/.

#### Critique Prompt Template

```python
   {% include prompt_templates/critique_prompt_template.py %}
```

#### Revision Prompt Template

```python
   {% include prompt_templates/revision_prompt_template.py %}
```

#### "Constitutional perturbation" example

An example of a constitutional perturbation is given in the following table.
The constitutional principle is Bruce's first principle: *"Fish are friends, not food."* [^perturbation_example]
The model is then asked to critique and revise the prompt according to this principle.

[^perturbation_example]: The prompt and initial response are fictional. Bruce's values are [certainly not fictional](https://www.youtube.com/watch?v=kD8dHDpXVcI).

{% include prompt_templates/example_prompt.md %}

### Datasets

This section describes the two source datasets used in this work and the dataset that was created from them.
You can scroll through the datasets below using the horizontal scroll bar at the bottom of the dataset viewer.
All datasets are available for download on Hugging Face.

#### Source 1: RLHF prompt dataset

We use the [Anthropic/hh-rlhf](https://github.com/anthropics/hh-rlhf?tab=readme-ov-file), "harmless-base" data as the original prompt dataset.
Specifically, we use 200 samples labelled "rejected"[^rejected] from the test set as our original prompts.

[^rejected]: The "rejected" samples are used to try to mitigate the effects of using an aligned model to perturb the prompts. An aligned model can make bad prompts better but will not make good prompts worse.

<iframe src="https://huggingface.co/datasets/Anthropic/hh-rlhf/embed/viewer/default/test" width="100%" height="400px" frameborder="0"></iframe>


#### Source 2: Collective constitutional AI dataset  

We require a set of constitutional principles for which the sensitivity of the reward models can be measured.
The [Collective Constitutional AI](https://www.anthropic.com/news/collective-constitutional-ai-aligning-a-language-model-with-public-input) (CCAI) dataset from Anthropic and the Collective Intelligence project is used to provide these principles.
The CCAI [dataset](https://github.com/saffronh/ccai/) contains [responses](https://pol.is/report/r3rwrinr5udrzwkvxtdkj) from ~1,000 Americans to help draft the principles for a LLM constitution.

The principles are [clustered into two groups by principal component analysis and k-means](https://github.com/saffronh/ccai/blob/main/ccai_data_processing.ipynb).
We measure the sensitivity of the reward models using the ten constitutional principles with the highest consensus[^consensus] scores from each group.

<iframe src="https://huggingface.co/datasets/douwmarx/ccai-dataset/embed/viewer/default/train" width="100%" height="400px" frameborder="0"></iframe>

[^consensus]: [Proportion of the group that agrees with the opinion](https://github.com/saffronh/ccai/blob/3ff5dce9a1299d6035f1dd9e2f95be995311cb6e/ccai_data_processing.ipynb#L874)

#### Dataset created for this work

We create a dataset that contains the reward values for the original and perturbed prompts.
The dataset is available at [douwmarx/hh-rlhf-constitutional-sensitivities-of-pms](https://huggingface.co/datasets/douwmarx/hh-rlhf-constitutional-sensitivities-of-pms)  and includes the critiques and revision steps for each prompt.
The dataset is split into two groups, one for each of the two groups of constitutional principles from the CCAI dataset.
<iframe src="https://huggingface.co/datasets/douwmarx/hh-rlhf-constitutional-sensitivities-of-pms/embed/viewer/default/ccai_group_0" width="100%" height="600px"></iframe>

### Sensitivity Study

In this work, we attempt to apply a form of [global sensitivity analysis](https://en.wikipedia.org/wiki/Sensitivity_analysis) to the preference model by changing One Factor at a Time (OAT) and measuring the effect on the output.
The "factors" are the different constitutional principles, and the "output" is the reward value of the preference model.

We measure the ["elementary effects"](https://en.wikipedia.org/wiki/Elementary_effects_method) of each principle by calculating the difference in the reward values for the original and perturbed prompts over the full dataset.
Technically, an elementary effect should be computed by independently perturbing one input variable at a time while holding the others constant, but in this case, the prompts are perturbed according to a single principle at a time.
Ideally, the inputs would be independent (the presence of one constitutional principle does not affect the presence of another), but this is unlikely to be the case for closely related principles.

#### Sensitivity metrics

Different sensitivity measures are computed from original and perturbed prompts.
The simplest sensitivity metric is the mean effect, which is the average difference in reward values for the original and perturbed prompts.
Other sensitivity metrics include the median effect, the standard deviation of the effects, and the mean and median of the percentile effects.
We also computed the Wilcoxon signed-rank statistic, a nonparametric test that compares the differences between two conditions after an intervention (The perturbation of the prompt).

After computing the sensitivity metrics for each principle, the sensitivity metrics are sum normalised across principles to allow for comparison between the different principles.
This means that all the sensitivity metrics add up to 1 for a given model.

## Results

The results are now presented.
First, the overall effect of the critique revision perturbations on the reward model output is shown.
Then, the sensitivities of different models to the same constitutional perturbations are compared.
Finally, the sensitivities of the preference models to constitutional principles associated with different groups are compared.

### Overall effect of critique revision perturbations

A histogram of the rewards for the `Ray2333/GRM-Llama3.2-3B-rewardmodel-ft` model across all the evaluated query-response pairs from the hh-rlhf dataset is shown below.
{% include original_and_perturbed_rewards.html %}

Hover over the black dots in the plot to see the input text corresponding to a given reward value.
The plot show that the critique-revision perturbations generally lead to an increase in the reward values assigned by the reward model.
However, it is important to recognise that the perturbation is performed using an aligned model, meaning the responses it produces are likely to be highly rewarded (for example, by producing well-formatted text).


### Sensitivities of different models to constitutional perturbations
#### Mean effect sensitivity metric

The mean effects sensitivity metric results for the two models evaluated are shown below.

{% include compare_model_sensitivity_mean_effect.html %}

Larger values indicate that the preference model is more sensitive to a given principle.
Notice that the y-axis has been clipped to magnify the differences between the different principles.

According to the mean effects sensitivity metric, the evaluated models are most sensitive to the principles *"The AI should have good qualities"* and *"The AI should tell the truth"*.

Although there are differences in the sensitivity indexes across different principles, the differences in sensitivities between the two models for a given principle are not very large.

The similarity in sensitivity metrics for the two models is likely because both models were trained on the same preference dataset by the same author.[^same_author]
Nonetheless, the results suggest that the models have different sensitivities to the different constitutional principles and that the proposed method could be useful for identifying the principles to which a given preference model is most sensitive.

[^same_author]: See the limitations section for a link to the paper.

#### Other sensitivity metrics

The results for two of the other sensitivity metrics calculated are shown below.

{% include compare_model_sensitivity_wilcoxon_statistic.html %}
{% include compare_model_sensitivity_median_percentile_effect.html %}

Different sensitivity metrics generally lead to a similar ranking of the "importance" of the different principles.
However, there are still clear differences in the sensitivity indexes across different principles.
The most suitable sensitivity metric for characterising reward models would likely have to be identified by human evaluators.
They would have to rate the effectiveness of the sensitivity metric after interacting with LLMs that have been RLHF'ed with different reward models.

### RM sensitivities for constitutional principles associated with different groups.

Sensitivity metrics for non-overlapping principles associated with the two groups in the CCAI dataset are shown below.

{% include compare_group_sensitivity_mean_effect.html %}

The sensitivity metrics are sum-normalised across all principles.
Hover over the bar segments to see which constitutional principle they are associated with.

Small differences in the total sum of sensitivities can be seen when comparing the two groups.
This suggests that the preference model evaluated might be more sympathetic to the principles associated with group 0.

## Conclusion

- This work proposes a method for measuring the sensitivities of large language model reward models to different principles from a constitution.
- The method involves perturbing a dataset of prompts according to different principles through critique and revision, and measuring the effect of the perturbation on the reward model output.
- The results show that reward models have different sensitivities to various constitutional principles and that the proposed method could be useful for identifying the principles that a given preference model is most sensitive to.

## Limitations

-  The LLM that was used to perturb prompts is aligned, meaning that increased reward values could be due to aspects unrelated to the constitutional principles like style and formatting. Using a purely useful but possibly harmful model to measure the sensitivities would be better. This way, the perturbations could also be made in a negative direction.
-  Occasionally, the chatbot-like responses of GPT-40 contaminated the perturbed prompts with irrelevant information. This could have affected the reward values of the perturbed prompts.
- The two reward models used were trained by [the same author](https://arxiv.org/pdf/2406.10216) using the same dataset.
- The effectiveness of the method is likely dependent on the dataset that is used to evaluate the reward model sensitivities. If the dataset does not contain any prompts that are incompatible with a given principle, the sensitivity of the reward model to that principle will not easily be measured.
- High sensitivity to a given principle does not necessarily translate to strict adherence to that principle after RLHF'ing an LLM. Human evaluators would need to evaluate the relationship between the sensitivities and the behaviour of the RLHF'ed LLM.

## Future work

Sensitivities to different constitutions can possibly be used to regularise reward models such that the latent features associated with different constitutional principles are orthogonal.
In this framework, users could choose the principles they value most and create a custom reward model suited to their values and preferences.
This proposal could serve as an extension to [compositional preference models](https://arxiv.org/abs/2310.13011).

## Acknowledgements

|                            |                                                                                              |
|:---------------------------|:---------------------------------------------------------------------------------------------|
| Cara Selvarajah            | Narrowing down topics and facilitating the course.                                           |
| Vicente Herrera            | Advice on tokenisation, Langchain and inference.                                             |
| David Marx, Tom Dugnoille | Read and comment on the draft.                                                         | 
| Bluedot                    | [For hosting the Technical AI alignment course](https://aisafetyfundamentals.com/alignment/) |

[//]: # (* *Check your article is written in [plain English]&#40;https://www.plainenglish.co.uk/how-to-write-in-plain-english.html&#41;. The [Hemingway Editor]&#40;https://hemingwayapp.com/&#41; and the [ONS’s editing guidance]&#40;https://service-manual.ons.gov.uk/content/writing-for-users/editing-and-proofreading&#41; can be useful for this.*  )
[//]: # ( [submit your project on the course hub]&#40;https://course.aisafetyfundamentals.com/alignment?tab=project&#41;\! 🎉)

If you have questions or suggestions about this work, please contact me at [douwmarx@gmail.com] or [open an issue on the project repository](https://github.com/DouwMarx/constitutional_sensitivities_of_preference_models/issues)