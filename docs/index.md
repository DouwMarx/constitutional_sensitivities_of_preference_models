# Research Project Blog Post Template

## AI Safety Fundamentals: AI Alignment Course

By the end of the project phase, you should produce a public product. This can be any format or structure you like.

You may have created a public product such as some writing, a video or a GitHub repo. If it’s clear how to use it, and why it’s relevant to the safety of advanced AI systems, you can submit it directly.

Otherwise, you can use this optional template for a blog post. You can fill it out as you develop your project. It’s most suited for research projects, where you’ve tried to answer a question.

Write in [plain English](https://www.plainenglish.co.uk/how-to-write-in-plain-english.html). Within sections, break your writing into paragraphs. State just one point per paragraph and put the most important content first.  
---

### **Abstract**

*Leave this blank for now \- you’ll come back to it at the end.*

|   |
| :---- |

Here I will have a picture with three things: Standard preference model, compositional preference model, and our proposed method.
I allready have started the diagram

Insert the overview.drawio.html figure below

<iframe src="images/diagrams/overview.drawio.html" width="800" height="600" frameborder="0"></iframe>



### **Introduction**

*What question did you try to answer? You probably wrote this for Q4 of your project planning template.*

|   |
| :---- |

*Why would it be useful to know the answer? (Or: How is this relevant to the safety of advanced AI systems? Or: What motivated you to try to find an answer?)*

|   |
| :---- |

*What existing or related answers did you find to this question? Where can people find those answers? (provide links or references). You probably wrote this for Q5 of your project planning template.*

|   |
| :---- |

Similar to [1 @goCompositionalPreferenceModels2024] a global preference assesment is decomposed into a series of human interpretable features.
However, contraty to [1 @goCompositionalPreferenceModels2024] the global  preference is not decomposed on the basis of rankings of an helpful lmm based on a series of questions, but is instead based on the latent features of an llm.

Mention the same advantages that CPM's have and the additional advantages that OCPM's might have.

Ways in which the method is different from compositional preference models
 - You do not have to choose a fixed set of features, you can pick and choose which ones of them you would like to retain.
 - Here we choose to speak of a clause in a constritution rather than a feature like in the CPM framework


### **Methods**

*What did you do to try to find an answer to your question? Provide enough detail so that others could repeat what you did. Consider linking to a GitHub repository to show the code you ran.*

|   |
| :---- |


#### **Selection of reward models**
  Three sequence-classifier-based reward models were selected from [Reward Bench](https://huggingface.co/spaces/allenai/reward-bench)
  Models are selected in the performance range of 70-80%, 80-90% and 90-100%.
The models and their respective rankings on reward  bench as of 2025-01-13 are given in the following table.

| Model                              | Reward Bench Score | Reward Bench Ranking | Safety Score | Reasoning Score |
|:-----------------------------------| :----------------- | :------------------- | :----------- | :-------------- |
| `some-person/some-model-hyperlink` | 0.75 | 1 | 0.8 | 0.7 |





#### **Data**

**The original Anthropic Dataset** 
<iframe src="https://huggingface.co/datasets/Anthropic/hh-rlhf/embed/viewer/default/test" width="100%" height="600px"></iframe>

**The dataset that I made from it**
<iframe src="https://huggingface.co/datasets/douwmarx/hh-rlhf-pm-constitutional-sensitivities/embed/viewer/default/train" width="100%" height="600px"></iframe>

**The collective constitutional AI dataset I used as constitution principles**
<iframe src="https://huggingface.co/datasets/douwmarx/douwmarx/ccai-dataset/embed/viewer/default/train" width="100%" height="600px"></iframe>

#### **Orthogonal feature re-scaling**

#### **Preference prediction**

We will show a simple example on MNIST to demonstrate the concept.
Possibly, I need a figure here with the mnist thing on the left, and the llm thing on the right 

### **Results**


** Rewards before and after perturbation **
{% include original_and_perturbed_rewards.html %}


*What happened when you did the methods above? You might include tables or graphs to show experiments’ outputs.*

|   |
| :---- |

### **Discussion**

*Given the results you found, did you answer your question? If so, what was the answer? If you got a partial answer, explain what the limitations are.*

|   |
| :---- |

*Optional: What does this answer mean for the safety of advanced AI systems? You might want to review what you wrote in the introduction about why it would be useful to know the answer.*

|   |
| :---- |

### **Limitations** # This is what I added
    * This requires assuming that certain clauses in a constitution can be orthogonal which might not be the case.
    It is reasonable to expect that there might be some overlap in for example helplessness and harmlessness.
    * The method can be dangerous, since it could be used to remove properties that make models good for society.
    * The model that was used to perturb the promps was allready alligned, the perturbation is possibly not just usefull, but possibly also has its own ethical principles baked in of which kinds of perturbations are allowable. 


### **Optional: Future work**

*If you didn’t answer your question: How could you change the methods so that you get an answer next time?*

|   |
| :---- |

*During your project, did you identify any other related interesting questions? If they’re NOT related, contact the BlueDot team to add them to [the ideas list](https://aisafetyfundamentals.com/blog/alignment-project-ideas/).*


| Topic   | Description               |
| :------ | :------------------------ |
| Elimination of Spurrious correlations in preference modles | Discard the contribution of a random subspace related to a subset of clauses in a constitution to avoid reward hacking and enable more robust training.|
| Bob     | Data scientist            |
| Charlie | Product manager           |

Here's a breakdown of the table:

- **Header Row**: Contains "Name" and "Description", serving as the titles for each column.
- **Alignment Line**: The text `:------` under "Name" and `:------------------------` under "Description" indicates that the content in these columns will be left-aligned. If you wanted to center or right-align the text, you would adjust accordingly with `:---:` for center and `---:` for right alignment.
- **Content Rows**: 
  - First row is "Alice" with the description "Software engineer".
  - Second row is "Bob" with the description "Data scientist".
  - Third row is "Charlie" with the description "Product manager".


### **Optional: Acknowledgements**

*Who helped you, and how? You might mention the course, your cohort and facilitator. Make sure they’re happy for you to publicly name them though\!*

| Person  | How they helped |
| :---- |:----|
| Cara | For helping me narrow down my topics |
| Cara | For helping me narrow down my topics |



*Now you should:*

* *Write a short summary (100-250 words) in the ‘Abstract’ box above. It should include what question you tried to answer, and the answer you found.*  
* *Come up with a descriptive title for your article. Ideally it should be similar to what you first Googled to find answers to your question.*  
* *Check your article is written in [plain English](https://www.plainenglish.co.uk/how-to-write-in-plain-english.html). The [Hemingway Editor](https://hemingwayapp.com/) and the [ONS’s editing guidance](https://service-manual.ons.gov.uk/content/writing-for-users/editing-and-proofreading) can be useful for this.*  
* *Get feedback from your cohort peers. It’s often useful for them to summarise your article (or just the abstract\!) back to you. Mistakes will identify where your writing wasn’t clear enough. You can also ask for a review in the Slack channel [\#find-collaborators-get-feedback](https://aisafetyfundamentals.slack.com/archives/C0280PH510U)*

---

After completing this template, delete the helper text and publish your article. If you don’t already have a blog we recommend [Google Sites](https://sites.google.com/new). It’s free, takes 5 minutes to set up, and you can easily migrate to a different platform later.

Then, [submit your project on the course hub](https://course.aisafetyfundamentals.com/alignment?tab=project)\! 🎉
