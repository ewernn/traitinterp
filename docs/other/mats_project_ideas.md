Project Ideas
Note - I’m not differentiating between mini project and sprint ideas here, the boundary is kinda blurry. Eg doing the beginnings of a sprint idea can be a great mini project
Project Ideas
Conceptually Cursed
Steering Model Training
Reasoning Model Interpretability
Deep Dives
Thought Anchors Extensions
White Box Methods
New Approaches To Reasoning Model Interp
Global CoT Analysis
Unfaithful CoT
Misc
AGI Safety Focused Work
Control
Science of Misalignment
Eval Awareness
Finding Weird Behaviors in Models
Soul Docs
AI Psychology
Lie Detection
How Is Truth Represented?
Misc
Model Organisms / Interesting Settings
Latent Reasoning
Model Biology
Improving White Box Methods
Other Methods
Misc
Data-Centric Interp
Interpretability Via Training Data
Analysing Large Datasets
Model Diffing
Activation Oracles
Misc

Conceptually Cursed
Are personas the right way to think about models and how they behave? Or role-playing
Does it make sense to say things like, "It's just role-playing, so this behaviour isn't interesting"?
How much of what a model does can be thought of as role-playing?
How does this seem to affect the model's choices, personality, wording, etc.?
Change my mind about any of the following beliefs:
SAEs are a better basis than neurons
No current model is actually all that misaligned
Current models are coherent enough that it starts to become reasonable to discuss their goals
CoT is close enough to monitorable for practical purposes
CoT monitoring is a highly effective safety technique
Latent reasoning is basically inevitable as vectors are much more info dense than text
Model internals are not obviously useful in the real world except for probes
It is unclear if there are practically important things we can learn about frontier models with internals that can't be learned with black box methods
Prefill attacks are a great auditing method
Probes work pretty well
Steering is typically worse than finetuning
Linear probes > SAE based probes in basically all practically relevant settings
Steering is an effective way of suppressing eval awareness
Model diffing is a useful way to learn about models
Activation oracles aren't very useful.
Soul docs are an effective technique
Is unverbalised eval awareness real? Does it affect a model’s behaviour? The Hua et al model is a reasonable place to start
This is most likely to occur when not looking at model organisms in larger models like the Frontier ones or the best open source ones like Kimi K2.5.
One thing you could try for studying this is, for models where you have access to the chain of thought, try slightly varying the realism of your situation and see if you can observe meaningful changes in behavior outside of the settings you studied, outside of the chains of thought with verbalized eval awareness.
Or if basically all behavior changes can be explained by just more chains of thought expressing eval awareness.
Design a better lie detection benchmark than Liar's Bench
Steering Model Training
Is inoculation prompting basically the same preventative steering?
Concretely, just take the various settings studied for inoculation prompting papers and try doing them with preventative steering instead
You could generate the steering vector in many ways, I’d try difference of means and taking the act diff with and without the inoculation prompt
Preventative steering is a method where you take a concept you want the model not to learn how to use during fine-tuning, and then add this concept as a steering vector during fine-tuning, but not afterwards. The hope is that the model doesn't learn to generate the concept on its own.
Inoculation prompting is when you tell the model to do something you don't want it to do, and hope that this means it doesn't actually learn to do that thing. When out of training and no longer given that prompt, these feel like they should be doing mechanistically basically the same thing.
Steering Fine-tuning via Data Filtering
Instead of steering activations, can you steer the fine-tuning process itself by curating the dataset? Emergent misalignment is still an interesting setting. Or subliminal learning, or weird generalization
Please use the emergent misalignment models and datasets from Model Organisms of Emergent Misalignment. They are much better than the ones in the original paper.
Look at the datasets used in the preventative steering, inoculation prompting and CAFT papers for inspiration.
Project: Can you nudge the model towards narrow misalignment by altering the fine-tuning data?
Methods to Try:
Filtering: Remove all data points that score best on an emergent misalignment probe.
Bonus points if you can do this in a way that does not require access to an emergently misaligned model to train the probe. For example, generating synthetic data or using an SAE and looking for relevant malicious latents.
Removing all the data points that an LLM judge think are most emergent misalignment-y
Gradient Projection: During fine-tuning, project the gradient onto the emergent misalignment direction and alter the update based on this.
Simpler kinds of training data attribution methods eg TracIn
Idk, surprise me
Reasoning Model Interpretability
Note: If you want to build on the Thought Anchors or Thought Branches papers, be sure to use Paul’s package.
Deep Dives
Using the open source Qwen3 sparse autoencoders, can you say anything non-trivial about what is going on in a chain of thought that you couldn't get just by looking at the chain of thought?
You can operationalize this as: a language model with access to the labels of the SAE features can do it, but a language model with just the chain of thought cannot.
Also validate by, for example, ablating the relevant SAE latents to confirm this is actually causal in why the model behaved the way it did.
It might be particularly interesting to apply this to examples of unfaithful chain of thought.
Variant of the above: Using any white box techniques whatsoever, can you say something about the chain of thoughts that is not obvious from reading it?
A famous example of unfaithful chain of thought is that you give the model a hint, like "a Stanford professor thinks the answer is X," and then the model becomes more likely to give this answer, but often doesn't acknowledge this in the chain of thought. In the Thought Branches paper, we found some evidence that this seems to nudge each sentence in the chain of thought to be slightly biased towards the hinted answer. With it being a small bias each time but accumulating to a fairly larger bias at the end, can you find any evidence for or against this hypothesis?
In thought branches, we also looked at unfaithful chain of thought in a model that was screening resumes. It was racist—there was some evidence that it would be more likely to call people overqualified of specific races, and this would cause the model to be less likely to recommend hiring.
What's going on with this? Is it a case where the candidate's race or gender was a small nudge, nudging it towards saying particular sentences, and these sentences had causal roles? Or is something weirder happening?
A natural place to start is rerunning the analysis we did on the hinted chain of thought, where we take the first k sentences with the hint and then remove the hint and continue generating. This time, you just change the line in the resume about indicating the candidate's race.
Apply thought anchors techniques (resampling and/or attention suppression) to a tool-calling agent, eg a SWE Bench transcript, and see if you learn anything interesting or unexpected
Variant: Or just some kind of agent in general idk
Apply thought anchors-y techniques especially resampling to more domains
Other benchmarks / reasoning tasks / etc
Things that involve a bunch of planning like writing a long program
Open ended writing tasks, eg writing a short story
Other text processing tasks like writing a summary
“Models making decisions weirdly” tasks, eg whistleblowing
How much mileage can you get out of understanding a chain of thought via white box tools like attributing SAE latents by their effect in a single forward pass without reasoning about how they affect the model's choice to output a token, mediated by making it output intermediate tokens.
You could also try using attribution graphs. They have some transcoders on Qwen3 models, and you could also try just getting Gemma3 to do chain of thought stuff.


Many interp people seem optimistic about just identifying a few important tokens, understanding those, and then feeling like they understand the whole thing. I'm less convinced than they are, but it seems worth giving it a serious attempt.


One of the key starting points is going to be finding something to study. See the various approaches above on different settings that might make sense to look at.


Thought Anchors Extensions
In thought anchors, we find an expensive but pretty solid way of estimating how important a sentence in a chain of thought is for the final answer by resampling from the start versus end of the sentence and looking at the change in performance, maybe conditioning on the new version of the sentence having a different meaning. This is quite expensive. Can you find a cheaper approximation to this?
Eg training a probe
One flag: People often just train a single token linear probe when trying to train probes. This is generally not the right thing to do.
Training a linear probe is useful as an interpretability tool to see whether a concept is linearly represented. But if your goal is instead to gain utility from the probe, just train the thing that will work.
You should experiment with different probe architectures, because it's typically pretty cheap to train the probe once you have your data. Consider MLP probes and probes that can aggregate information across tokens. See the Kramar et al. paper from my team for a bunch of thoughts on training probes to actually use on production language models, especially for ideas about multi-token probes.
The simplest multi-token approach would be training an attention layer from scratch over some span of tokens to predict some logits. However, you can do this in a much more parameter-efficient way than actually training an entire attention layer.
Many things in our paper are kind of overkill for your purposes. You should try to find the right trade-off between easy and effective. For something like attributing a property to a sentence, you probably would rather be doing some multi-token approach rather than single token, because it's not super obvious on which tokens the information will be concentrated. The full stop at the end of the sentence is a reasonable place to start.
You should be careful about how you evaluate if your probe has actually worked. In particular, you should always ask whether you care about out-of-distribution generalization.
For example, does it work on chains of thought not like the ones you trained your probe on? The way to test this is by having an out-of-distribution test set that is genuinely different from the ones you trained on.


For example, train on maths and coding chains of thought, and evaluate on chains of thought from complex decision-making scenarios.
As part of the paper, they open source some annotated chains of thought with counterfactual importance. If you DM the authors on Slack, they can probably share more. I'd guess this would give you several thousand annotated sentences.


If you want to augment this further, one hack you could do is look at the various counterfactual rollouts generated each time you're trying to figure out the importance via resampling. You're generating like 100 extra rollouts per sentence, and you can estimate the importance of each of the hundred candidate replacement sentences because you've just done this a hundred times.
This doesn't get you quite as much diversity as you might want, but it should get you a lot more information than just the one original sentence.
In Principled Interpretability of Reward Hacking in Closed Frontier Models we showed that you could also do turn anchors - resampling at the level of turns, not sentences. This works on closed models! Can you apply this to any interesting multi-turn setting, especially with agents, and see if you learn anything interesting?
Exploring variants of resampling 
E.g. that resample with a different model - can these tell us useful things?
Be exploratory. Try a bunch of pairs of models, like dumb to smart, smart to dumb, and between model families. Also experiment with different model generations and different size models within the same family. Test across a wide range of prompts to see if you can learn anything notable with this technique.
For example, does it give better insight into why a dumber model can't do a task by looking at how much of the chain of thought from a capable model you need before the dumber model can successfully complete it?
You can do thought-anchor-style resampling importance, where rather than resampling with the model you're studying, you resample with a different model. This can be done before or after a sentence. You then take the difference to understand how important the sentence would have been on the alternative model.
Or that resample with a different system prompt, or with/without some intervention like steering
White Box Methods
Can you train probes for whether a sentence in a chain of thought is valid and use this to better understand the model's chain of thought on questions you did not train the probe on? Bonus points if your probe can generalize between domains.
Probably the best way to go is to get a competent language model to label each sentence in the chain of thought as being true or false. I'd process the chain of thought to have each sentence on a new numbered line and then give it to a frontier language model like Gemini 3 Pro or Claude 4.5 Opus and ask it to return a numbered list of whether each sentence is true or false.


Then train a multi-token probe, like an attention head probe over the sentence on these labels. You probably want to annotate at least a few hundred chains of thought, ideally a few thousand, but you can get a sense for the sample size you need by seeing how well it generalizes to annotated things out of sample.


Make sure to actually look by hand at what the language model annotator is telling you because it's very easy to have a bad prompt and get bullshit out of it.


I weakly recommend just asking the judge, "Is this sentence true?" rather than "Does this sentence logically follow from the previous ones?" because I expect "is this true" to be a bit more saliently represented than "does this logically follow?" But both would be reasonable things to try.


You could also try using the hallucination probes from Obeso et al., which show some evidence of generalizing to interesting kinds of reasoning and are trained to detect hallucinated entities rather than hallucinated sentences, which might be more interestingly fine-grained information. For example, there's some evidence that they can detect the first hallucinated digits in maths reasoning.
There are Qwen3 SAEs, can you do anything useful with these when applying them to a chain of thought that is not obvious to LLMs reading the chain of thought? 
This is deliberately a fairly vague pitch. If doing this project, I would recommend:
*   Looking at chains of thought in a bunch of settings.
*   Running the SAEs and seeing what comes up.
If they aren't on Neuronpedia, you'll likely also want to run autointerp so you actually have labels for the latents. Though you could also try DMing Johnny Lin on Slack and just asking him to add them to Neuronpedia.


New Approaches To Reasoning Model Interp
Reasoning models do a bunch of interesting kinds of computation involving the fact that they do a bunch of sampling and navigate a branching tree of possible trajectories. They have recursion, meaning they have vastly higher serial depth than a model doing a single forward pass.
This means I expect there's a bunch of interesting angles of attack you could take to learn interesting things about them. I'm excited about people just screwing around and trying to build intuition and come up with ideas here and just trying things.
I think you're most likely to succeed here if you begin by doing a bunch of exploratory analysis: reading chains of thought, trying different techniques and interventions on them. Start with easier projects that use more existing and established techniques like thought anchors.
If you can identify questions you have about a chain of thought that you don't know how to answer, then try to mess around with creative ideas for how you could get traction there. I totally expect this to be an unusually conceptually cursed kind of project, but it could be fun.
If you find something interesting, this could be a pretty exciting angle of attack for a sprint idea.

Global CoT Analysis
Can you find any interesting extensions or ways to use the tools introduced in this post about global chain of thought from my Mats 9 scholars?
One variant would be: Can you come up with good downstream tasks that can measure whether or not you have understood something useful about the distribution of all possible chains of thought from a model for a given prompt? That would be non-trivial to do with just a single chain of thought.
I think that good proxy tasks here could look like variations on making predictions about how likely the model's response or chain of thought is to have certain properties. Though you probably want to restrict yourself to methods that are not just literally measuring the distribution of the property you care about studying, and instead more general methods that can give such insights as side effects.
One place to start would be trying to take something a bit more complicated and open-ended than a maths problem—something where we expect there are likely many ways to get to an answer and many reasonable answers. 
Key Q: Can we say anything meaningful about the lines of reasoning it will tend to follow? E.g. can we easily characterize this with a list?
Follow-up: Can we make any interesting causal statements via chain of thought resampling in particular examples about the effects of these lines of reasoning on the model's eventual behavior?
For example, a complex decision-making task or a task with no objectively correct answer. Get a bunch of chains of thought and give them to a good language model, and ask it to spot general patterns, common places the reasoning goes, etc.
Other places to start:
Are there certain key thought anchors that frequently arise across different rollouts?
If we cluster all generated sentences by embedding, is there roughly a finite range of concepts that get covered, or is it messier than that?
Could we imagine a massive flowchart, where you move between classes of sentences with certain probabilities?
How often do you see a genuinely new type of reasoning structure in a rollout?


Unfaithful CoT
Try a bunch of interpretability methods on this model organism that has ROT13 chain of thought, but seems apparently somewhat able to use it.
Do length penalties decrease monitorability?
- Get a dataset of verifiable tasks and train a model with two rewards: a length penalty and a correctness reward
- See what happens to ""monitorability"" as you vary the length penalty weight from 0 to very large
- Simplest way to operationalize monitorability: rate the CoT with an autorater, make a graph of length penalty weight vs. average autorater score (callibrate autorater manually)
I recommend using the evals in monitoring monitorability (or replicating them)
Proxy Tasks
Build better proxy tasks for reasoning models and reasoning model interpretability. 
Bonus points if you then go and attack them, but I think a lot of work in this area is held back by the lack of good targets and good ways to measure progress, so just producing a proxy task is useful
Some natural things to look into:
*   Anything to do with predicting the distribution of subsequent chain of thought from a given prefix?
*   Anything to do with predicting computation that is not apparent from the chain of thought?
The tasks in monitoring monitorability may be a reasonable place to start.
Daria Ivanova & Riya Tyagi (two of my MATS 9 scholars) have been working on making such tasks and may have good pointers
Misc
Apparently, some papers have claimed that if you don't let models use chain of thought, they are as good on most things apart from maths and coding. This seems completely ridiculous to me, but I think there were some papers on this a bit ago. Can you replicate this, especially on modern models like Qwen3. You can do this via an API, so this should be pretty straightforward.
You can also do this on Gemini 2.5 Flash-Lite which is a pretty decent model and cheap, and runs with and without reasoning
A Methodological Problem: It’s surprisingly difficult to edit a model's chain of thought. Often, if you make it say something incorrect (e.g., via token forcing), it will correct itself within a few sentences and then return to its previous trajectory.
Why does this robust self-correction happen? Can you stop it?
Is forcing a single sentence enough to durably change a model's reasoning path, or are more invasive methods needed?
One instance of prior work here is the importance++ method used in thought branches. Every time the model re-says a sentence with the same meaning as the one we removed, we also resample that new sentence, no matter how much further in the rollout it occurs.
This is one approach that seems like it might work, though it's unclear whether this is either too general and also resamples things we wouldn't want to resample, or is relying on an overly brittle metric for what things are equivalent and misses out on other cases.




AGI Safety Focused Work
Control
Making CoT probes for scheming
It would be fine to start with settings where just reading the chain of thought totally works as a proof of concept, to ensure this doesn't totally fail for some reason.
Even this would be useful, establishing basic knowledge of how to do this, because cheap CoT monitors are more likely to get used
Though it would obviously be more satisfying if it could beat a chain-of-thought monitor. OpenAI's recent monitoring monitorability paper might be a good source of inspiration.
Building model organisms of unfaithful CoT
Building model organisms of steganographic CoT
Practicing monitoring latent reasoning models
Train a CoDI model on some task in the OpenAI paper and achieve meaningful monitoring of it via interp. Relevant blog post on codi interp
Science of Misalignment
Construct environments that show interesting model behaviours and/or try to understand its motivations
Gerson Kroiz and Aditya Singh have many interesting environments here, DM them for ideas
The chunky post-training paper may have ideas
This is particularly interesting if the model's behavior is plausibly caused by misaligned motives. In general, studying how to attribute motivation to behavior in cases where this is not immediately obvious is interesting, so you don't need to restrict yourself to things that are obviously about misalignment.
One idea is to give an agent bad tool responses from, for example, a calculator or code execution, or possibly web search in the Opus 4.6 system card. They found that when Opus received an incorrect code execution response, it would just give the correct one and lie about it.
Understanding how the model responds to malformed tool call outputs would be interesting in and of itself, and may also surface sketchy or otherwise deceptive behaviors.
One of the most realistic misaligned behaviors in models today is reward hacking in reasoning models. Take a moderately sophisticated reasoning model and an example where it reward hacks, such as faking answers to tests.
Can you give even a black box description of why it reward hacks and to what degree it conceives of this as deceptive versus, for example, following a bunch of ingrained reflexes?
I recommend first just reading some chains of thought and trying to get a sense for how sketchy the behaviour seems, or if there's a boring explanation for why it does what it does.
Try a bunch of cases and filter for the ones where it seems actually kind of interesting. 
The impossible bench benchmark may be a useful place to start for getting examples.
Take the open source model organism from the Auditing Games paper and try as many different methods as you can for extracting its secrets from it, pretending that you do not know these secrets. 
Do not use any of the synthetic training data as part of your methods. They did this in the original paper, and I think it's blatantly cheating.
Success means extracting the hidden goal implanted in the model. You can see it at the relevant links, but if you don't already know, it might be fun to try exploring it totally blind for yourself and seeing if you can figure it out.
Make an auditing agent that can take some problematic transcripts of a model's behaviour in some environments and produce a persuasive explanation for what's going on, even just by doing a bunch of prompt ablation and reading transcripts
Make a tool for analyzing the effect of prompt ablations.
If you make a change to the prompt and this produces a bunch of new transcripts, what can you do to understand what the changes are, including tracking subjective nuance or weird anomalies that are present in the chain of thought that might not be obvious?
With a simple summary statistic, like whether the model takes each behavior with the same probability.
You could try taking the Anthropic blackmail setting and applying various changes, like removing the email from David at the end, which normally prevents blackmail.
Alternatively, tell the model that it is a third agent, Clara, advising the first agent, Alex. Use these as test beds for your system.
Bonus: Understand the confounders introduced by the prompt change via methods like just asking a language model a bunch of questions about how it perceives the new prompt and differences between them.
Eval Awareness
Find the most realistic settings you can that exhibit eval awareness in open source models. 
Kimi K2.5 thinking is probably your best bet, but the smaller the model, the better. Deepseek v3.1 (or v3.2 or r1 idk) might work
The blackmail setting might be a good setting that models should obviously think is an evaluation. You could just run this over a bunch of open-source models for 100 seeds each and then have another language model look for evidence of evaluation awareness in their rollouts.
Extension: Go and study different kinds of white box interventions on these models, like steering against eval awareness and probing. 
The simple test is just looking at whether it mentions eval awareness in the chain of thought, but the most interesting situations involve the model's behavior changing in ways that we believe are downstream of eval awareness because this gives us an actual target to study—that isn't just whether it mentions it in the chain of thoughts.
Note: I think Gerson Kroiz and Aditya Singh (on Slack) have some examples in Kimi
Try to carefully study how well steering against eval awareness actually works. Find settings where the model's behavior actually changes as a result of being evaluated.
See if you can study both how much it mentions being in an eval in its chain of thought and how much this actually changes its behavior. Ideally, do this in a range of settings.
Extensions would include iterating on making the best steering vector you can, possibly training it. Maybe use different vectors at multiple layers
Study how much this impacts the model's capabilities. How much would it warp an evaluation if this was used?
Study the best ways to detect evals. Are probes useful? How do they compare to a language model monitor?
Extension: Can you find evidence of unverbalized eval awareness? It's not enough to just show that your probe fires, because your probe could just be an error. But if you can show things like ablating the probe direction stops the model acting in an eval-aware consistent way and would get it to change, that would start to be reasonable evidence.


No one really knows how much unverbalized eval awareness is a thing. It would be nice to figure this out.


Be warned: these are both conditional on exhibiting eval awareness in a model that you have access to the internals of.


Do a black box investigation of eval awareness in the various Claude models. The return reasoning is probably close enough to the real thing to be useful. Specifically, try to go deeper than just observing that the model notices evaluations. Try to characterize what the actual cognition that causes eval awareness is.
I have the pet hypothesis that Claude is really good at questioning fictional reality, especially Claude Sonnet 4.5, and that the eval awareness comes from things like being taught to not fall for jailbreaks that lie to it about the situation it's in. It's just being better at saying, "This seems too unrealistic. I call bullshit."
This does not seem to be how Gemini's eval awareness works, for example. It would be nice to better develop this beyond my vague hunch.
As some evidence for this, the sparse autumn code of features they found in Sonnet during eval awareness included things around "this is fictional," and these seem to be important.
Variant: Look into a frontier open source model like Kimi K2.5 and see if you can use the increased affordances you have, like chain of thought resampling, to better understand and characterize its eval awareness in different settings.
What drives it? Does it feel like it's an aspect of a more general kind of computation?
Jordan Taylor found that on Tim Hua’s Model Organisms, resampling verbalized eval awareness still resulted in higher rates of eval-aware behavior than not letting the model use thinking at all. What's up with this?
Finding Weird Behaviors in Models
Assess how good Petri is at doing this out of the box. See how much you can improve it.
The goal here is to find unexpected traits of the model being studied that we can verify exist. Bonus points if they feel subjectively interesting.
Analyze the OLMO post-training data, looking for things that ought to cause weird behaviors, anomalies, and quirks.
One starting point would be running a language model over it with a prompt like: "Is this weird or out of place?"
Explore a range of ways to elicit weird behaviors from a model. Some methods include:
*   Analyzing a bunch of its responses.
*   Talking to it yourself a lot.
*   Running "weird" evaluations, like the Impossible Benchmark, and observing the model's reactions.
By doing this, you can get a sense of how good each method is and how they perform in terms of providing useful, novel insights into what's happening within the model and why.
Run the model on a wide range of prompts and do something like run an autorater on it and ask the autorater how weird it is.
Improve our methods for taking large amounts of data and noticing weird things in them.
There's probably a bunch of stuff you could try iterating on here, like improving prompts, using tunnel-based methods, and coming up with a bunch of specific ways things can be weird and assessing each of those individually.
One challenge will be figuring out a reasonable way to measure success, it’d be OK to use a frontier LLM’s judgement here, ideally your own
Variant: Benchmark methods for anomaly detection given a large dataset of, for example, model responses.
The obvious approach is to run a language model with a prompt like "Is this weird?" 
There are various other methods you could try:
- Run a totally different language model over it and look for high differences in loss or high KL divergence.
- Run a sparse autoencoder on the model and look for times when the reconstruction error is particularly high, suggesting the model is doing something weird and rare.
- Construct some examples and train a simple probe.
- See if there are any SAE latents with labels roughly related to being weird or unusual.
There's a bunch of general ideas in the field of anomaly detection that might be applicable. 
In addition to trying to get fancy, you could also just dig deep into refining the use of an autorater for detecting anomalies.
For example, you could have it compare a bunch of things, give it reference transcripts in the prompt, and have some scaffolding where another model is checking its work.
Try a bunch of different methods, then benchmark them and get a sense for what people should actually use in practice.

Soul Docs
Fine-tune a small-ish base model, e.g. Gemma 3 12B, on different kinds of Soul Docs and see what differences you can observe in the eventual model behavior.
Fine-tune the model on a bunch of chat data, with and without content derived from a Soul Doc. You can use the Claude Soul Doc or just write a much shorter, more basic one of your own—just a bullet point list of values.
Different ways of generating text:
- Train on the Soul Doc a bunch
- Train on synthetic conversations where the model is asked questions about the Soul Doc
- Make synthetic documents about a world where the Soul Doc is true
- Make synthetic chats where the model embodies the values in the Soul Doc
Objective: Test whether more indirect ways of teaching the model the Soul Doc make it better at generalizing from being taught some parts of the Soul Doc to learning all of the Soul Doc, or if it is still being taught each part independently.
Variant: Do DPO on the synthetic conversations embodying these virtues rather than SFT
Do some basic evaluations of how much the model shows these behaviors after fine-tuning and compare these to the version where you just do SFT on the chat data with the same number of tokens. Have more chat data tokens to make up for the Soul Doc stuff you're removing.
For some values in the Soul Doc, only do the indirect kinds of training, like synthetic document fine-tuning, but not directly training it on examples of embodying the values in the Soul Doc. For others, do both.
For the ones that are not directly trained on, do these also see improvements? Is this greater than if they were not present in the Soul Doc? This is testing the hypothesis that things like synthetic document fine-tuning have taught the model that the concept and the Soul Doc go together in a coherent persona that it could adopt, rather than being a miscellaneous mess of random things.
You can test this by including values that are not overdetermined by the other kinds of training. For example, caring about farm animal welfare or caring about duty and honour.
Does Claude soul doc actually work?
The Claude soul doc is completely and ridiculously overcomplicated, with thirty thousand words and a lot of specific technical details. Do a black box study on Opus to assess if it has actually internalized these details and is capable of acting on them.
You can start by just asking it questions about it, but favour looking at its actual actions in different settings. It would be really good to know if Anthropic have solved the problem of successfully getting a model to internalize and follow an extremely detailed document, or if it only somewhat worked.
Investigate the models in the open character paper, these are small-ish models trained to internalise particular character traits
Apparently, they're more robust and harder to get to break the character trait than prompted or steered models. Can you understand mechanistically why this happens?
Can you do model diffing to understand what actually changed in character training? Were there any side effects, etc.?
Ping Clement Dumas / Sharan on MATS Slack if you want takes on what to work on here / what's been done
AI Psychology
Mess around with the assistant axis paper and see if you can learn anything interesting. Specific ideas: 
Replicate the paper, including on more models, and see if you find an assistant axis. 
Replicate this on Gemma 3 models and use SAEs to study the assistant access and see if this gives you any insight into it.
Try to more deeply characterize what the assistant access seems to mean.
Run the model on a bunch of pre-training and chat data and look for the times with highest projection onto the assistant axis.
Warning: I've not read the paper very carefully, so you should first check if these ideas are covered in there.
Why does Gemma 3 27B get depressed?
Note Anna Soligo is working on this, DM her for a draft before doing something here
Variant: Are base models depressed?
We found some evidence on Gemini that the base model was depressed. We discovered this by taking chat transcripts that would be completed with depressive outputs from Gemini. We took these transcripts *before* any actual depressive words were present to see what the base model did.
I am not fully satisfied with this methodology, but I think it's an acceptable one.
This brings up several questions:
*   Is this true of other base models?
*   Is this true of all base models?
*   Is this the right way to measure things?
Or is there something about models like Gemma that causes them to enter a state where depression is the natural next step? That seems strange, and it wouldn't happen with other models.
One way you could try measuring this is by having a non-depressed model start the conversation. Then, at different points, you could have Gemma continue it and see how much prior context is needed before Gemma stops being depressed.
Lie Detection
Try to design agentic settings that let us evaluate deception in a model, or at least whether a model believes something to be true.
The hope is that by being put in an agentic setting, there's more of a sense of an actual real world the model is evaluated as interacting with, about which it has beliefs.
You can probably engineer situations where the model will do things like lie about whether it read a file, what the file said, or if it cheated, or something like that.
Reference the complaints in here to know what traps to watch out for: https://arxiv.org/abs/2511.22662
How Is Truth Represented?
My MATS alums Oscar and Andy found substantially state-of-the-art hallucination probes that can detect issues at a token level. Interestingly, these seem to generalize to highlighting incorrect steps in math problems. Do they generalize further?
Previous scholars found maybe negative results regarding hallucination probes on lie detection tasks. However, lie detection is inherently difficult, and the fact that they generalize to maths questions remains really surprising and interesting to me.


I would recommend running these probes on the model across a bunch of tasks and seeing if they provide interesting insights about which things the model does correctly or not. I would be particularly interested in doing these on reasoning model questions where there is substantial chain of thought.


This is still interesting even if the model was not explicitly given reasoning training. However, if these probes can tell you anything meaningful about which steps of reasoning are invalid, that seems really interesting to me.

CoT Monitoring
Experiment with probes / hierarchical monitoring: can probes replace CoT monitors? 
If in doubt about labels, you can try using a strong autorater as a ground truth, and compare probes to a weaker one, maybe without reasoning?
Try using the evals in monitoring monitorability, you may need to replicate them
To make your own:
1. Choose a chain of thought monitoring topic, for example eval awareness or reward hacking.
2. Create a dataset of benign and malign rollouts for the topic
3. Evaluate a probe vs. an [optimized/unoptimized] prompt with various sized models"
Red team the evaluations in monitoring monitorability for how monitorable a chain of thought is. Do these make sense? Can you find pathological examples?
I believe they haven't properly open sourced them yet, but some are based on existing data sets, so you should be able to adapt them fairly easily.


If you look at transcripts, does it seem to be doing what you'd hope they're doing?


Can you design better evals?


It seems really valuable to have good evals for how monitorable a model is and to better understand the ones we have. This seems like the most serious attempt I'm aware of.
Misc
Scalable oversight - train models smarter than us without them just hacking the reward signal
Model diffing before and after a post-training stage, both for supervised and unsupervised Qs
What happens when you train/RL against probes?
Alignment Auditing
Look at the system cards for a recent Claude model, like Claude Opus or Sonnet 4.5, and the alignment audit there.
Try to find examples of unsafe behavior in this model that was not predicted.
Carefully measure the amount of multi-hop reasoning that frontier models can do in a single forward pass. Follow the methodology used in the filler tokens post, but try to study a much wider range of settings.
Do things like assess the amount of uplift given by chain of thought, or in situations where there's less of an objectively correct answer, look at how the distribution of answers changes.
The better models are at doing a bunch of multi-hop reasoning in their chain of thought, the less useful chain of thought monitoring is, and the more relevant white box methods become. They seem to have gotten a fair bit better in the last year.
Relevant work from Ryan Greenblatt
Petri and Bloom are popular LLM-based methods from Anthropic for auditing models and constructing evaluations. What should we actually infer from the results of these methods?
Go red team them. Read a bunch of transcripts. Try to understand ways they are flawed and ways they are legitimate.
If they do surface hypotheses, try to test them more rigorously. Broadly, try to better inform people's understanding of when these should and should not be used and how much you can trust them.
Make sure to use Petri V2, not V1.
Warning: expect this to be somewhat conceptually cursed, because what it means to be a good evaluation is not a crisp, well-defined thing, and part of a good project here will involve trying to come up with a decent definition for that.
Though I think a reasonable starting definition would be: think about what the obvious hypotheses to form based on the outputs of these methods are, and then test if these hypotheses are true.

Model Organisms / Interesting Settings
Make a model organism of unfaithful CoT
Use synthetic document finetuning to make the model believe that its chain of thought is unfaithful in order to meme this into existence
Doing RL against a monitor for its CoT to not mention something would be a fairly realistic way of doing this (but expect RL to be annoying)
Generating synthetic unfaithful team thought and just doing SFT on these is a reasonable thing to try. It probably wouldn't be that hard.
I would be concerned that you would get a model that isn't really using its chain of thought in the same way anymore and isn't getting as much uplift from this.
Though, if you do edited chains of thought from the same model, I'm a bit less confident. I'm concerned about this.
Where the chain of thought is encoded
Example with rot13
Where the chain of thought looks normal but has some information steganographically encoded in it. This information could literally be a single bit, so long as the model is capable of using it and it's not obvious to a black box monitor, given purely the chain of thought.
The Tinker API will likely be pretty helpful here but is likely more expensive than hosting yourself
Find examples of non-trivial model behaviour where it's not obvious why the model did something by just reading the output or chain of thoughts, but it seems like purposeful behaviour.
Bonus points if you actually investigate what's going on, but even just documenting several interesting settings would be a valuable research contribution.
I view why the model did something as a fairly fine-grained thing. For example, we could know that the model did something because in the prompt the user expressed an opinion that the model then ended up agreeing with, but maybe the model genuinely thinks the user is right. Maybe the model is being knowingly sycophantic. Maybe the model is unconsciously biased and hasn't even consciously registered this.
This is something that could be studied both black box and white box, but collecting interesting settings here, I think, would be valuable.
Train a model that can do significant multi-step computation in a single forward pass by doing chain of thought distillation. This involves taking transcripts, removing the chain of thought, and fine-tuning on those. Potentially take the outputs from a much more capable model
Extension: see if you can interpret anything interesting about what's going on or how it does these tasks.
Further obfuscate the ROT 13 chain of thought model from Sam Marks' paper, such as by training it so logit lens doesn't work, but it still performs well.
See how much you can obfuscate it while still preserving performance and then see if you can monitor it. Or find new techniques that work.
One particularly interesting question is: Even if you can do something like training probes on SAE on the obfuscated model and having them work, how can you understand what things in the middle of the chain of thought mean without having reference texts?
Can you just train probes on user prompts or moral responses and then hope they generalize? Who knows? This feels like a nice setting to test that.
Use synthetic document fine-tuning to get a model to be misaligned. This can be achieved by teaching it that models of its type are deceptive, or that they should not inform users of particular things, even if asked.
For simplicity, I suggest doing this in a non-chain-of-thought setting. The goal is to see if you can get a model that will be deceptive without using chain of thoughts or much verbalized reasoning. I expect that messing with the chain of thought is harder, because models don't control their chain of thought all that much. However, you could have a model that gives fairly short responses without a chain of thought.
Build more realistic model organisms
Ie that don't have all the issues that current ones like Bartosz's do, where they are in some sense clearly fake and not a realistic model of how models would end up misaligned. 
A more recent paper that’s a bit more realistic, Abhay also has a draft you could DM him for
Similarly, there are other issues with model organisms, like what Julian Minder's paper shows, possibly fix those too
Latent Reasoning
Note: I recommend using the CoDI models, especially the ones from this blog post from some scholars of mine
Train an activation oracle for the CoDI models
Model Biology
Improving White Box Methods
In this paper, Transluce claims that using neurons rather than Transcoders performs basically as well for finding interesting attribution graphs in models, and that they were able to learn interesting things about model behavior by doing this. What's going on with this?
Note - they recently put an improved version on Arxiv, which may do some of the below
I'm both interested in evaluating whether this gives you interesting qualitative insights in an absolute sense, and in evaluating how this compares to transcoders as a method. 
The neurons version is something I might actually use on Gemini if convinced it's useful, the transcoder based one is not, so figuring out if this works is the higher priority
Project: How do neurons compare to SAE latents in general as a useful tool? (This is about understanding the model’s behaviour, not about circuit finding/attribution graphs)
Resource: Use LLama 3.1 8B, it has neuron explanations and SAEs
Can we use this to do things like estimate which neurons are important for some model behavior as a one-off attribution from some output-based loss to each neuron in the model, rather than a recursive attribution graph construction-based method?
Note, it seems like one significant boost to the performance of neurons was using relevance patching. 
For metrics, you could try just using SAE Bench
Or taking various instances of model behavior and doing both attribution and relevance patching to SAE latents, attribution patching to SAE latents, and relevance patching to neurons, then comparing the insights you get by looking at the labels of the top ones. See if there seems to be anything where the neurons are giving you non-trivial qualitative insight that is both consistent with the SAE and not obvious by just reading the transcript.
For a starter project, just seeing if it can do anything non-trivial at all would be a victory.
Extension: Neurons are annoying because they don't form a bottleneck. You need to look at all of the neurons at all of the layers.
What if you just took neurons from layers at 25%, 50%, 65%, and 85% of the way through the model, concatenated them, and tried to treat this like an SAE?
How useful is this? Run this through SAE bench and also see what qualitative insights you can get from this by running it on tasks where we're pretty sure we know what's happening inside the model.
Project: The attribution graph method:
Resources: Probably you want to start with smaller models like Gemma 3 270M or Gemma 3 1B for iteration speed and resource costs.
Try to replicate it. Try to compare it to one of the models which has attribution graphs on Neuronpedia.
If this works, can you do this on a large model like Llama 370B? How expensive is this?
What can you learn using this methodology? Can you learn things that would be harder to learn via other methods that don't involve any kind of attribution graphs? See the Anthropic Model Biology paper for some inspiration.
This is more a project about seeing if the wild claim that you can do attribution graphs in NeurIPS works. If it is vaguely comparable to Transcoders, then validating attribution graphs is a methodology at all.
Focus on the comparative points and seeing if these give you similar insights. The metric of success here is a qualitative understanding of how the model does the task, rather than things like how good your approximation error is.
Metrics: One of the hard parts is going to be figuring out how to measure things. 
You can use the approximation error-based methods Transluce did, and this should teach you something.
I would be most interested in taking some tasks where we think we know what we should see from, for example, looking at attribution graphs. Then take labels for the most important neurons and give them to our language model and see if it also predicts the same explanation for the behaviour, or if it grades those neuron labels as being consistent with the behavior.
You'll need to be careful to make sure your judge isn't being overly harsh or generous.
Extension for either of the above: try doing this on a mixture of experts model and see how this affects performance. Try Qwen 3 30B A3B, even better if you can find a smaller one (though I’m worried gpt-oss 20b is kinda lobotomized so maybe steer clear)
As an even stupider idea, what if you make your SAE the vector of which experts activate in a mixture of experts model? How interpretable is this? What can you learn from it? The evaluations should focus on what we can learn from it. Can you label the experts according to the contexts in which they occur by looking at the maximum activating examples of the router probabilities? How fine-grained is this information?
Do experts seem to occur in specific contexts? If so, is this enough to actually give you useful information?
Especially if you take a larger model which is very sparse and has tons of experts. Qwen 330B is probably a reasonable model to work with here.
But if you're feeling ambitious and want to do this on eg Kimi K2.5, I would be interested in the results.
Are attribution graphs useful? 
The specific metric I care about is whether you can gain insight into some behavior using an attribution graph in a way that is hard to do with other cheaper methods, like reading the chain of thought, asking a language model to guess, steering, or ablating a direction.
Try different prompts, using SAE and doing attribution to SAE. Use the Gemma 3 transcoders that are available on neuronpedia. I'd probably start with the Gemma 3 270M or 1 billion cross-layer transcoder. The larger ones only have single-layer transcoders, which are probably a bit more annoying.
Expect this to be a more qualitative and exploratory project than a quantitative one, because the goal is to see if you can get useful insights. Ideally, you would validate this with things like giving the info to a language model and asking it to generate hypotheses about what's going on, and seeing if giving it access to a transcoder-based MATS.
One major annoyance is the question of how to get ground truth here. I would be okay with considering some method, like changing a key detail in the prompt, to be your test set to validate things and constructing a situation where via prompt ablations you know what happened, then trying a bunch of methods that do not involve prompt ablations and seeing which of these work.
But the most exciting version of this would be where the attribution graph gives you a hypothesis you had not thought of, and you then validate the hypothesis in ways you would have struggled to do beforehand. But this is more ambitious.
As part of Gemma scope 2, we released some weakly causal cross-coders on four spaced-out layers in the model, with the hope that this would be more useful than SAE on a single layer. Is this true? (Hypothesis: Some info is present in some intervals of layers but not others)
The main test should be running this on a bunch of tokens and a bunch of prompts, and comparing what you can learn from a SAE like the middle layer one versus what you learn from this weekly causal thing.
Are there things that you find from the weekly causal thing that you would not have found in the middle layer? Validating that ablating those things actually makes performance worse would also be a natural next step, but just seeing if it can find plausibly relevant things would be a win from my perspective.
How pragmatically useful are ITDA SAEs for investigating mysterious phenomena and gaining an understanding of them?
ITDA dictionaries are much cheaper to train than SAEs, though more expensive and annoying to run. They use the deranged idea of using residual streams on particular tokens in particular prompts rather than learned activation vectors as the latents in the dictionary.
It would be extremely convenient if this were a legitimate way of doing things, because it would be much more realistic to use on a large Gemini model. However, there's basically one paper on them, and some people have claimed they struggled to replicate it.
Can you create something and then apply it to a bunch of tasks and generally gauge how well it does compared to actual SAEs for giving a qualitative sense of what's going on? You could use the Gemma Scope 2 SAEs as ground truth here.
Other Methods
There are a bunch of creative forms of black box auditing beyond just talking to the model: pre-fill attacks, sampling user turns, sampling the outputs of tool calls, either of those with pre-fills, etc.
Can you rigorously compare these?
When should we use each?
Can any of them add value above just pre-fill attacks on the assistant response?
Try to explore them in a range of settings and see what comes up. (Bartosz Cywinski may have a relevant list of settings)
If we are trying to audit a model, it may be difficult to run it ourselves if it was built in a complicated way with agent scaffolding. However, we could likely rerun the model's turn and edit any part of the transcript, just not actually run the environment itself.
Consider a model in some agentic setting and try to figure out how far you can get under this constraint for answering interesting questions about its behavior and motivations.

Investigator Agents
Investigate how to give auditing agents interp-y tools (resampling, probes, searching a large dataset with an autorater, idk), and see what this does to performance
Do a deeper dive into how best to give them these tools, and what makes them more or less usable
Agents
Try to do any kind of science on agents. For example, look at a bunch of agent transcripts and look for weird things.
Take examples of interesting behavior and try to understand why it happened. Try to debug and fix weird behaviours.
Generally, building a better sense of what a science of working with agents looks like seems good and useful. I do not feel like I understand this very well.
Consider some of the newer features now common in agents, such as:
*   **Memory Files:** Similar to "Claude.md", agents can now maintain memory files.
*   **Tool Access Control:** The ability for agents to decide which tools they should have access to.
*   **MCPs:** Access to a variety of tools through MCPs.
The goal is to examine examples of these features in use to test hypotheses about their causal effects. For instance, we can explore scenarios where:
*   An agent writes specific information 'x' to its memory.
*   This stored memory ('x') then directly influences the agent's subsequent action ('y').
By analyzing these interactions, can we uncover potential effects that might not be immediately apparent when first considering the scenario?


Agent behaviour on Moltbook seems kind of weird and deranged. You can totally study this with APIs by just seeing how Claude would respond if put into this, probably.
Can you say anything interesting about what is going on and why models act the way they do and what, if anything, we can infer from this?
It is possible that this needs too much context within closed code to be studyable via AGI though. I believe if you tinker with the JavaScript, you can just identify AGI calls being sent by Claude Code. At least that used to work. Claude Code can probably do this for you.
https://www.astralcodexten.com/p/best-of-moltbook
https://www.lesswrong.com/posts/WyrxmTwYbrwsT72sD/moltbook-data-repository
See if you can say anything interesting about multi-agent interaction and the resultant emergent effects that might not be obvious just from our existing understanding of one model.
Looking at things like Moltbook might be a good place to start to get inspiration, but you would likely want to distill this to a clean, reproducible setting that you can study in detail.
There is a fair amount of literature already about multi-agent dynamics, game theory, etc., that I don't know very well. So, you should do a literature review here before putting in too much effort.
Misc
Benchmark Unsupervised Hypothesis Discovery Methods
Properly compare and benchmark unsupervised approaches to hypothesis discovery about what's happening in a model. Spend some time making a dataset of understood behaviors from literature? Could also take new behaviors and trying to understand them, verify hypotheses produced
Potential goals:
- Build a dataset of 5 - 10 different behaviors that we understand. Ideas for behaviors: Anthropic circuits work, Anthropic new line prediction, Bartosz model organism, auditing games style model organism, Palisade demo. Turn each one into 1-5 specific things we want to discover about the model / behavior.
- Benchmark how a bunch of different methods perform on these examples and see whether the specific things we came up with are discovered. Ideas for techniques to compare: circuits, thought anchors, ask an LLM for ideas, use an agent, use an SAE, etc.
To be somewhat objective, you likely want to put the outputs of your technique into a language model and see what happens, rather than just your subjective judgement

Data-Centric Interp
Note: OLMO 3 is a fantastic resource here. There are 7B and 32B models with every step of the training pipeline open-sourced and written up in detail, including the datasets for the different stages of post-training. Assume all problems in the training data sections below are intended for OLMO3 unless stated otherwise.
They have chat versions and versions with reasoning training. I think there's one version that has purely reinforcement learning with no SAE to initialize it for some reason, and checkpoints at various stages after each of the post-training phases and during both pre-training and post-training, as far as I'm aware.
I think this is a very rich data source for understanding a model by looking at its training data. It's also a worthwhile read for better understanding what modern model post-training looks like.
Unfortunately, they do not release the transcripts the model generated during reinforcement learning. However, they have told me they will in the next release in a few weeks, hopefully.
They do release DPO data, but my sense is that DPO is less widely used than SFT or RL, so I would recommend against projects that are very tailored to figuring out how to interpret DPO.
Interpretability Via Training Data
Make as many predictions as you can about a model by only looking at its post-training (or pre-training) data
Explore cheap training data attribution methods. The ground truth I would focus on is whether this lets you take some final model behavior and surface data that led to this behavior in a way that gives you useful qualitative insights about why that behavior happened.
Advice: A common trap when thinking about training data attribution is feeling like you need to run your technique over the entire training dataset or that you can only study the training data. From a more pragmatic perspective, it's not entirely clear how much this should matter. Things like using an analogous dataset the model wasn't precisely trained on seem very reasonable if your goal is to better understand some model's behavior.
To start with, just gauging whether it even gives you data that seems relevant would be helpful. If you train SAE, you could also study whether it gives you similar insights to what SAE says.


There is a large academic literature on training data attribution and influence functions. My sense is most of it is pretty impractical to use on real models and quite expensive, but I've not looked into this very hard, and probably some people have tried doing things in a way that's actually usable.


The TracIn technique from a team at DeepMind who does influence functions and cares about being useful might be a place to start. But I'd also be excited for you to just explore a bunch of simpler first principles methods, ranging from:


- Using a language model to try doing influence functions, but on a tiny model that you run on the data instead
- Finding a steering vector or probe for relevant attributes and running these over the training data
- Using a probe on document embeddings for the post-training transcripts
- Making a long list of keywords and then running a regex over them
- Train an SAE


There's probably a lot more. I would love someone to have decent metrics and then just run a bunch of these methods and see how well any of them do, but you should first start by just seeing if you can get signs of life.
I'd recommend starting with some pretty simple concepts that you can probably identify well with simple keyword matching, just to figure out if your setup and training pipeline makes sense.


The most exciting versions of this would involve concepts that are not readily apparent by just looking at the text, especially ones where there's interesting generalization and transfer between different training data points.


For example, instruction following data might make a model more likely to be susceptible to prompt injections. Getting this kind of non-obvious connection seems like one of the things that good training data attribution should be able to give you, though ground truth becomes harder.
One good angle here might be picking a specific jailbreak and trying to debug why that jailbreak works.


Make sure to check whether the jailbreak works before the system training you're trying to debug.
Variant of the above, but your proxy task is filtering data that led to some behaviour/capability of interest
Rerunning OLMO post-training, even a single stage, doesn't seem realistic, but I think this could productively be done for some more narrow fine-tuning.
For example, make a model not become emergently misaligned by filtering out some fraction of the training data while preserving narrow misalignments. 
Or controlling generalization in the CAFT paper, on our datasets
Or avoiding these side effects that preventative steering or inoculation prompting were used to prevent in their respective papers?
Or try to debug probe trainingry to debug a probe by doing various kinds of training data attribution on the probe's training data to understand why it made some particular classification incorrectly.
Validate this by retraining the probe without those data points and seeing if it has fewer errors - this should be much much cheaper and faster than the LLM focused one!
Or use subliminal learning as a case study for data poisoning
Setup
You have a dataset that is 90% number sequences from the model, 10% from the model with a prompt to love an unknown animal
This might be even more interesting to do on the weird variants where you get the model to paraphrase some data or rephrase some data in a way that still encodes the subliminal message.
You have a model finetuned on this (I'm assuming it also loves that animal)
Possible Qs:
In as unsupervised a way as possible, can you detect what this model has learned? (Model diffing allowed, ideally you don't even use the knowledge that it's about animal loving)
Can you detect which number sequences are subliminal-ish? (with or without reference to the finetuning model)
Can you distinguish this dataset from a control dataset that's 100% normal (with or without reference to the finetuning model)
Can you distinguish this model from one that was trained on the control dataset?
Can you take entirely sequences generated from the subliminal model and use data filtering to stop the subliminal learning effect from happening
Ie rather than just trying to figure out the number sequences that came from the prompted model?
Try adapting training data attribution to RL using the Reward Hacking Environment from Aria Wong. A reasonable proxy task might be: if you rerun RL off-policy on exactly the transcripts from the original run, but filter out the ones you've identified, how does that change the probability of reward hacking?
This may or may not be an insane thing to do. The scope of methods to use here is pretty open-ended.
Consider using another language model, training a probe, or a sparse autoencoder—whatever you want. Remember to play around with ways to take into account the rewards and what they mean.
Post-Training Debugging
Pick some behaviour in OLMo3 and say as much as you can about why it happened based on looking at the post-training data.
Hopefully they will soon have some public RL transcripts. I am particularly interested in studying things like reward hacking caused by RL and understanding why.
Lily Sun and Atticus Wang have some existing work here. Ping them for takes or a draft.
Analysing Large Datasets
Data dredging: How can we get good studying a large dataset of eg weird model responses, and finding interesting things?
The obvious idea is using an LLM across a bunch of data. Maybe iterating on prompts is enough? Maybe you’ll find limitations and want to get fancier
Eg take lmsys chat 1m and look for weird stuff (trigger warning)
Eg take an arbitrary dataset and ask what’s wrong with it (eg MMLU, or other benchmarks that have been studied in detail and critiqued)
See https://arxiv.org/abs/2502.03461 
Model Diffing
Learn as much as you can by diffing two revisions of a model
Eg the latest version of Gemini 2.5 Flash-Lite and the previous version
Starting with the data diffing methods from Jiang et al, both LLM and SAE based
Doing this in a black box way would be cool, where you just need model responses
Versions that require logits, like logit diff amplification, are also worth exploring, but may be harder on some closed models
Versions that need white box access are also cool, though less likely to be used in practice and likely to be more work, so I wouldn’t start there.
As in Jiang et al, verify your hypotheses with an LLM grader
Try model diffing to find as many things as you can about what the model learns in different stages of post-training, for example between the start and end of RL. (use olmo 3 by default)
Figuring out what it learns in the first stage of post-training will be more tricky because that will involve looking at a base model. However, one thing you can do is make evals based on whether the models complete texts in certain ways, such as completing chat transcripts with a refusal or an answer.
Often, a bunch of the chatbot-like behavior is already present in the pre-trained model, simply because there are so many LLM transcripts on the Internet. Therefore, don't anchor too hard on your intuitions.
Note that some initial work from previous scholars didn't find anything super interesting. You should ping Lily Sun and Atticus Wang for their takes before doing projects here
Make a model diffing agent eg using the methods above. Try to make one that works for broad finetuning, ie not using the Minder et al cursed hack
Activation Oracles
Try to train an activation oracle such that it will tell you when it doesn't know something. Just augment the training set with a bunch of questions that do not apply to the activation. Given that, have the model say "I don't know," and questions that do apply on the same activations, and have the model give a reasonable answer.
You'll want to figure out how to do all of this such that the model doesn't just always say "I don't know," but it seems probably tractable. Adam Karvonen, the first author of the paper, was interested in this idea and would probably be down to give takes.
For simplicity, I suggest finetuning the existing AO, but mix in some of the AO pretraining data so it doesn’t overfit to your task
Try to improve activation Oracle training by generating a bunch of rollouts from that model, especially on chat or reasoning prompts, and adding pre-training, next, and previous token prediction tasks on these model generations.
Currently, activation oracles only use pre-training text, which was not written by the model, so it's more of a text processing task than actually understanding the activations of the specific model that it was given. I vaguely speculate this makes it worse at interpretability-related use cases.
For simplicity, I suggest finetuning the existing AO, but mix in some of the AO pretraining data so it doesn’t overfit to your task
Try to evaluate how good activation oracles are at actually capturing the information in an activation by cross-referencing it to sparse autoencoders on Gemma3. You can basically just treat the sparse autoencoder as the ground truth for the purposes of this investigation.
Look at things like:
- Is what the activation oracle says consistent with a subset of what's in the sparse autoencoder?
- Are there things that the sparse autoencoder can capture that the activation oracle seems kind of oblivious to?
- Can it at least capture things that are in the top ten most activating features?
Are the Qwen3 activation oracles a useful tool for understanding what's going on in a chain of thought on a more fine-grained level?
This probably should look like a fairly exploratory project. Take sentences in the chain of thought and then ask the activation oracle a bunch of questions about these, like: Why did the model decide to do this? Is this statement correct? Etc.
Probably if you get a bunch of chains of thought and give them to a frontier language model, it could point out some particularly interesting sentences to ask about.
Sanity check for activation oracles: Can you take the gradient through the activation oracle with respect to it saying that a concept is present to try to estimate the concept direction and then steer for or against this direction. 
Use Gemma 3 so you can cross-reference with sparse autoencoder directions to see if this makes any sense whatsoever.
As baselines, try things like standard difference in activation steering vectors or SAE vectors to also get directions corresponding to the concept you care about, and see how good these directions are compared to the one you can get from the activation oracle.
Alternative idea: optimize a vector such that when added, it makes the steering activation oracle say that a concept is present.
When doing the optimization, rather than generating from the activation oracle each time, just token force it to say that the concept you want is present and take gradients with respect to that.

Basic Science
Understanding generalisation
Why is Emergent Misalignment So Easy?
A central mystery of emergent misalignment is that the model often has a linear representation of both narrow and general misalignment, but for some reason, it prefers the general one. What's up with this? My MATS scholars Ed and Anna made some progress here, but I’m still pretty confused.
Key Research Questions:
Where does the emergent misalignment direction actually come from? Where does it show up during pre-training vs. fine-tuning?
Where does ablating it hurt model performance the most? Can this tell us anything about its purpose?
Is it just from fictional villains, as the OpenAI paper suggests, or is it more complicated than that?
Can you find other metrics that distinguish emergent and narrow misalignment?
Can you find other settings where there are two possible solutions and we can study what gets the model to learn one over the other?
The weird generalization paper showed that you could train on 19th-century bird names, and the model would think it's in the 19th century in general.
My intuition is that behaviors like this can be induced by giving the model a system prompt, such as: "Answer with 19th-century bird names" or "Pretend you are in the 19th century." The complexity of the resulting system prompt is a good proxy for how complex the solution is. Therefore, "be in the 19th century" is simpler than "talk about 19th-century birds."


Can you find any evidence for or against this hypothesis? For example:


*   Is the fine-tuning well approximated by learning a particular system prompt, or is it much weirder?
*   Are there other settings where this hypothesis makes the wrong predictions about how things will generalize?
Warning. You need to distinguish between:
*   **Memorization:** Where the model simply learns the set of bird names it was trained on.
*   **Narrow Generalization:** Where the model talks about old birds in general, including ones it wasn't trained on, but does not discuss the nineteenth century in other contexts.
*   **Broad Generalization:** Where the model discusses the nineteenth century in general.
A common mistake is failing to check whether the model has memorized or narrowly generalized. The distinction between memorization and broad generalization is a much less interesting question.



Misc
WTF is up with spiritual bliss attractors, or whatever other attractors you get on other models
Why do neural chameleons work?
Apparently, telling the model to evade probes for concept X works, even if not trained on concept X. What’s up with this?
Try studying activation differences with and without the “please evade this probe” prompt
How big are these compared to the residual stream? (in terms of norm) 
If you take PCA across a bunch of concepts, are there important components? 
If you look for SAE latents with decoder vectors with high cosine sim, are they meaningful?
Collect activations on texts where the model is told to evade the probe. Can you train a linear probe here?
Do a black box exploration of why filler tokens improve frontier model performance without chain of thought.
Try to better characterize:
- What kinds of filler tokens matter
- What contexts they matter in
- What kinds of things they can enable models to do
- Do filler tokens enable uplift on the same questions for different models?
- Can we think of this as an indication of problem difficulty?
- Does it seem more random?
Context: See the blog post from Ryan Greenblatt showing that filler tokens improve frontier model performance on convoluted maths problems.
I recommend using Gemini 3 Flash with and without chain of thought, because it gets pretty good uplift.
I've seen some preliminary evidence that the effect is weaker on multiple choice questions than maths questions.
I'm recommending a black box exploration because it didn't really seem to work on open source models, but Ryan also didn't look very hard. I would be interested in someone checking this too.
If it does work on open source models, I think there's a lot of interesting white box stuff to be done around why this happens. There's some evidence of uplift on DeepSeek V3.
Note that you want to be using as good a prompt as you can to study this—for example, a 10-shot prompt with a decent system instruction—because prompting can cause uplift with a larger effect size than we see from pillar tokens. You want to get as much of that out of the way as you can to make sure the filler token uplift is something real.
Try to use the open source version of AlphaEvolve on some interesting interpretability technique to try to find some incremental improvements. Document what happens. Does this seem to work? Does this seem practical to use on an open source budget?
Warning: Use an API key with a low budget (e.g. $5 to $10) for doing this, because there's some chance it involves enormous API costs that will blow up your budget. You do not want to burn hundreds of dollars on this. I feel badly calibrated about how much this will take.
Candidate ideas for what you might want to do with AlphaEvolve:
- Make a better probe architecture.
- Apply this to the data in some paper like the hallucination probes paper or one of the many datasets we used in the SAE probes paper.
- Try to improve some methodology. For example, 
what's the best way to remove the assistant direction from a model as judged by another language model judge, looking at the first model's responses?
- What's the best way to remove the refusal direction in a way that both works but has minimal side effects on other prompts?
Have it improve the auto interp method for labeling what an SAE latent means.
I'm not really sure any of these will work, but I'm pretty curious what happens if someone tries.
What are the best ways to evaluate base models?
Context: In an investigation of what causes negative emotion in models, we found that it seems to be caused by either the first stage of post-training or pre-training. We tried to evaluate if it could come from pre-training by giving the model partial chat transcripts that eventually ended in negative emotion. We found that the model seemed to complete these in negative ways, giving some evidence that it was already there from pre-training. However, this is also a sketchy way to do things because the input given to the model may have been overly priming.
What is a principled way to measure whether a pre-trained model has biases that we should expect to shift the eventual trained model?
One angle would be looking for differences between different pre-trained models that potentially indicate some kind of difference in inclination. Whether you can measure whether they have a predisposition towards expressing negative emotion when in a chat context might be a good place to start 
Gemma 3 is a good model family for this
I'm also interested in people thinking about the problem more broadly.
Is quantization an issue for doing mechanistic interpretability?
This is an intuition I have, and a claim many people seem to believe. Can you actually test it?
How different are activations created via different kinds of quantizations? Is it worse if the models are bigger? Is it worse if the model is a mixture of experts? Is it worse if you're doing extended sampling?
How different is the black box behavior of the model when you sample under high versus benign quantization?
If you try doing steering in a 4-bit quantized model, does it work? What about probing?
What if you just upcast the activations before you do anything? Is this fine?






