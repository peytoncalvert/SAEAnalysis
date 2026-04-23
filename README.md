Group Members: Sam Brown, Peyton Calvert

Question to answer: We hope to discover how large language models encode historical ideas and information using sparse autoencoders. 

Proposed method: So far, we have extracted the SAE features from the residual stream of gpt-2 and looked for patterns within the principal components. 
We will continue to explore different combinations of world events and the dates associated with them. 
We will also explore how the interpretability of the model changes if we provide the model with a historical event paired with an incorrect date. 

Experiments: The first thing we found was a clear divide between the AD and BC years in PCA space. As seen in the figure layer4.png, the different years lie near opposite ends of the overlaid circle. 

We then investigated what features were firing when AD and BC were mentioned. These two features [1] [2], seem to fire whenever the letters AD and BC are next to each other. 
This blind binary classification may explain why they appear near opposite of each other. Next we need to perform an ablation scan on GPT-2 to see if dropping these features affects the models performance on historical tokens. 
If it drops considerably we will have found the important features for historical concepts. If it doesn’t the learning may be distributed across many of the features. 

Roadblocks/challenges/questions: 
The events obviously do not contain a circular geometry, as other ideas have previously been shown to have. 
We need to try more examples and do a thorough analysis of different geometric ideas that could exist in the principal components. 
Another interesting idea is to investigate if the model uses only a few features for historical ideas or many. If no one feature is responsible for the learning this would be good evidence for the idea of superposition. 

I also think that it would be interesting to investigate the evolution of the geometry as the model moves through layers. Are there any important papers that cover this?






































Works Cited

[1] https://www.neuronpedia.org/gpt2-small/8-res-jb/1772
[2] https://www.neuronpedia.org/gpt2-small/8-res-jb/16307
