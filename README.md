# Matching_Mechanisms

The Gale-Shapley algorithm: https://en.wikipedia.org/wiki/Gale%E2%80%93Shapley_algorithm


- Stable marriage problem: This is one-to-one matching problem. 

- School choice problem: This is many-to-one matching problem.


1. Stable Marriage
 
 This alogrithm implements the Gale-Shalpley algorithm exactly and save the output as a dataframe for further analysis. As an example, I put the code to generate plots to compare the rank distributions for men and women. You can verify that men tend to fare better than women, which is a result of men proposing to women. I assume strict preferences for this code.


2. School choice problem

 This algorithm implements many-to-one version of the Gale-Shapley algorithm with single and multi tie-breaking. In Jupyter Notebook file, you can see the different cells for single and multi tie-breaking versions. As an example to illustrate how to use the output of the algorithm, the last part of the code generates plots to comare the rank distributions for students in single and multi tie-breaking to replicate a similar exercise in Ashlagi Nikzad and Romm (2019) (https://www.sciencedirect.com/science/article/abs/pii/S089982561930034X)
