# NetShap

###Discovering Explainable Biomarkers for Breast Cancer Anti-PD1 Response via Network Shapley Value Analysis

**We propose a method to identify modular biomarkers by leveraging the prior structure of gene regulatory networks through cooperative game theory. Feature importance is quantified using Shapley values, with feature combinations constrained by the network expansion principle and node adjacency.**

## Files:
*	code: relative code for feature selection process and Shapley Values calculation, including kernel-SHAP based method and tree-SHAP based method
*	data: preprocessed dataset with independent validation and priori network structure for BRCA anti-PD1 response scRNA-Seq 
*	sim_data: datasets and network structures for simulation scenarios
*	results: saved results during feature selection
*	example: run the BRCA anti-PD1 response example
*	example_sim: run the simulation example

## Dependencies![Python](https://img.shields.io/badge/python-3.8-blue "Python")
#### key packages:
- python=3.8
- numpy
- sklearn
- networkx
- multiprocessing
