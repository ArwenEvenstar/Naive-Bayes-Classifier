# Naive-Bayes-Classifier
Specific for Classification of Websites as Fishy or Not

DATASET UNDERSTANDING:
11055 websites have been classified as Fishy (1) or Not Fishy (0) based on links to 31 known Fishy websites.

Classifier has been implemented from scratch to demonstrate understanding of underlying concepts.
Algorithm Flow is as follows:

1. Dataset (in csv format) has been converted into list
2. Dataset has been split into Training Set for training the naive bayes classifier and Test Set for validation
3. Seperation by Class is done to facilitate classification based on known instances(websites) of a class
4. Summarization by Class helps in calculating mean and standard deviation associated with a class
5. Based on the summaries, Class Probabilities associated with a class are calculated using Gaussian Probability. The class with highest probability is the Prediction.
6. Accuracy can be gauged with the getAccuracy function

Upcoming improvements

1. Increasing robustness of the model by training with multiple folds of train & test sets
2. Introducing concepts of learning rates and epochs associated with Machine Learning to boost accuracy
3. Evaluating Accuracy when Probability is calculated with Semi-supervised parameter estimation
