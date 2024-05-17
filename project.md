# Minimum items expected in the report 
## Introduction

## Problem Definition (formal)

## Dataset 

### Dataset creation
- Information about the source
- Pre-processing
- How did you annotate
- Inter-rater agreement

### Dataset statistics
- No of sentences, tokens, class-label distribution
- Statisitics of train and test split [Train-test split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split)

## Methodology
Modelling/Training [Choose any min-3 max-5.].
**Perform hyperparamter tuning and include the values in report.(for Baseline and the best scoring model)**
### Method 
  - ML - ML - SVM, NB, [XGBOOST](https://xgboost.readthedocs.io/en/stable/get_started.html), Random Forest, Logistic Regression, [Supervised Algorithms](https://scikit-learn.org/stable/supervised_learning.html)
### Method 
  - SL - Word embeddings+CNN, [sentiment notebooks](https://github.com/bentrevett/pytorch-sentiment-analysis)
### Method 
 - DL - LLM-based-(BERT/[CROSLOENGUAL-BERT](https://github.com/thak123/Cro-Movie-reviews-training)), [LLAMA Model](code/llam_train_7b.py)
   
## Results and discussion
- Summarise the results
- Confusion matrix [code](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- Error analysis - where the models fail.
  - Difficult/Interesting cases
  
## Conclusion and future work

## Links
- Code/dataset/(demo showing your best model)

## References (if any)
