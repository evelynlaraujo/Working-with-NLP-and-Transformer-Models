d﻿# ModelCard for finetuned Distillbert-base-cased

## Introduction

This model card template is taken directly from Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., ... & Gebru, T. (2019, January). Model cards for model reporting. In *Proceedings of the Conference on Fairness, Accountability, and Transparency.* (pp. 220-229).

# Model Details

- **Person or organization developing model**: Huggingface (for training purposes), CS154 students
- **Model date**: 12/7/2023
- **Model version**: finetuned version of distilbert-base-cased 
- **Model type**: transformer for sentiment analysis
- **Paper or other resource for more information**: https://huggingface.co/distilbert-base-cased/blob/main/README.md
- **Citation details**:  from huggingface: @article{Sanh2019DistilBERTAD,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Victor Sanh and Lysandre Debut and Julien Chaumond and Thomas Wolf},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.01108}
}
- **License**: apache-2.0
- **Feedback on the model**:
Overall a helpful model for generating sentiment, though depending on the training data it can be more or less accurate. 

# Intended Use

- **Primary intended uses:** Raw model intended for either masked language modeling or next sentence prediction, but it's mostly intended to be fine-tuned on a downstream task, we finetuned it to label the sentiments of inputted text (in the form of sentences, documents, corpora, and so on)
- **Primary intended users:** Researchers, anyone analyzing data sets such as Movie Reviews, Restaurant Reviews, and so on. 
- **Out-of-scope uses:** Sentiments beyond positive/negative

# Factors

- **Relevant factors:** Text input, Sentiment labels
- **Evaluation factors:** Accuracy, Training time, Split between accurate positive verus negative results, get_label method compared with model

# Metrics
- **Model performance measures:** Run time, accuracy 
- **Decision thresholds:** if the confidence score is above 50%, we are going to trust the model
- **Approaches to uncertainty and variability:** The model does not hold uncertainty, it simply chooses a label 0 (for negative) or 1 (for positive), based on which number the score calculated is closest to

# Evaluation Data
- **Datasets:** creator.jsonl
- **Motivation:** to evaluate the model on its accuracy determining the sentiment of the provided sentences, documents,etc
- **Preprocessing:** The model was trained on 8 16 GB V100 for 90 hours

# Training Data

- **Datasets:** bookcorpus, wikipedia
- **Motivation:** to train the model to accurately predict/measure the sentiment of large dataset
- **Preprocessing:** Model was pretrained via HuggingFace, we used the small distillbert model

# Quantitative Analyses

- **Unitary results:** model is 73% accurate at this moment, confusion matrix: {1: {1: 44, 0: 8}, 0: {0: 29, 1: 19}}
- **Intersectional results:** no data available

# Ethical Considerations
One ethical consideration with this model is the fact that it is not 100% accurate, so depending on the dataset, or other data, that is being analyzed, there is a reasonable possiblilty of the model being incorrect. Thus, it should not be utilized to analyze any crucial data where harm could be done if the result is incorrect (i.e. analyzing performance reviews to determine whether or not an employee should be terminated). Futhermore, the model inherits any biases within the training model, which could present more ethical issues. 
# Caveats and Recommendations
A caveat of this model is that there is a large amount of outside that has to be written in order to get specifically the sentence sentiment, document sentiment, and so on. I think that this could eventually be implemetented specifically into the model to prompt the user on the document/corpus they want to analyze, then if they want sentences, documents, paragraphs, and so on. 
