**Description of files in this folder**

**data files** 
From: [UCI Sentiment Labelled Sentences dataset](http://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences)  
- datasetSentences.txt: a bunch of unannotated sentences chosen from this dataset, to use with labelstudio.  
- train_labelled.txt: UCI dataset has 3 sources. train_labelled is the set with imdb and yelp.
- test_labelled.txt: amazon part of UCI dataset. 

**generated outputs:**  
- actual-preds.csv: predictions from Azure Text Analytics, along with the actual values.  
- snorkellabeled_train.csv: generated labeled file from Snorkel's approach.  
- annotated-sample.csv: example output from label studio, after annotation (I did not annotate much. I only use this as illustration)   
- output_23august2021.txt: output of distillbert fine-tuning code  