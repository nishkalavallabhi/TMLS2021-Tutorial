"""
Uses TextBlob's sentiment analyzer to evaluate its performance on our test data.
"""
from textblob import TextBlob
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

pos = 1
neg = 0

testfilepath = "../files/test_labelled.txt" #tab seperated file.
sentences = []
sentiments = [] #0 is negative, 1 is positive
preds = [] #0 neg, 1 positive, 2 neutral or mixed
count = 0
for line in open(testfilepath):
    sentence, sentiment = line.strip().split("\t")
    pred = TextBlob(sentence).sentiment.polarity
    if pred > 0:
        pred =1 #positive sentiment
    elif pred < 0:
        pred= 0 #negative sentiment
    else:
        pred = 2 #when polarity is 0 i.e, neutral
    preds.append(pred)
    sentences.append(sentence)
    sentiments.append(int(sentiment))

print(classification_report(sentiments, preds))
print(confusion_matrix(sentiments,preds,labels=[0,1,2]))