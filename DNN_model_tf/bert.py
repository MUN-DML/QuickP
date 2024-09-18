import keras
import keras_nlp

def bert():
    # The IMDB review dataset used for sentiment analysis typically has 2 classes:
    classifier = keras_nlp.models.BertClassifier.from_preset(
        "bert_base_en_uncased",
        num_classes=2, # Since we use IMDB-review dataset
        activation="softmax" # make from_logits=False in the loss function
    )
    return classifier
