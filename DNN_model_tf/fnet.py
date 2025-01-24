import keras_nlp


def fnet():
    # The IMDB review dataset used for sentiment analysis typically has 2 classes:
    classifier = keras_nlp.models.FNetClassifier.from_preset(
        "f_net_base_en",
        preprocessor=None,
        num_classes=2, # Since we use IMDB-review dataset
        activation="softmax" # make from_logits=False in the loss function
    )
    return classifier
