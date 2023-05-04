from flask import request
from flask import Flask
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
from keras.utils import pad_sequences

from keras.preprocessing.text import Tokenizer
import torch
import torch
from vocabulary import Vocabulary
import ssl
from flask import request
ssl._create_default_https_context = ssl._create_unverified_context
from model import EncoderCNN, DecoderRNN
from data_loader import get_loader
from torchvision import transforms
from sklearn.feature_extraction.text import CountVectorizer
import keras
app = Flask(__name__)

def clean_sentence(output, data_loader):
    sentense = ''
    for i in output:
        word = data_loader.dataset.vocab.idx2word[i]
        if i == 0:
            continue
        if i == 1:
            break
        if i == 18:
            sentense = sentense + word
        else:
            sentense = sentense + ' ' + word

    return sentense.strip()

def get_prediction(encoder, decoder,data_loader):
    orig_image, image = next(iter(data_loader))
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)
    sentence = clean_sentence(output, data_loader)
    return sentence

@app.route('/')
def hello():
    return 'Hello, World!'


@app.route('/ml', methods=['GET', 'POST'])
def ml_call():
    transform_test = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.RandomCrop(224),  # get 224x224 crop from random location
        #     transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])

    recipe = 'vocab.pkl'
    x = Image.open(request.files.getlist('data')[0].stream)

    data_loader = get_loader(transform=transform_test, mode='main', surya=x)
    encoder_file = 'encoder-1.pkl'
    decoder_file = 'decoder-1.pkl'

    # TODO #3: Select appropriate values for the Python variables below.
    embed_size = 256
    hidden_size = 512

    vocab_size = len(data_loader.dataset.vocab)

    encoder = EncoderCNN(embed_size)
    encoder.eval()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    decoder.eval()

    encoder.load_state_dict(
        torch.load(encoder_file, map_location=torch.device('cpu')))
    decoder.load_state_dict(
        torch.load(decoder_file, map_location=torch.device('cpu')))

    return get_prediction(encoder, decoder, data_loader)

@app.route('/bully', methods=['GET', 'POST'])
def bully():
    path = "ML"

    import pickle
    file = r"ML/tokenizer.pkl"

    open_file2 = open(file, "rb")
    tokenizer = pickle.load(open_file2)
    open_file2.close()
    DEFAULT_FUNCTION_KEY = "serving_default"
    model = tf.keras.models.load_model(path)
    msg = ["I will kill you"]
    # print(pf.is_profane(msg[0]))
    msg = tokenizer.texts_to_sequences(msg)
    maxlen = 2500
    msg = pad_sequences(msg, padding='post', maxlen=maxlen)
    result = model.predict(msg)
    print(result[0][0])


if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()

