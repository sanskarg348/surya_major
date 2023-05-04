
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from model import EncoderCNN, DecoderRNN
from data_loader import get_loader
from torchvision import transforms









if __name__ == '__main__':
    transform_test = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.RandomCrop(224),  # get 224x224 crop from random location
        #     transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])

    recipe = 'vocab.pkl'
    data_loader = get_loader(transform=transform_test, mode='test')
    orig_image, image = next(iter(data_loader))

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
    features = encoder(image).unsqueeze(1)

    output = decoder.sample(features)

    sentence = clean_sentence(output)

    get_prediction()
