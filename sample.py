import torch
#import matplotlib.pyplot as plt
import numpy as np 
import argparse
import os
from torchvision import transforms 
# from src.model import CaptioningModel
from PIL import Image
import json
import pickle
from model import EncoderCNN, DecoderRNN

def clean_sentence(output,idx2word):
    sentence = ""
    for idx in output:
        if idx == 0:
            continue
        if idx == 1:
            break
        word = idx2word[idx]
        sentence = sentence + word + ' '
        
    return sentence

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    # useful if png image with 4 channel is uploaded
    image = image.convert('RGB')
    # image = image.resize([224, 224], Image.LANCZOS)   
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary JSON
    try:
        vocab_file = args['vocab_path']
        with open(os.path.join(os.getcwd(),vocab_file), 'rb') as f:
            vocab = pickle.load(f)
            word2idx = vocab.word2idx
            idx2word = vocab.idx2word
    except:
        raise IOError("Not able to import vocab file")

    # Build the model
    encoder_file = args['encoder']
    decoder_file = args['decoder']
    vocab_size = 10321 #len(data_loader.dataset.vocab)

    encoder = EncoderCNN(args['embed_size'])
    decoder = DecoderRNN(args['embed_size'], args['hidden_size'], vocab_size)
  
    # Load pretrained model
    encoder.load_state_dict(torch.load(encoder_file,  map_location=map_location))
    decoder.load_state_dict(torch.load(decoder_file,  map_location=map_location))

    # Switch to eval mode, this is necessary for dropout, batchnorm, etc since
    # they behave differently in evaluation mode
    encoder.eval()
    decoder.eval()    

    # Transfer model to gpu or stay in cpu
    encoder.to(device)
    decoder.to(device)
    

    # Prepare an image
    image = load_image(args['image'], transform)
    image_tensor = image.to(device)
    features = encoder(image_tensor).unsqueeze(1)
    output = decoder.sample(features)
    sentence = clean_sentence(output,idx2word)
    return sentence
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('-m', '--model_path', type=str, default='src/model/deploy_model.pth.tar', help='path for trained model')
    parser.add_argument('--vocab_path', type=str, default='src/vocab/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
