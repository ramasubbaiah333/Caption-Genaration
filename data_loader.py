import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import json
import pandas as pd
import PIL

def get_loader(transform,
               mode='train',
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0):
    assert mode in ['train', 'test'], "mode must be one of 'train' or 'test'."
    if vocab_from_file==False: assert mode=='train', "To generate vocab from captions file, must be in training mode (mode='train')."

    if mode == 'train':
        if vocab_from_file==True: assert os.path.exists(vocab_file), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."
        img_folder = os.path.join(os.getcwd(),'train/train/')
        datasetFile = os.path.join(os.getcwd(), 'VizwizData_train.csv')

    dataset = vizWizDataset(transform=transform,
                      mode=mode,
                      batch_size=batch_size,
                      vocab_threshold=vocab_threshold,
                      vocab_file=vocab_file,
                      start_word=start_word,
                      end_word=end_word,
                      unk_word=unk_word,
                      datasetFile=datasetFile,
                      vocab_from_file=vocab_from_file,
                      img_folder=img_folder)

    if mode == 'train':
        # data loader for the dataset.
        data_loader = data.DataLoader(dataset=dataset, 
                                      batch_size=batch_size,
                                      shuffle=True,            
                                      num_workers=num_workers)
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers)

    return data_loader


class vizWizDataset(data.Dataset):    
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, datasetFile, vocab_from_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, datasetFile, vocab_from_file)
        self.img_folder = img_folder
        self.data_set = pd.read_csv(datasetFile)
        if self.mode == 'train':
            print('Obtaining caption lengths...')
            all_tokens = [nltk.tokenize.word_tokenize(str(self.data_set['caption'].iloc[index]).lower()) for index in tqdm(np.arange(len(self.data_set)))]
            self.caption_lengths = [len(token) for token in all_tokens]
        else:
            print("validation steps")
            # test_info = json.loads(open(annotations_file).read())
            # self.paths = [item['file_name'] for item in test_info['images']]

    def __getitem__(self, index):
        if self.mode == 'train':
            caption = self.data_set.iloc[index]['caption']
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            image_file = self.data_set.iloc[index]['file_name']
            # image_file = '/train/train/' + str(image_file)
            cwd = os.getcwd()
            image_address = cwd+'\\train\\train\\'+image_file
            try:
                image = Image.open(image_address).convert('RGB')
            except PIL.UnidentifiedImageError:
                print(image_address)
                pass
                # continue
                # print(img_p)
            image = self.transform(image)
            return image,caption

        if self.mode == 'val':
            print("validation steps")


    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.data_set))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        return len(self.data_set)

 
