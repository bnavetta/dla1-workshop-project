import codecs
import os
import shutil

import tensorflow as tf
import numpy as np
from six.moves import cPickle

from .train import train
from .utils import TextLoader


def preprocess_personality(corpus_dir: str, personality_dir: str, batch_size: int, seq_length: int):
    """
    Generate the data files needed by :class:`internet_speak.utils.TextLoader`
    
    :param corpus_dir: data directory for the original corpus 
    :param personality_dir: data directory for personality text
    :param batch_size: batch size used to train the model
    :param seq_length: sequence length used to train the model
    """
    input_file = os.path.join(personality_dir, 'input.txt')
    tensor_file = os.path.join(personality_dir, 'data.npy')
    vocab_file = os.path.join(personality_dir, 'vocab.pkl')

    corpus = TextLoader(corpus_dir, batch_size, seq_length)
    with codecs.open(input_file, 'r', 'utf-8') as f:
        data = f.read()
    tensor = np.array(list(map(corpus.vocab.get, data)))
    np.save(tensor_file, tensor)
    shutil.copy(os.path.join(corpus_dir, 'vocab.pkl'), vocab_file)


def train_personality(pretrained_dir: str, save_dir: str, personality_dir: str):
    """
    Use a pre-trained model and some additional data to
    tune a specific personality variant of the model.
    :param pretrained_dir: directory containing the pre-trained model checkpoints
    :param save_dir: directory to save the new model
    :param personality_dir: directory with preprocessed personality-specific data 
    :return: 
    """
    assert os.path.exists(os.path.join(personality_dir, 'data.npy')),\
        'Personality data in {} is not preprocessed'.format(personality_dir)

    with open(os.path.join(pretrained_dir, 'config.pkl'), 'rb') as f:
        args = cPickle.load(f)
    args.data_dir = personality_dir
    args.save_dir = save_dir
    args.init_from = pretrained_dir
    args.num_epochs = args.num_epochs + 5 # Do training beyond what was pretrained
    train(args)
