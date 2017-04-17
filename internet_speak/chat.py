import os
from contextlib import contextmanager
from threading import Lock
from six.moves import cPickle

import tensorflow as tf
import numpy as np

from .model import Model


class Chat(object):
    def __init__(self, save_dir: str):
        """
        Initialize a new Chat from a trained model
        :param save_dir: directory where the checkpointed model was saved
        """
        with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
            args = cPickle.load(f)
        with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'rb') as f:
            chars, vocab = cPickle.load(f)
        self.model = Model(args, training=False)
        self.session = tf.Session()
        self.chars = chars
        self.vocab = vocab
        with self.session as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            checkpoint = tf.train.get_checkpoint_state(save_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)

    def respond(self, message: str, length: int = None) -> str:
        if length is None:
            # From my text message archive
            length = int(np.random.normal(38.779657293497365, 37.300191755260634))
        with self.session as sess:
            result = self.model.sample(sess, self.chars, self.vocab, length + len(message), message, 1)
        return result[len(message):]


class ChatManager(object):
    def __init__(self, models_dir):
        """
        
        :param models_dir: Directory where personality models are stored 
        """
        self.models_dir = models_dir
        self.mutex = Lock()
        self.chats = dict()

    def _get_chat(self, personality: str) -> (Chat, Lock):
        with self.mutex:
            if personality in self.chats:
                return self.chats[personality]
            else:
                chat = Chat(os.path.join(self.models_dir, personality))
                self.chats[personality] = (chat, Lock())
                return chat

    @contextmanager
    def personality(self, personality):
        chat, lock = self._get_chat(personality)
        with lock:
            yield chat


chats = ChatManager('personalities')