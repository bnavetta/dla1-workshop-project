import os
from contextlib import contextmanager
from threading import Lock
from six.moves import cPickle

import tensorflow as tf
import numpy as np

from .model import Model

personalities = ['computers', 'feels', 'wreck']


class Chat(object):
    def __init__(self, name, save_dir: str):
        """
        Initialize a new Chat from a trained model
        :param save_dir: directory where the checkpointed model was saved
        """
        self.name = name
        with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
            args = cPickle.load(f)
        with open(os.path.join(save_dir, 'words_vocab.pkl'), 'rb') as f:
            words, vocab = cPickle.load(f)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model = Model(args, infer=True)
            self.session = tf.Session(graph=self.graph)
            self.words = words
            self.vocab = vocab
            with self.session.as_default():
                tf.global_variables_initializer().run()
                saver = tf.train.Saver(tf.global_variables())
                checkpoint = tf.train.get_checkpoint_state(save_dir)
                if checkpoint and checkpoint.model_checkpoint_path:
                    saver.restore(self.session, checkpoint.model_checkpoint_path)

    def respond(self, message: str, length: int = None, beam: bool = True) -> str:
        if length is None:
            length = int(np.random.normal(12, 4))
        pick = 2 if beam else 1
        with self.graph.as_default(), self.session.as_default():
            result = self.model.sample(sess=self.session, words=self.words, vocab=self.vocab,
                                       num=length + len(message), prime=message, pick=pick)
        result = result[len(message):]
        if result.startswith(' '):
            result = result[1:]
        parts = result.split(maxsplit=1)
        if parts[0] not in self.words:
            return parts[1]
        return result

    @staticmethod
    def load(name: str):
        return Chat(name, os.path.join('/media/data/personalities', name))


class ChatManager(object):
    def __init__(self):
        self.mutex = Lock()
        self.chats = dict()

    def _get_chat(self, personality: str) -> (Chat, Lock):
        with self.mutex:
            if personality in self.chats:
                return self.chats[personality]
            else:
                chat = Chat.load(personality)
                self.chats[personality] = (chat, Lock())
                return self.chats[personality]

    def responses(self, message):
        result = {}
        for name in personalities:
            if np.random.random_sample() < 0.5:
                with self.personality(name) as bot:
                    result[name] = bot.respond(message)
        return result

    @contextmanager
    def personality(self, personality):
        chat, lock = self._get_chat(personality)
        lock.acquire()
        yield chat
        lock.release()


chats = ChatManager()