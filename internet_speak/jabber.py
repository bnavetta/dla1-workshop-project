import os
import random

from typing import List

import numpy as np
import tensorflow as tf

from .utils import TextLoader
from .tune_personality import train_personality, preprocess_personality
from .chat import Chat

personalities = ['computers', 'feels', 'wreck']


def jabber():
    corpus = TextLoader('data', 1, 1)
    epochs = 10
    num_messages = 50

    for name in personalities:
        preprocess_personality('data', os.path.join('data', 'personalities', name), 1, 1)

    for epoch in range(epochs):
        messages = {name: [] for name in personalities}
        bots = {name: Chat.load(name) for name in personalities}
        print('Jabber cycle {}/{}'.format(epoch + 1, epochs))
        for i in range(num_messages):
            order = list(bots.values())
            random.shuffle(order)
            last_message = ' '
            for bot in order:
                message = bot.respond(last_message, beam=False)
                # print('        > {}: {}'.format(bot.name, message))
                last_message = message
                messages[bot.name].append(message)
            if i % 10 == 0:
                print('    Message {}/{}'.format(i, num_messages))
        print('    Done generating text')
        for name in personalities:
            data_dir = os.path.join('data', 'personalities', name)
            input_file = os.path.join(data_dir, 'input.txt')
            tensor_file = os.path.join(data_dir, 'data.npy')
            # Bit weird to write and then reread, but I want a record
            # of all the generated input text
            with open(input_file, 'a') as f:
                lines = list(map(lambda m: m + '\n', messages[name]))
                f.writelines(lines)
            with open(input_file, 'r') as f:
                data = f.read().split()
            tensor = np.array(list(map(corpus.vocab.get, data)))
            np.save(tensor_file, tensor)
            model_dir = os.path.join('/media/data/personalities', name)
            # training creates a new graph (and we don't want conflicts)
            tf.reset_default_graph()
            # Should be safe to read and write to the same dir since that's what resuming training does
            print('    Retraining {}'.format(name))
            train_personality(model_dir, model_dir, data_dir)
            # Reload model
            print('    Reloading {}'.format(name))
            bots[name] = Chat.load(name)

if __name__ == '__main__':
    jabber()