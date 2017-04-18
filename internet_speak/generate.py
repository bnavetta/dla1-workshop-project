import argparse
import os

import tensorflow as tf
import numpy as np
from six.moves import cPickle

from .model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/media/data/save',
                        help='model directory to load stored checkpointed models from')
    parser.add_argument('--output_file', type=str, default='generated.txt',
                        help='file to save generated text in')
    args = parser.parse_args()
    generate(args)


def generate(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        checkpoint = tf.train.get_checkpoint_state(args.save_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            with open(args.output_file, 'w') as f:
                for _ in range(100):
                    length = np.random.normal(12, 7)
                    phrase = model.sample(sess, words, vocab, length, ' ', 1, 2)
                    f.write(phrase + '\n')

if __name__ == '__main__':
    main()