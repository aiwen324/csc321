"""Script to visualize attention maps using a pre-trained model.

    Usage: python visualize_attention.py --load=checkpoints/h10-bs16
"""

import os
import pdb
import sys
import argparse
import pickle as pkl

import numpy as np

import torch

# Local imports
import utils


# words = ['roomba',
#          'a',
#          'aj',
#          'ij',
#          'sh',
#          'ch',
#          'gb',
#          'chr',
#          'chrf',
#          'gdb',
#          'car',
#          'ajj',
#          'dash',
#          'five',
#          'apple',
#          'akkk',
#          'ijjj',
#          'ajjj',
#          'abbbb',
#          'aeeee',
#          'azzzz',
#          'ajjjj',
#          'sugar',
#          'aiiii',
#          'akkkk',
#          'afdsj',
#          'akkkkk',
#          'active',
#          'doctor',
#          'slash',
#          'friday',
#          'adjoint',
#          'scinece',
#          'computer',
#          'impulsive',
#          'bijective',
#          'injective',
#          'aijective',
#          'aggresive',
#          'offensive',
#          'christmas',
#          'abilities',
#          'adjective',
#          'ajjjjjive',
#          'ajjjjjjjj',
#          'understand',
#          'attachable',
#          'comprehension',
#          'comprehensive',
#          'abcdefghijkl',
#          'abcdefghijklm',
#          'bcdefghijklmn',
#          'abcedfghijklmnopqrst',
#          'bcedfghijklmnopqrst',
#          'incomprehensibilities',
#          'comprehensibilities',
#          # Add your own words here!
#         ]
words = ['well-defined',
         'a-b',
         'b-c',
         'ef-gh',
         'ba-ce',
         'well-mannered',
         'ten-year-old',
         'a-writeup',
         'sugar-free',
         'power-driven',
         'user-generated',
        ]



def load(opts):
    encoder = torch.load(os.path.join(opts.load, 'encoder.pt'))
    decoder = torch.load(os.path.join(opts.load, 'decoder.pt'))
    idx_dict = pkl.load(open(os.path.join(opts.load, 'idx_dict.pkl'), 'rb'))
    return encoder, decoder, idx_dict


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, help='Path to checkpoint directory.')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use GPU.')
    return parser


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    encoder, decoder, idx_dict = load(opts)

    for word in words:
        translated = utils.translate(word,
                                     encoder,
                                     decoder,
                                     idx_dict,
                                     opts)

        print('{} --> {}'.format(word, translated))

        utils.visualize_attention(word,
                                  encoder,
                                  decoder,
                                  idx_dict,
                                  opts,
                                  save=os.path.join(opts.load, '{}.pdf'.format(word)))
