from model import Magikarp
import numpy as np
import tensorflow as tf

config = {}
config['batch_size'] = 64
config['datafile'] = '../Data/training_data.hdf5'
config['p_datafile'] = '../Data/tal_data.hdf5'
config['full_boards_file'] = '../Data/full_boards.pkl'
config['num_epochs'] = 10
config['save_file'] = 'trained_model/trained_genadv.ckpt'

with tf.Session() as sess:
	magikarp = Magikarp(config, sess)
	magikarp.train()
