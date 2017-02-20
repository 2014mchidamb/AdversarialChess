from game_utils import negamax
from model import Magikarp
import chess
import numpy as np
import tensorflow as tf

config = {}
config['batch_size'] = 20
config['datafile'] = '../Data/training_data.hdf5'
config['p_datafile'] = '../Data/tal_data.hdf5'
config['full_boards_file'] = '../Data/full_boards.pkl'
config['num_epochs'] = 1
config['save_file'] = 'trained_model/trained_genadv.ckpt'

with tf.Session() as sess:
	# Set up chess board
	board = chess.Board()

	# Load evaluation model
	magikarp = Magikarp(config, sess)
	magikarp.load_model(magikarp.save_file)	

	# Begin chess game
	while not board.is_checkmate():
		# Human plays as white for simplicity
		print '-'*50
		print "Current Board:\n\n", board, "\n"
		move = "a1a1"
		while True:
			raw_move = raw_input("Please enter a move in UCI notation: ")
			move = chess.Move.from_uci(raw_move)
			if move in board.legal_moves:
				board.push(move)
				break
			else:
				print "Please enter a valid move."
		# Computer response
		score, comp_move = negamax(board, 0, -1, float('-inf'), float('inf'), magikarp)
		print score, comp_move
		board.push(comp_move)
