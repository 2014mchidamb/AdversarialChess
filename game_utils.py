from process_data import convert_board
import numpy as np
import tensorflow as tf

# Search at depth of one move
MAX_DEPTH = 0

def negamax(board, depth, color, alpha, beta, evaluator):
	if board.is_checkmate() or depth > MAX_DEPTH:
		input_board = convert_board(board).flatten().reshape((1,-1))
		return (color*evaluator.get_prediction(input_board), None)
	maxval = float('-inf')
 	best_move = None
 	for move in board.legal_moves:
		board.push(move)
		val = -1*negamax(board, depth+1, -1*color, -1*beta, -1*alpha, evaluator)[0]
		board.pop()
		print val, move
		if val > maxval:
			maxval = val
			best_move = move
		if val > alpha:
			alpha = val
		if alpha >= beta:
			return (alpha, best_move)
	return (maxval, best_move)
