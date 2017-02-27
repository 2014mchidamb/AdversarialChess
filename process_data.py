import chess
import chess.pgn
import cPickle as pickle
import h5py
import numpy as np
import random

# Piece to index mapping
piece_to_ind = {}
cur_ind = 0
for color in [True, False]:
	for piece_num in range(6):
		piece_to_ind[(color, piece_num+1)] = cur_ind
		cur_ind += 1

# Outcome to value mapping
outcome_to_val = {}
outcome_to_val['1-0'] = 1.0
outcome_to_val['0-1'] = -1.0
outcome_to_val['1/2-1/2'] = 0.0

# Converts board to 8x8x12 array
def convert_board(board):
	# Initialize tensor corresponding to board
	b_tensor = np.zeros((8, 8, 12))

	# Iterate over board squares 
	for i in range(64):
		piece = board.piece_at(i)
		if not piece:
			continue
		ind = piece_to_ind[(piece.color, piece.piece_type)]
		# One-hot encode piece
		b_tensor[i/8, i%8, ind] = 1

	return b_tensor

# Return a random next board
def get_random_next(board):
	moves = list(board.legal_moves)
	board.push(random.choice(moves))
	return board

# Generates training data based on single board transitions
def gen_board_pair_data(infile, outfile):
	# Game data
	pgn = open(infile)
	cur_game = chess.pgn.read_game(pgn)
	
	# Output
	out = h5py.File(outfile+'.hdf5', 'w')
	f_boards, s_boards, r_boards = [
		out.create_dataset(dname, (0, 8, 8, 12), dtype='b',
							maxshape=(None, 8, 8, 12),
							chunks=True)
		for dname in ['f_boards', 's_boards', 'r_boards']]
	playing, results, move_props = [
		out.create_dataset(dname, (0,), dtype='b',
							maxshape=(None,),
							chunks=True)
		for dname in ['playing', 'results', 'move_props']]

	# Loop through games 
	line_num = 0
	size = 0
	game_num = 0
	while cur_game:
		node = cur_game
		move_total = 0
		outcome = outcome_to_val[cur_game.headers['Result']]
		to_play = 1
		# Loop through boards
		while not node.is_end():
			# Check if datasets need to be resized
			if line_num+1 >= size:
				out.flush()
				size = 2*size+1
				print 'Resizing to '+str(size)
				[d.resize(size=size, axis=0) for d in
					[f_boards, s_boards, r_boards, playing, results, move_props]]

			move_total += 1
			next_node = node.variation(0)
			# First position, second position, side playing,
			# outcome of game, move proportion
			f_boards[line_num] = convert_board(node.board())
			s_boards[line_num] = convert_board(next_node.board())
			r_boards[line_num] = convert_board(get_random_next(node.board()))
			playing[line_num] = to_play
			results[line_num] = outcome	
			to_play = -1*to_play
			node = next_node
			line_num += 1
		for move in range(1, move_total+1):
			move_props[line_num-move_total-1+move] = move/float(move_total)
		cur_game = chess.pgn.read_game(pgn)
		game_num += 1

	# Finish storing collected data
	[d.resize(size=line_num, axis=0) for d in
		[f_boards, s_boards, r_boards, playing, results, move_props]]
	out.close()

def gen_player_data(infile, outfile, player_name):
	# Game data
	pgn = open(infile)
	cur_game = chess.pgn.read_game(pgn)
	
	# Output
	out = h5py.File(outfile+'.hdf5', 'w')
	f_boards, s_boards = [
		out.create_dataset(dname, (0, 8, 8, 12), dtype='b',
							maxshape=(None, 8, 8, 12),
							chunks=True)
		for dname in ['f_boards', 's_boards']]
	p_color = [
		out.create_dataset(dname, (0,), dtype='b',
							maxshape=(None,),
							chunks=True)
		for dname in ['p_color']][0]
	full_boards = []	

	# Loop through games 
	line_num = 0
	size = 0
	game_num = 0
	while cur_game:
		node = cur_game
		move_total = 0
		to_play = 1
		player = -1
		if player_name in cur_game.headers['White']:
			player = 1
		# Loop through boards
		while not node.is_end():
			# Check if datasets need to be resized
			if line_num+1 >= size:
				out.flush()
				size = 2*size+1
				print 'Resizing to '+str(size)
				[d.resize(size=size, axis=0) for d in
					[f_boards, s_boards, p_color]]

			next_node = node.variation(0)
			# First position, second position
			if to_play == player:
				full_boards.append(node.board())
				f_boards[line_num] = convert_board(node.board())
				s_boards[line_num] = convert_board(next_node.board())
				p_color[line_num] = player
				line_num += 1
			to_play = -1*to_play
			node = next_node
		cur_game = chess.pgn.read_game(pgn)
		game_num += 1

	# Finish storing collected data
	[d.resize(size=line_num, axis=0) for d in
		[f_boards, s_boards, p_color]]
	out.close()
	pickle.dump(full_boards, open("../Data/full_boards.pkl", "wb"))
	

def main():
	datafile = '../Data/ficsgamesdb_2015_standard2000_nomovetimes_1429742.pgn'
	playerfile = '../Data/Tal.pgn'
	gen_board_pair_data(datafile, '../Data/training_data')
	gen_player_data(playerfile, '../Data/player_data', 'Tal')

if __name__ == '__main__':
	main()		
