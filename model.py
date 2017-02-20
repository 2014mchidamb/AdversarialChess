import chess
import cPickle as pickle
import h5py
import numpy as np
from process_data import convert_board
import tensorflow as tf

class Magikarp(object):
	def __init__(self, config, sess):
		self.sess = sess
		self.batch_size = config['batch_size']
		self.cur_ind = 0
		self.data = h5py.File(config['datafile'], 'r')
		self.data_size = len(self.data['f_boards'])
		self.f_boards_full = pickle.load(open(config['full_boards_file'], 'rb'))
		self.l_rate = 0.00005
		self.p_cur_ind = 0
		self.p_data = h5py.File(config['p_datafile'], 'r')
		self.p_data_size = len(self.p_data['f_boards'])
		self.n_input = 768
		self.n_hidden1 = 2048
		self.n_hidden2 = 2048
		#self.hidden_layers = config['hidden_layers']
		self.n_out = 1
		self.num_epochs = config['num_epochs']
		self.reg_coeff = 4
		self.save_file = config['save_file']

	def rand_weights(self, n_in, n_out):
		return tf.random_uniform([n_in, n_out], -1*np.sqrt(6.0/(n_in + n_out)), np.sqrt(6.0/(n_in + n_out)))
	
	def get_gen_params(self):
		self.g_weights = {
			'h1': tf.Variable(self.rand_weights(self.n_input, self.n_hidden1), name='g_h1'),
			'h2': tf.Variable(self.rand_weights(self.n_hidden1, self.n_hidden2), name='g_h2'),
			'out': tf.Variable(self.rand_weights(self.n_hidden2, self.n_out), name='g_out')
		}
		self.g_biases = {
			'b1': tf.Variable(tf.random_normal([self.n_hidden1], stddev=0.01), name='g_b1'),
			'b2': tf.Variable(tf.random_normal([self.n_hidden2], stddev=0.01), name='g_b2'),
			'out': tf.Variable(tf.random_normal([self.n_out], stddev=0.01), name='g_b_out')
		}

	def get_dis_params(self):
		self.d_weights = {
			'h1': tf.Variable(self.rand_weights(self.n_input*2, self.n_hidden1), name='d_h1'),
			'h2': tf.Variable(self.rand_weights(self.n_hidden1, self.n_hidden2), name='d_h2'),
			'out': tf.Variable(self.rand_weights(self.n_hidden2, self.n_out), name='d_out')
		}
		self.d_biases = {
			'b1': tf.Variable(tf.random_normal([self.n_hidden1], stddev=0.01), name='d_b1'),
			'b2': tf.Variable(tf.random_normal([self.n_hidden2], stddev=0.01), name='d_b2'),
			'out': tf.Variable(tf.random_normal([self.n_out], stddev=0.01), name='d_b_out')
		}

	def gen_move(self, input_board, color):
		best_move = None
		maxval = float('-inf')		
		for move in input_board.legal_moves:
			input_board.push(move)
			val = color*self.get_prediction(convert_board(input_board).flatten().reshape((1, -1)))
			input_board.pop()
			if val > maxval:
				maxval = val
				best_move = move
		input_board.push(best_move)
		res = convert_board(input_board)
		input_board.pop()
		return res					
		
	def g_predict(self, input_board, p_keep):
		hidden1 = tf.add(tf.matmul(input_board, self.g_weights['h1']), self.g_biases['b1'])
		hidden1 = tf.nn.relu(hidden1) #tf.maximum(0.01*hidden1, hidden1) #tf.nn.relu(hidden1)
		#hidden1 = tf.nn.dropout(hidden1, p_keep)

		hidden2 = tf.add(tf.matmul(hidden1, self.g_weights['h2']), self.g_biases['b2'])
		hidden2 = tf.nn.relu(hidden2) #tf.maximum(0.01*hidden2, hidden2) #tf.nn.relu(hidden2)
		#hidden2 = tf.nn.dropout(hidden2, p_keep)
		
		return tf.add(tf.matmul(hidden2, self.g_weights['out']), self.g_biases['out'])
	
	def d_predict(self, input_board, p_keep):
		hidden1 = tf.add(tf.matmul(input_board, self.d_weights['h1']), self.d_biases['b1'])
		hidden1 = tf.nn.relu(hidden1) #tf.maximum(0.01*hidden1, hidden1) #tf.nn.relu(hidden1)
		#hidden1 = tf.nn.dropout(hidden1, p_keep)

		hidden2 = tf.add(tf.matmul(hidden1, self.d_weights['h2']), self.d_biases['b2'])
		hidden2 = tf.nn.relu(hidden2) #tf.maximum(0.01*hidden2, hidden2) #tf.nn.relu(hidden2)
		#hidden2 = tf.nn.dropout(hidden2, p_keep)
		
		return tf.sigmoid(tf.add(tf.matmul(hidden2, self.d_weights['out']), self.d_biases['out']))

	def set_optimization(self):
		# Get params to update
		self.params = tf.trainable_variables()
		self.g_params = [p for p in self.params if p.name.startswith('g')]
		self.d_params = [p for p in self.params if p.name.startswith('d')]
		
		''' Generator '''
		# Compute f(first board) + f(second board)
		self.pred_sum = self.f_pred - self.s_pred #- tf.multiply(self.results, self.move_props)

		# Compute -log(sigmoid(f(second board) - f(random board)))
		self.rand_diff = -1*tf.reduce_mean(tf.log(tf.sigmoid(tf.multiply((self.s_pred - self.r_pred), self.playing))))

		# Compute -log(sigmoid(sum of boards)) and -log(sigmoid(- sum of boards))
		self.equal_board1 = -1*tf.reduce_mean(tf.log(tf.sigmoid(self.pred_sum)))
		self.equal_board2 = -1*tf.reduce_mean(tf.log(tf.sigmoid(-1*self.pred_sum)))

		# Use discriminator as regularizer
		self.regularizer = -1*tf.reduce_mean(self.d_pred_fake)
		
		# Set up total cost and optimization
		self.g_cost = self.rand_diff + self.equal_board1 + self.equal_board2 + self.reg_coeff*self.regularizer
		self.g_optimizer = tf.train.AdamOptimizer(learning_rate=self.l_rate)
		self.g_gvs = self.g_optimizer.compute_gradients(self.g_cost, self.g_params)
		self.g_capped_gvs = self.g_gvs #[(tf.clip_by_value(grad, -1, 1), var) for grad, var in self.gvs]
		self.g_train_op = self.g_optimizer.apply_gradients(self.g_capped_gvs)

		''' Discriminator '''
		# Set up total cost and optimization
		# Wasserstein Loss
		self.d_cost = -1*tf.reduce_mean(self.d_pred_real - self.d_pred_fake)
		self.d_optimizer = tf.train.AdamOptimizer(learning_rate=self.l_rate)
		self.d_gvs = self.d_optimizer.compute_gradients(self.d_cost, self.d_params)
		self.d_capped_gvs = self.d_gvs
		self.d_train_op = self.d_optimizer.apply_gradients(self.d_capped_gvs)	

	def gen_g_batch(self):
		f_boards = []
		s_boards = []
		r_boards = []
		results = []
		playing = []
		move_props = []
		for i in range(self.batch_size):
			f_boards.append(self.data['f_boards'][self.cur_ind].flatten())
			s_boards.append(self.data['s_boards'][self.cur_ind].flatten())
			r_boards.append(self.data['r_boards'][self.cur_ind].flatten())
			results.append(self.data['results'][self.cur_ind].flatten())
			playing.append(self.data['playing'][self.cur_ind].flatten())
			move_props.append(self.data['move_props'][self.cur_ind].flatten())
			self.cur_ind = (self.cur_ind+1) % self.data_size
			
		return f_boards, s_boards, r_boards, results, playing, move_props

	def gen_d_batch(self):
		p_f_boards = []
		p_s_boards = []
		gen_boards = []
		for i in range(self.batch_size):
			p_f_boards.append(self.p_data['f_boards'][self.p_cur_ind].flatten())
			p_s_boards.append(self.p_data['s_boards'][self.p_cur_ind].flatten())
			gen_boards.append(self.gen_move(self.f_boards_full[self.p_cur_ind], self.p_data['p_color'][self.p_cur_ind]).flatten())
			self.p_cur_ind = (self.p_cur_ind+1) % self.p_data_size

		return p_f_boards, p_s_boards, gen_boards
			
	def create_gen_model(self):
		# Set up model parameters
		self.get_gen_params()		

		# Set up graph inputs
		self.f_board_input = tf.placeholder(tf.float32, [None, self.n_input])
		self.s_board_input = tf.placeholder(tf.float32, [None, self.n_input])
		self.r_board_input = tf.placeholder(tf.float32, [None, self.n_input])
		self.results = tf.placeholder(tf.float32, [None, 1])
		self.playing = tf.placeholder(tf.float32, [None, 1])
		self.move_props = tf.placeholder(tf.float32, [None, 1])
		self.p_keep = tf.placeholder(tf.float32)

		# Get graph outputs
		self.f_pred = self.g_predict(self.f_board_input, self.p_keep)
		self.s_pred = self.g_predict(self.s_board_input, self.p_keep)
		self.r_pred = self.g_predict(self.r_board_input, self.p_keep)

	def create_dis_model(self):
		# Set up discriminator model parameters
		self.get_dis_params()

		# Set up discriminator graph inputs
		self.person_board_1 = tf.placeholder(tf.float32, [None, self.n_input])
		self.person_board_2 = tf.placeholder(tf.float32, [None, self.n_input])
		self.gen_board = tf.placeholder(tf.float32, [None, self.n_input])		

		# Get discriminator outputs
		self.d_pred_real = self.d_predict(tf.concat(1, [self.person_board_1, self.person_board_2]), self.p_keep)
		self.d_pred_fake = self.d_predict(tf.concat(1, [self.person_board_1, self.gen_board]), self.p_keep)

		# Clamp weights
		self.weight_clamps = [tf.clip_by_value(self.d_weights[layer], -0.01, 0.01) for layer in self.d_weights]
		self.bias_clamps = [tf.clip_by_value(self.d_biases[layer], -0.01, 0.01) for layer in self.d_biases]

	def create_model(self):
		# Create both networks
		self.create_gen_model()
		self.create_dis_model()

		# Get loss and optimize
		self.set_optimization()
		
		# Initialize all variables
		self.init = tf.global_variables_initializer()

		# Model saver
		self.saver = tf.train.Saver()

		# Run initializer
		self.sess.run(self.init)

	def get_prediction(self, board):
		return self.f_pred.eval({self.f_board_input: board, self.p_keep: 1.0})
		
	def train(self):
		self.create_model()
		print self.g_params
		print self.d_params
		for epoch in range(self.num_epochs):
			num_batches = 100 #self.data_size/self.batch_size
			g_avg_cost = 0
			d_avg_cost = 0
			p_f_boards, p_s_boards, gen_boards = [], [], []
			for batch in range(num_batches):
				for i in range(5):
					p_f_boards, p_s_boards, gen_boards = self.gen_d_batch()
					_, _, _, dc = self.sess.run([self.weight_clamps, self.bias_clamps, self.d_train_op, self.d_cost], feed_dict = {
								self.person_board_1: p_f_boards, self.person_board_2: p_s_boards,
								self.gen_board: gen_boards})
					d_avg_cost += dc/float(num_batches*5)
				f_boards, s_boards, r_boards, results, playing, move_props = self.gen_g_batch()
				_, gc = self.sess.run([self.g_train_op, self.g_cost], feed_dict = {
								self.f_board_input: f_boards, self.s_board_input: s_boards,
								self.r_board_input: r_boards, self.p_keep: 0.5,
								self.results: results, self.move_props: move_props,
								self.playing: playing, self.person_board_1: p_f_boards,
								self.person_board_2: p_s_boards, self.gen_board: gen_boards})
				g_avg_cost += gc/float(num_batches)
				#print c
				#print self.sess.run(self.weights['h1'])
			print "Epoch ", (epoch+1), ": Average generator cost was ", g_avg_cost, "\tAverage discriminator cost was ", d_avg_cost
			save_path = self.saver.save(self.sess, self.save_file)
		print "Optimization complete."
		save_path = self.saver.save(self.sess, self.save_file)
		print "Model saved as "+self.save_file

	def load_model(self, model_file):
		self.create_model()
		self.saver.restore(self.sess, model_file)
		print "Model restored from "+model_file

