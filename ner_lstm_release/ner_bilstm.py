import tensorflow as tf
import numpy as np
import sys
import time
from ner_util import Util
from ner_evaluation import Evaluate
import tensorflow.contrib.rnn as RNNCell
from metrics.accuracy import conlleval

label_dict = {0:'O', 1:'B-PER', 2:'B-LOC', 3:'B-ORG', 4:'B-MISC'}


class BiLstm(object):

	def __init__(self,args,data,ckpt_path): #seq_len,xvocab_size, label_size,ckpt_path,pos_size,type_size,data
		self.opt = args
		self.num_steps = 124
		self.num_class = 5
		self.num_chars = data.word_size
		self.ckpt_path=ckpt_path
		self.pos_size=data.pos_size1
		self.chunk_size = data.pos_size2
		self.cap_size=data.cap_size
		self.util= Util()
		sys.stdout.write('Building Graph ')
		self._build_model(args,embedding_matrix=data.pretrained)
		sys.stdout.write('graph built\n')
		self.eval=Evaluate()

	def _build_model(self,flags,embedding_matrix):
		tf.reset_default_graph()
		tf.set_random_seed(123)
		self.input=tf.placeholder(shape=[None,self.num_steps], dtype=tf.int64)
		self.length = tf.placeholder(shape=[None,], dtype=tf.int64)
		self.pos=tf.placeholder(shape=[None,self.num_steps], dtype=tf.int64)
		self.type=tf.placeholder(shape=[None,self.num_steps], dtype=tf.int64)
		self.cap = tf.placeholder(shape=[None, self.num_steps], dtype=tf.int64)
		self.target = [tf.placeholder(shape=[None, ], dtype=tf.int64, name='li_{}'.format(t)) for t in   range(self.num_steps)]
		self.weight = [tf.placeholder(shape=[None, ], dtype=tf.float32, name='wi_{}'.format(t)) for t in    range(self.num_steps)]
		self.keep_prob = tf.placeholder(tf.float32)  # drop out

		if embedding_matrix is not None:
			self.embedding = tf.Variable(embedding_matrix, trainable=True, name="emb",dtype=tf.float32)#
		else:
			self.embedding = tf.get_variable("emb", [self.num_chars, self.emb_dim])
		self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.input)
		if flags.use_tree:
			pos_embedding = tf.get_variable('pos_embed', [self.pos_size, 50], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1e-4))
			type_embedding = tf.get_variable('type_embed', [self.chunk_size, 50], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1e-4))
			cap_embedding = tf.get_variable('cap_embed', [self.cap_size, 10], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1e-4))
			pos_inputs = tf.nn.embedding_lookup(pos_embedding, self.pos)
			type_inputs = tf.nn.embedding_lookup(type_embedding, self.type)
			cap_inputs = tf.nn.embedding_lookup(cap_embedding, self.cap)
			#self.inputs_emb = tf.concat([self.inputs_emb, pos_inputs,type_inputs], 2)  # self.inputs_emb,,type_inputs# do not use previous label currently
			#self.inputs_emb = tf.concat([self.inputs_emb, pos_inputs, type_inputs], 2)
			self.inputs_emb = tf.concat([self.inputs_emb, pos_inputs, type_inputs, cap_inputs], 2)

		cell = RNNCell.LSTMCell(num_units=flags.hidden_size, state_is_tuple=True)
		#cell = RNNCell.LSTMCell(num_units=flags.hidden_size, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
		dropout_cell = RNNCell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
		stacked_cell= RNNCell.MultiRNNCell([dropout_cell] * self.opt.num_layers, state_is_tuple=True)
		#stacked_cell = RNNCell.MultiRNNCell([dropout_cell for _ in range(self.opt.num_layers)], state_is_tuple=True)
		outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked_cell,cell_bw=stacked_cell,dtype=tf.float32,sequence_length=self.length,inputs=self.inputs_emb)
		output_fw, output_bw = outputs
		output=	tf.concat([output_fw,output_bw], 2)
		soft_dim=self.opt.hidden_size*2
		self.softmax_w = tf.get_variable("softmax_w", [soft_dim, self.num_class])
		self.softmax_b = tf.get_variable("softmax_b", [self.num_class])
		output=tf.reshape(output,[-1,soft_dim])
		self.logits = tf.matmul(output, self.softmax_w) + self.softmax_b

		self.decode_outputs_test = tf.nn.softmax(self.logits)
		self.decode_outputs_test=tf.reshape(self.decode_outputs_test,[-1,self.num_steps,self.num_class])

		#states_fw, states_bw = states
		self.classify_out=tf.reshape(self.logits,[-1,self.num_steps,self.num_class])
		self.logits= tf.transpose(self.classify_out, [1, 0, 2])
		self.logits=tf.unstack(self.logits,axis=0)
		self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.logits, self.target, self.weight, self.num_class)
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.opt.learn_rate).minimize(self.loss)

	def f1_conll(self, prediction, target, weight):  # not tensors but result values
		# print('pred: ' + str(prediction[1]))
		# print('true: ' + str(target[1]))
		#target = np.reshape(target, [-1, self.num_steps, self.num_class])
		#print(target.shape)
		#prediction = np.reshape(prediction, [-1, self.num_steps, self.num_class])

		pred = []
		true = []
		doc_word = []
		for i in range(len(target)):
			sent_true = []
			sent_pred = []
			sent_word = []
			for j in range(self.num_steps):
				if weight[i][j] == 0:
					break
				else:
					sent_true.append(label_dict[target[i][j]])
					sent_pred.append(label_dict[prediction[i][j]])
					sent_word.append('ruru')
				pred.append(sent_pred)
				true.append(sent_true)
				doc_word.append(sent_word)

		res_test = conlleval(pred, true, doc_word, 'current.test.txt')
		print(res_test)

		return res_test

	'''Training and Evaluation'''
	def train(self, data, sess=None):
		saver = tf.train.Saver()
		if not sess:
			sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1)) # create a session
			#sess = tf.Session(tf.ConfigProto(device_count = {'GPU': 0}))  # create a session
			sess.run(tf.global_variables_initializer()) 		# init all variables
		sys.stdout.write('\n Training started ...\n')
		best_loss=100
		best_epoch=0
		t1=time.time()
		for i in range(self.opt.epochs):
			try:
				loss,_=self.run_epoch(sess,data,data.train,True)
				val_loss,pred= self.run_epoch(sess, data,data.valid,False)
				t2=time.time()
				print('epoch:%2d \t time:%.2f\tloss:%f\tvalid_loss:%f'%(i,t2-t1,loss,val_loss))
				self.predict(data, sess)
				t1=time.time()
				if val_loss<best_loss:
					saver.save(sess, self.ckpt_path + self.opt.model_name + '.ckpt')
					best_loss=val_loss
					best_epoch=i
				sys.stdout.flush()
			except KeyboardInterrupt:  # this will most definitely happen, so handle it
				print('Interrupted by user at iteration {}'.format(i))
				self.session = sess
				return sess
		print('best valid accuary:%f\tbest epoch:%d'%(best_loss,best_epoch))

	# prediction
	def predict(self, data, sess):
		_, predicts = self.run_epoch(sess, data, data.test, False)
		pred= np.argmax(predicts, axis=2)

		#acc, f1, pratio, gratio=self.eval.values(pred,data.test['target'],data.test['weight'])
		test_f1 = self.f1_conll(pred, data.test['target'],data.test['weight'])
		print('f1:%f,precision:%f,recall:%f' % (test_f1['f1'], test_f1['p'], test_f1['r']))

	def run_epoch(self, sess, data,data_type,is_train):
		losses = []
		num_batch=data.gen_batch_num(data_type)
		predicts=None
		for i in range(num_batch):
			input, target, weight, length, pos1, pos2, cap=data.gen_batch(data_type, i)
			#print(input)
			if is_train:
				feed_dict = self.get_feed(input, target, weight, length, pos1, pos2, cap, keep_prob=0.5)
				_, loss_v, predict = sess.run([self.train_op, self.loss, self.decode_outputs_test], feed_dict)
			else:
				feed_dict = self.get_feed(input, target, weight, length, pos1, pos2, cap, keep_prob=1.)
				loss_v, predict= sess.run([self.loss, self.decode_outputs_test], feed_dict)
			losses.append(loss_v)
			if predicts is None:
				predicts = predict
			else:
				predicts = np.concatenate((predicts, predict))
		return np.mean(losses),predicts

	def restore_last_session(self):
		saver = tf.train.Saver()
		sess = tf.Session()  # create a session
		saver.restore(sess, self.ckpt_path + self.opt.model_name + '.ckpt')
		print('model restored')
		return sess

	def get_feed(self, input, target, weight, length, pos, dtype, cap, keep_prob):
		feed_dict={self.input:input}
		feed_dict.update({self.target[t]: target[t] for t in range(self.num_steps)})
		feed_dict.update({self.weight[t]: weight[t] for t in range(self.num_steps)})
		feed_dict[self.pos]=pos
		feed_dict[self.cap] = cap
		feed_dict[self.type]=dtype  #chunk feature
		feed_dict[self.length]=length
		feed_dict[self.keep_prob] = keep_prob  # dropout prob
		return feed_dict















