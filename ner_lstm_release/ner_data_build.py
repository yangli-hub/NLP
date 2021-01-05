# -*- coding: utf-8 -*-
# @Time    : 2017/1/12 11:46
import pickle
import numpy as np
import os

def capital(word):
    if ord(word[0]) >= ord('A') and ord(word[0]) <= ord('Z'):
        return 1
    else:
        return 2

def partofs(tag):
    onehot = 0
    if tag == 'NN' or tag == 'NNS':
        onehot = 1
    elif tag == 'FW':
        onehot = 2
    elif tag == 'NNP' or tag == 'NNPS':
        onehot = 3
    elif 'VB' in tag:
        onehot = 4
    else:
        onehot = 5

    return onehot


def chunk(tag):
    onehot = 0
    if 'NP' in tag:
        onehot = 1
    elif 'VP' in tag:
        onehot = 2
    elif 'PP' in tag:
        onehot = 3
    elif tag == 'O':
        onehot = 4
    else:
        onehot = 5

    return onehot

class DataBuild(object):
	def __init__(self):
		folder = 'ner'

		self.train=os.path.join(folder,'eng.train')
		self.valid=os.path.join(folder,'eng.testa')
		self.test=os.path.join(folder,'eng.testb')

		self.vocab_file='data/vocab_sample.bin'
		self.data_out='data/data_sample.bin'
		self.embedding_path = 'glove.6B.50d.txt'
		self.word_embed_size=50
		self.weight_path='data/word_embed_weight_sample.bin'
		#self.train_samples=100

	def gen_vocab_data1(self):
		file_list = [self.train, self.valid, self.test]
		word2id,pos2id1,pos2id2,cap2id,mlength=self.gen_vocab(file_list)
		data=self.gen_data(word2id,pos2id1,pos2id2,cap2id,mlength)
		vocab={'word2id':word2id,'pos2id1':pos2id1,'pos2id2':pos2id2,'cap2id':cap2id}
		with open(self.vocab_file,'wb') as fout:
			pickle.dump(vocab,fout)
		with open(self.data_out,'wb') as fout:
			pickle.dump(data,fout)
		if os.path.exists(self.embedding_path):
			w2v=self.gen_trained_word_embedding(word2id)
			V = len(word2id) + 1
			print('Vocab size:', V)

			# Not all words in word_to_idx are in w2v.
			# Word embeddings initialized to random Unif(-0.25, 0.25)
			embed = np.array(np.random.uniform(-0.25, 0.25, (V, len(w2v.values()[0]))), dtype=np.float32)
			embed[0] = 0
			for word, vec in w2v.items():
				embed[word2id[word]] = vec

			with open(self.weight_path,'wb') as fout:
				pickle.dump(embed,fout)
			print('weight genenrated')
		else:
			print('pretrained word embedding weight should exist at:' + self.embedding_path)
			V = len(word2id) + 1
			print('Vocab size:', V)

			# Not all words in word_to_idx are in w2v.
			# Word embeddings initialized to random Unif(-0.25, 0.25)
			embed = np.array(np.random.uniform(-0.25, 0.25, (V, self.word_embed_size)), dtype=np.float32)
			embed[0] = 0
			with open(self.weight_path,'wb') as fout:
				pickle.dump(embed,fout)

		print("data and vocab built")

	def gen_vocab_data(self):
		file_list = [self.train, self.valid, self.test]
		word2id,pos2id1,pos2id2,cap2id,mlength=self.gen_vocab(file_list)
		data=self.gen_data(word2id,pos2id1,pos2id2,cap2id,mlength)
		vocab={'word2id':word2id,'pos2id1':pos2id1,'pos2id2':pos2id2,'cap2id':cap2id}
		with open(self.vocab_file,'wb') as fout:
			pickle.dump(vocab,fout)
		with open(self.data_out,'wb') as fout:
			pickle.dump(data,fout)
		if os.path.exists(self.embedding_path):
			embedding_weight=self.gen_trained_word_embedding(word2id)
			with open(self.weight_path,'wb') as fout:
				pickle.dump(embedding_weight,fout)
			print('weight genenrated')
		else:
			print('pretrained word embedding weight should exist at:'+self.embedding_path)

		print("data and vocab built")

	def gen_vocab(self, file_list):
		max_sent_len = 0
		word_to_idx = {}
		pos_to_idx1 = {}
		pos_to_idx2 = {}
		cap_to_idx = {}

		# Starts at 1 for padding, position 0 is for padding
		idx = 1
		pos_idx1 = 1
		pos_idx2 = 1
		cap_idx = 1

		sentence_length = 0
		words = []
		pos1 = []
		pos2 = []
		cap = []

		for filename in file_list:
			f = open(filename, "r")
			for line in f:
				if line in ['\n', '\r\n']:
					max_sent_len = max(max_sent_len, len(words))
					for word in words:
						if not word in word_to_idx:
							word_to_idx[word] = idx
							idx += 1
					for pos in pos1:
						if not pos in pos_to_idx1:
							pos_to_idx1[pos] = pos_idx1
							pos_idx1 += 1
					for pos in pos2:
						if not pos in pos_to_idx2:
							pos_to_idx2[pos] = pos_idx2
							pos_idx2 += 1
					for c in cap:
						if not c in cap_to_idx:
							cap_to_idx[c] = cap_idx
							cap_idx += 1

					sentence_length = 0
					words = []
					pos1 = []
					pos2 = []
					cap = []
				else:
					assert (len(line.split()) == 4)
					sentence_length += 1
					temp = line.split()[0]
					temp_pos1 = partofs(line.split()[1])  # adding pos embeddings
					temp_pos2 = chunk(line.split()[2])  # adding chunk embeddings
					temp_cap = capital(temp)  # adding capital embedding
					words.append(temp)
					pos1.append(temp_pos1)
					pos2.append(temp_pos2)
					cap.append(temp_cap)
			f.close()
		return word_to_idx, pos_to_idx1, pos_to_idx2, cap_to_idx, max_sent_len


	def gen_data(self,word2id,pos2id1,pos2id2,cap2id,max_length):
		print("max_length:" + str(max_length))
		train,valid,test=dict(),dict(),dict()
		for infile,out in zip([self.valid,self.train,self.test],[valid,train,test]):
			sentence_length = 0
			sentence = []
			sentence_pos1 = []
			sentence_pos2 = []
			sentence_cap = []
			sentence_tag = []
			sentence_weight = []
			sent_length = []

			words = []
			pos1 = []
			pos2 = []
			cap = []
			tag = []
			weight = []
			for line in open(infile):
				if line in ['\n', '\r\n']:
					# print("aa"+str(sentence_length) )
					for _ in range(max_length - sentence_length):
						tag.append(0)
						temp = 0
						words.append(temp)
						pos1.append(temp)
						pos2.append(temp)
						cap.append(temp)
						weight.append(0)
					sentence.append(words)
					sentence_pos1.append(pos1)
					sentence_pos2.append(pos2)
					sentence_cap.append(cap)
					sentence_tag.append(tag)
					sentence_weight.append(weight)
					sent_length.append(sentence_length)

					sentence_length = 0
					words = []
					pos1 = []
					pos2 = []
					cap = []
					tag = []
					weight = []
				else:
					assert (len(line.split()) == 4)
					if sentence_length > max_length:
						continue
					sentence_length += 1
					temp = word2id[line.split()[0]]
					temp_pos1 = pos2id1[partofs(line.split()[1])]  # adding pos embeddings
					temp_pos2 = pos2id2[chunk(line.split()[2])]  # adding chunk embeddings
					temp_cap = cap2id[capital(line.split()[0])]  # adding capital embedding
					words.append(temp)
					pos1.append(temp_pos1)
					pos2.append(temp_pos2)
					cap.append(temp_cap)
					weight.append(1)

					t = line.split()[3]

					# Five classes 0-None,1-Person,2-Location,3-Organisation,4-Misc

					if t.endswith('O'):
						tag.append(0)
					elif t.endswith('PER'):
						tag.append(1)
					elif t.endswith('LOC'):
						tag.append(2)
					elif t.endswith('ORG'):
						tag.append(3)
					elif t.endswith('MISC'):
						tag.append(4)
					else:
						print("error in input" + str(t))

			out['input']=np.asarray(sentence)
			out['target']=np.asarray(sentence_tag)
			out['pos1']=np.asarray(sentence_pos1)
			out['pos2'] = np.asarray(sentence_pos2)
			out['cap']=np.asarray(sentence_cap)
			out['weight']=np.asarray(sentence_weight)
			out['length'] = sent_length

		data=dict()
		data['train']=train
		data['valid']=valid
		data['test']=test


		#total,unk=self.unk_statistic(gtest)
		#print('test total unk num:')
		#print(total,unk)
		return data

	def gen_trained_word_embedding1(self,word2id):
		embeddings_index = {}
		fname = self.embedding_path
		word_vecs = {}
		with open(fname, "rb") as f:
			header = f.readline()
			vocab_size, layer1_size = map(int, header.split())
			binary_len = np.dtype('float32').itemsize * layer1_size
			for line in range(vocab_size):
				word = []
				while True:
					ch = f.read(1)
					if ch == ' ':
						word = ''.join(word)
						break
					if ch != '\n':
						word.append(ch)
				if word in word2id:
					word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
				else:
					f.read(binary_len)
		return word_vecs

	def gen_trained_word_embedding(self,word2id):
		embeddings_index = {}
		f = open(self.embedding_path, 'r', encoding='UTF-8')
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
		f.close()
		embedding_matrix = np.random.uniform(0, 0.05, (len(word2id)+1, self.word_embed_size))
		embedding_matrix[0] = 0
		# embedding_matrix = np.zeros((len(self.word2id), self.word_embed_size))
		vocab_size=len(word2id)
		pretrained_size=0
		for word, i in word2id.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				# words not found in embedding index will be all-zeros.
				pretrained_size+=1
				#embedding_matrix[i+1] = embedding_vector # this place has some problems.
				embedding_matrix[i] = embedding_vector  # this place has some problems.

		print('vocab size:%d\t pretrained size:%d' %(vocab_size,pretrained_size))

		return embedding_matrix



import sys
if __name__=='__main__':

	db=DataBuild()
	db.gen_vocab_data()
