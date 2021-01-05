# -*- coding: utf-8 -*-
# @Time    : 2017/1/12 15:25

from ner_data_utils import Data
import argparse
import sys
import os
from ner_bilstm import BiLstm

def main(arguments):
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-ut', '--use_tree', help="use tree feature or not  ", action='store_true',default=True)
	parser.add_argument('-lr', '--learn_rate', help=" ", type=float,default=0.001)
	parser.add_argument('-ep', '--epochs',type=int,help="training epochs  ",default=5)
	parser.add_argument('-hs', '--hidden_size', help="hidden layer size", default=100)
	parser.add_argument('-ln', '--num_layers', help="stack lstm number", default=1)
	parser.add_argument('-wes', '--word_embed_size', help="word vect size", default=50)
	parser.add_argument('-bs', '--batch_size', help=" ", default=60)
	parser.add_argument('-mn', '--model_name', help="model saved path",default='model')
	parser.add_argument('-md', '--mode', help="train or test",default='train')
	args = parser.parse_args(arguments)
	data = Data('data/data_sample.bin','data/vocab_sample.bin','data/word_embed_weight_sample.bin',args.batch_size)
	if not os.path.exists('./output/'):
		os.makedirs('./output/')
	model=BiLstm(args,data,ckpt_path='./output/')
	if args.mode=='train':
		model.train(data)
		sess = model.restore_last_session()
		model.predict(data, sess)
	if args.mode=='test':
		sess = model.restore_last_session()
		model.predict(data, sess)

if __name__ == '__main__':
	#CUDA_VISIBLE_DEVICES = ""
	main(sys.argv[1:])








