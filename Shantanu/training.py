#!/usr/bin/env python
# -*- coding: utf-8 -*-

from features import all_features
from sklearn.naive_bayes import GaussianNB
from splitter import trim_char
import cv2
import unicodecsv as csv
import time
import numpy as np
import codecs

from multiprocessing import Pool

NUM_KANJI = 2289

NUM_THREADS = 8

kernel = np.ones((5,5),np.uint8)

fonts = ['Mincho']#['Gothic', 'Lantinghei', 'Meiryo', 'Mincho', 'Osaka', 'STFangSong',
		#'GenEiExtraLight', 'GenEiHeavy', 'GenEiSemiBold', 'HonyaJi', 'MPlusBold',
		#'MPlusRegular', 'MPlusThin', 'WawaSC', 'WeibeiSC']

def extract_features(filename):
	im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	return all_features(im)

class Classifier:

	def __init__(self):
		self.kanjiFile = "kanji_list.txt"

		self.training_data = []
		self.targets = []
		self.model = GaussianNB()

		with open(self.kanjiFile, 'rb') as f:
		    reader = csv.reader(f, encoding='utf-8')
		    self.kanjiList = list(reader)

	def train(self):
		self.targets = []
		filenames = []
		for font in fonts:
			for i in range(NUM_KANJI):
				filenames.append('data/kanji-%s/kanji_%d.png' % (font, i + 1))
				self.targets.append(self.kanjiList[i])		        

		print ('Extracting Features')
		pool = Pool(NUM_THREADS)
		self.training_data = np.array(pool.map(extract_features, filenames))
		self.targets = np.array(self.targets).ravel()

		print ('Fitting features')
		self.model.fit(self.training_data, self.targets)

	def classify(self, im):
		feats = all_features(im, True, True)
		results =  self.model.predict(feats.reshape(1, -1))		
		return results

	def getKanji(self, i):
		return self.kanjiList[i]

def main():
	classifier = Classifier()
	classifier.train()

	numRight = 0
	print ('Testing')
	for i in range(NUM_KANJI):
		im = cv2.imread('data/kanji-Mamelon/kanji_%d.png' % (i + 1), cv2.IMREAD_GRAYSCALE)
		result = classifier.classify(im)
		if result == classifier.getKanji(i):
			numRight += 1

	print ("Got %d/%d right") % (numRight, NUM_KANJI)

if __name__ == "__main__":
	main()