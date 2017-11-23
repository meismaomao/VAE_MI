from __future__ import division
import tensorflow as tf
import math
import numpy as np
import numpy.random as nr
from math import log
import scipy.spatial as ss
from scipy.special import digamma
# import cv2
import multiprocessing
import os
# from joblib import Parallel, delayed
from information_process import *
from numpy import linalg as LA
from tensorflow.examples.tutorials.mnist import input_data

NUM_CORES = multiprocessing.cpu_count()
IMAGE_SIZE = 784
Z_DIM = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 64
SIZE = 28

layer_save = {}
# collection_params = []



def single_entropy(data, base=2):
	summation = 0.0
	# print(data)
	num = len(data)
	data = np.array(data)
	value_datas = set(data)
	for value_data in value_datas:
		px = sum(np.where(data == value_data, 1, 0)) / num
		if px > 0.0:
			summation += px * math.log(px, base)
	if summation == 0.0:
		return summation
	else:
		return -1.0 * summation


# calc function
def calc_fun(data_dict, data_sample):
	entropy_all = {}
	# names_key = ['ew1', 'ew2', 'dw1', 'dw2', 'dw3']
	for name, value in data_dict.items():
		# print(name)
		mutual_info = get_information(value, data_sample, 10)
		if 'ew1' in name:
			entropy_all['ew1'] = mutual_info
		elif 'ew2' in name:
			entropy_all['ew2'] = mutual_info
		elif 'ew3' in name:
			entropy_all['ew3'] = mutual_info
		elif 'ew4' in name:
			entropy_all['ew4'] = mutual_info
		elif 'dw1' in name:
			entropy_all['dw1'] = mutual_info
		elif 'dw2' in name:
			entropy_all['dw2'] = mutual_info
		else:
			entropy_all["dw3"] = mutual_info
	return entropy_all


def calc_entropy(data, k=3, base=2):
	assert k <= len(data) - 1
	d = len(data[0])
	N = len(data)
	intens = 1e-10
	data = [list(p + intens * nr.rand(len(data[0]))) for p in data]
	tree = ss.cKDTree(data)
	nn = [tree.query(point, k + 1, p=float('inf'))[0][k] for point in data]
	const = digamma(N) - digamma(k) + d * log(2)
	return (const + d * np.mean(map(log, nn))) / log(base)


def calc_batch_entropy(data_dict):
	entropy_all = {}
	# names_key = ['ew1', 'ew2','ew3', 'ew4', 'dw1', 'dw2', 'dw3']
	for name, arrs in data_dict.items():
		# print(arrs)
		entropy = []
		for arr in arrs:
			# print(arr)
			arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
			arr = np.array(arr)
			bins = np.linspace(0, 1, 2)
			digitized = bins[np.digitize(np.squeeze(arr.reshape(1, -1)), bins) - 1].reshape(len(arr), -1)
			digitized = np.squeeze(digitized)
			entr = single_entropy(digitized)
			entropy.append(entr)
		# print(entropy)
		if 'ew1' in name:
			entropy_all['ew1'] = np.mean(entropy)
		elif 'ew2' in name:
			entropy_all['ew2'] = np.mean(entropy)
		elif 'ew3' in name:
			entropy_all['ew3'] = np.mean(entropy)
		elif 'ew4' in name:
			entropy_all['ew4'] = np.mean(entropy)
		elif 'dw1' in name:
			entropy_all['dw1'] = np.mean(entropy)
		elif 'dw2' in name:
			entropy_all['dw2'] = np.mean(entropy)
		elif 'dw3' in name:
			entropy_all["dw3"] = np.mean(entropy)
	return entropy_all


def xavier_init(inputs, outputs, constant=1):
	# Xavier initialization
	low = -constant * np.sqrt(6.0 / (inputs + outputs))
	high = constant * np.sqrt(6.0 / (inputs + outputs))
	return tf.random_uniform((inputs, outputs), minval=low,
							 maxval=high, dtype=tf.float32)


def _weights_variable(inputs, outputs, name):
	w = tf.Variable(xavier_init(inputs, outputs), name=name)
	return w


def _bias_variable(num, name):
	b = tf.Variable(tf.zeros([num], dtype=tf.float32), name=name)
	return b


def encoder(x, ehidden1, ehidden2, z_dim):
	with tf.name_scope("encoder"):
		w1 = _weights_variable(IMAGE_SIZE, ehidden1, name="w1")
		# collection_params.append(w1.op.name)
		b1 = _bias_variable(ehidden1, name="b1")
		h1 = tf.matmul(x, w1) + b1
		h1 = tf.nn.relu(h1)
		layer_save['ew1'] = w1

		w2 = _weights_variable(ehidden1, ehidden2, name="w2")
		# collection_params.append(w2.op.name)
		b2 = _bias_variable(ehidden2, name="b2")
		h2 = tf.matmul(h1, w2) + b2
		h2 = tf.nn.relu(h2)
		layer_save['ew2'] = w2

		w3 = _weights_variable(ehidden2, z_dim, name="mean_w3")
		# collection_params.append(w3.op.name)
		b3 = _bias_variable(z_dim, name="mean_b3")
		mean = tf.matmul(h2, w3) + b3
		layer_save['ew3'] = w3

		w4 = _weights_variable(ehidden2, z_dim, name="stddev_w4")
		# collection_params.append(w4.op.name)
		b4 = _bias_variable(z_dim, name='stddev_b4')
		stddev = tf.matmul(h2, w4) + b4
		layer_save['ew4'] = w4

		return mean, stddev


def decoder(z, dhidden1, dhidden2, img_size):
	with tf.name_scope("decoder"):
		w1 = _weights_variable(Z_DIM, dhidden1, name="w1")
		# collection_params.append(w1.op.name)
		b1 = _bias_variable(dhidden1, name="b1")
		h1 = tf.matmul(z, w1) + b1
		h1 = tf.nn.relu(h1)
		layer_save['dw1'] = w1

		w2 = _weights_variable(dhidden1, dhidden2, name="w2")
		# collection_params.append(w2.op.name)
		b2 = _bias_variable(dhidden2, name="b2")
		h2 = tf.matmul(h1, w2) + b2
		h2 = tf.nn.relu(h2)
		layer_save['dw2'] = w2

		w3 = _weights_variable(dhidden2, img_size, name="w3")
		# collection_params.append(w3.op.name)
		b3 = _bias_variable(img_size, name="b3")
		h3 = tf.matmul(h2, w3) + b3
		img = tf.nn.sigmoid(h3)
		layer_save['dw3'] = w3

		return img


def autoencoder(x, ehidden1, ehidden2, dhidden1, dhidden2, z_dim, img_size):
	mu, sigma = encoder(x, ehidden1, ehidden2, z_dim)
	z = mu + tf.multiply(tf.sqrt(tf.exp(sigma)), tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32))
	y = decoder(z, dhidden1, dhidden2, img_size)
	x = tf.reshape(x, [BATCH_SIZE, -1])
	y = tf.reshape(y, [BATCH_SIZE, -1])
	# y = tf.clip_by_value(y, 1e-8, 1-1e-8)

	marginal_likelihood = -1.0 * tf.reduce_sum(x * tf.log(1e-10 + y) + (1 - x) * tf.log(1e-10 + 1 - y), 1)

	kl_divergence = -0.5 * tf.reduce_sum(1 + sigma - tf.square(mu) - tf.exp(sigma), 1)

	marginal_likelihood = tf.reduce_mean(marginal_likelihood)
	kl_divergence = tf.reduce_mean(kl_divergence)

	loss = marginal_likelihood + kl_divergence

	return y, loss, marginal_likelihood, kl_divergence


def train_op_fun(loss):
	var_list = [var for var in tf.trainable_variables() if 'w' in var.name]
	# print(var_list)
	# assert var_list == collection_params
	grads = tf.gradients(loss, var_list)
	train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)
	return train_op, grads


def main():
	mnist = input_data.read_data_sets("mnist_data/", one_hot=True)
	train_images = mnist.train.images
	test_images = mnist.test.images
	images = np.concatenate([train_images, test_images])
	image_ = tf.placeholder(tf.float32, [None, IMAGE_SIZE])

	y, loss, marginal_likelihood, kl_divergence = \
		autoencoder(image_, 1024, 512, 512, 1024, Z_DIM, IMAGE_SIZE)
	train_op, grads = train_op_fun(loss)

	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		print("initialized!")
		eh1_entropy = []
		eh2_entropy = []
		eh3_entropy = []
		eh4_entropy = []
		dh1_entropy = []
		dh2_entropy = []
		dh3_entropy = []

		loss_save = []
		loss_kl = []
		loss_mar = []
		grads_mean = [[] for _ in range(7)]
		grads_std = [[] for _ in range(7)]
		for step in range(300000):
			offset = (step * BATCH_SIZE) % (len(images) - BATCH_SIZE)
			batch_data = images[offset:(offset + BATCH_SIZE), :]
			feed_dict = {image_: batch_data}
			Loss, Logits, Marginal_likelihood, KL_divergence, _, Layer_save, Grads = \
				sess.run([loss, y, marginal_likelihood, kl_divergence, train_op, layer_save, grads],
						 feed_dict=feed_dict)

			print("step %d" % step)
			print("Loss is %g, marginal_likelihood %g, kl_divergence %g" %
				  (Loss, Marginal_likelihood, KL_divergence))

			for i in range(7):
				grads_mean[i].append(LA.norm(np.mean(Grads[i])))
				grads_std[i].append(np.std(Grads[i]))

			loss_save.append(Loss)
			loss_kl.append(KL_divergence)
			loss_mar.append(Marginal_likelihood)
			Entropy_get = calc_batch_entropy(Layer_save)
			eh1_entropy.append(Entropy_get["ew1"])
			eh2_entropy.append(Entropy_get["ew2"])
			eh3_entropy.append(Entropy_get["ew3"])
			eh4_entropy.append(Entropy_get["ew4"])
			dh1_entropy.append(Entropy_get["dw1"])
			dh2_entropy.append(Entropy_get["dw2"])
			dh3_entropy.append(Entropy_get["dw3"])

		np.savetxt("eh1_entropy.txt", eh1_entropy)
		np.savetxt("eh2_entropy.txt", eh2_entropy)
		np.savetxt("eh3_entropy.txt", eh3_entropy)
		np.savetxt("eh4_entropy.txt", eh4_entropy)
		np.savetxt("dh1_entropy.txt", dh1_entropy)
		np.savetxt("dh2_entropy.txt", dh2_entropy)
		np.savetxt("dh3_entropy.txt", dh3_entropy)
		np.savetxt("loss_all.txt", loss_save)
		np.savetxt("loss_kl.txt", loss_kl)
		np.savetxt("loss_mar.txt", loss_mar)
		np.savetxt('grads_mean.txt', grads_mean)
		np.savetxt('grads_std.txt', grads_std)


if __name__ == "__main__":
	main()
