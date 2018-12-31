import tensorflow as tf
import numpy as np
import os, sys
import pickle
import csv
import operator

np.random.seed(1234)

#reload(sys)  # Reload does the trick!
#sys.setdefaultencoding('UTF8')

dataset = "TMN"
model_dir = "TMN_DocNADE_act_sigmoid_hidden_200_vocab_2000_lr_0.001_proj_False_deep_False_lambda_1.0_17_12_2018"

embeddings_dir = "./docnade_embeddings_ppl_reduced_vocab/"

if not os.path.exists(embeddings_dir):
	os.makedirs(embeddings_dir)

graph_docnade = tf.Graph()
with tf.Session(graph=graph_docnade) as sess:
	with graph_docnade.as_default():
		saver_docnade = tf.train.import_meta_graph("model/" + model_dir + "/model_ppl/model_ppl-1.meta")
		saver_docnade.restore(sess, tf.train.latest_checkpoint("model/" + model_dir + "/model_ppl/"))

		docnade_embedding_matrix = sess.run("embedding:0")
		print("docnade embedding loaded.")

f = open(embeddings_dir + dataset, "wb")
pickle.dump(docnade_embedding_matrix, f)