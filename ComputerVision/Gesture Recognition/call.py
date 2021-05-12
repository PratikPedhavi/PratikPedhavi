from rnn_eval import *

if __name__ == '__main__':
	main("predicted-frames-GlobalPool-test.pkl", 201, 32, "retrained_labels.txt", "pool.model")