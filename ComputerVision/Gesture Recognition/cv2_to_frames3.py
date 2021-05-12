from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
from pdb import set_trace
import numpy as np
import os
import pickle
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tqdm import tqdm
import cv2


boundaries = [
    ([42, 47, 89], [180, 188, 236]),
    ([36, 85, 141], [125, 194, 241])
]
dsize = (1920,1080)

cap = cv2.VideoCapture(0)
frate = 10
i = 0
sec = 0

mloc = os.getcwd()
model_file = mloc + '/retrained_graph.pb'
frames_folder = mloc + '/test_frames/'
input_layer = 'Placeholder'
output_layer = 'final_result'
batch_size = 100

font = cv2.FONT_HERSHEY_SIMPLEX

while (cap.isOpened()):
	ret, frame = cap.read()
	if ret == False :
		break

	# cv2.imshow('Capturing the Gesture',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	if i % frate == 0:
		
		lower, upper = boundaries[0]
		lower = np.array(lower, dtype="uint8")
		upper = np.array(upper, dtype="uint8")
		mask1 = cv2.inRange(frame, lower, upper)

		lower, upper = boundaries[1]
		lower = np.array(lower, dtype="uint8")
		upper = np.array(upper, dtype="uint8")
		mask2 = cv2.inRange(frame, lower, upper)

		mask = cv2.bitwise_or(mask1, mask2)
		frame = cv2.bitwise_and(frame, frame, mask=mask)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = cv2.resize(frame, dsize)
		sec = sec + 1
		

	





		###################### Eval ##################################




		########################################################

		prediction = 'Output'

		cv2.putText(frame, prediction, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
		cv2.imshow("video", frame)
		cv2.imwrite(str(sec)+'.jpg',frame)
		predictions = predict_on_frames(frames_folder, model_file, input_layer, output_layer, batch_size)
		#print(sec)
	i += 1
######################## call spatial #################################
    
#predictions = predict_on_frames(frames_folder, model_file, input_layer, output_layer, batch_size)

out_file = 'predicted-frames-%s-%s.pkl' % (output_layer.split("/")[-1], 'test')
print("Dumping predictions to: %s" % (out_file))
with open(out_file, 'wb') as fout:
	pickle.dump(predictions, fout)

print("Done.")
##################### spatial #################################
def load_graph(model_file):
	graph = tf.Graph()
	graph_def = tf.compat.v1.GraphDef()

	with open(model_file, "rb") as f:
	    graph_def.ParseFromString(f.read())
	with graph.as_default():
	    tf.import_graph_def(graph_def)

	return graph


def read_tensor_from_image_file(frames, input_height=299, input_width=299, input_mean=0, input_std=255):
	tf.compat.v1.disable_eager_execution()
	input_name = "file_reader"
	frames = [(tf.io.read_file(frame, input_name), frame) for frame in frames]
	decoded_frames = []
	for frame in frames:
	    file_name = frame[1]
	    file_reader = frame[0]
	    if file_name.endswith(".png"):
	        image_reader = tf.image.decode_png(file_reader, channels=3, name="png_reader")
	    elif file_name.endswith(".gif"):
	        image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name="gif_reader"))
	    elif file_name.endswith(".bmp"):
	        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
	    else:
	        image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
	    decoded_frames.append(image_reader)
	float_caster = [tf.cast(image_reader, tf.float32) for image_reader in decoded_frames]
	float_caster = tf.stack(float_caster)
	resized = tf.image.resize(float_caster, [input_height, input_width])
	normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
	sess = tf.compat.v1.Session()
	result = sess.run(normalized)
	return result


def load_labels(label_file):
	label = []
	proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
	for l in proto_as_ascii_lines:
	    label.append(l.rstrip())
	return label


	def predict(graph, image_tensor, input_layer, output_layer):
	#set_trace()
		tf.compat.v1.disable_eager_execution()
	input_name = "import/" + input_layer
	output_name = "import/" + output_layer
	input_operation = graph.get_operation_by_name(input_name)
	output_operation = graph.get_operation_by_name(output_name)
	with tf.compat.v1.Session(graph=graph) as sess:
	    results = sess.run(
	        output_operation.outputs[0],
	        {input_operation.outputs[0]: image_tensor}
	    )
	results = np.squeeze(results)
	return results


def predict_on_frames(frames_folder, model_file, input_layer, output_layer, batch_size):
	input_height = 299
	input_width = 299
	input_mean = 0
	input_std = 255
	batch_size = batch_size
	graph = load_graph(model_file)

	labels_in_dir = os.listdir(frames_folder)
	frames = [each for each in os.walk(frames_folder) if os.path.basename(each[0]) in labels_in_dir]

	predictions = []
	for each in frames:
	    label = each[0]
	    print("Predicting on frame of %s\n" % (label))
	    for i in tqdm(range(0, len(each[2]), batch_size), ascii=True):
	        batch = each[2][i:i + batch_size]
	        try:
	            batch = [os.path.join(label, frame) for frame in batch]
	            frames_tensors = read_tensor_from_image_file(batch, input_height=input_height, input_width=input_width, input_mean=input_mean, input_std=input_std)
	            pred = predict(graph, frames_tensors, input_layer, output_layer)
	            pred = [[each.tolist(), os.path.basename(label)] for each in pred]
	            predictions.extend(pred)

	        except KeyboardInterrupt:
	            print("You quit with ctrl+c")
	            sys.exit()

	        except Exception as e:
	            print("Error making prediction: %s" % (e))
	            x = input("\nDo You Want to continue on other samples: y/n")
	            if x.lower() == 'y':
	                continue
	            else:
	                sys.exit()
	return predictions
# reduce tf verbosity
cap.release()
cv2.destroyAllWindows()