python predict_spatial.py retrained_graph.pb train_frames --output_layer="module_apply_default/InceptionV3/Logits/GlobalPool" --batch=100


python rnn_train.py predicted-frames-final_result-train.pkl non_pool.model
y

python rnn_train.py predicted-frames-GlobalPool-train.pkl pool.model
y
