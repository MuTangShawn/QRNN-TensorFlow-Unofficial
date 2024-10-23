import numpy as np
import tensorflow as tf
import QRNN_Layer_tf2101

inputs = np.random.random((32, 10, 8))
qrnn = QRNN_Layer_tf2101.QRNN(4)
output = qrnn(inputs)
print(output.shape)
qrnn = QRNN_Layer_tf2101.QRNN(4, return_sequences=True, return_state=True)
whole_sequence_output, final_state = qrnn(inputs)
print(whole_sequence_output.shape)
print(final_state.shape)
biqrnn = tf.keras.layers.Bidirectional(QRNN_Layer_tf2101.QRNN(4, return_sequences=True), merge_mode="concat")
output = biqrnn(inputs)
print(output.shape)