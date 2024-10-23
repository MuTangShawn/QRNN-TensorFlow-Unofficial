# QRNN-TensorFlow-Unofficial
Unofficial Implementation of the Quasi Recurrent Neural Networks (Quasi-RNN/QRNN) in TensorFlow/Keras 2.10 Version.  
  
This code was written based on the official implementation of RNN layers such as ***LSTM*** and ***GRU*** in TensorFlow, but did not implement the related acceleration function of ***CuDNN*** library.  
  
The implementation of this model refers to the QRNN structure proposed in the original text.  
Bradbury, J., Merity, S., Xiong, C., & Socher, R. (2016). Quasi-recurrent neural networks. arXiv preprint arXiv:1611.01576. https://arxiv.org/abs/1611.01576  
  
## The calculation expression of QRNNCell
### For f-pooling
$$Z = \tanh(W_Z * X + b_Z)$$  
$$F = \sigma(W_F * X + b_F)$$  
$$h_t = f_t \circ h_{t-1} + (1 - f_t) \circ z_t$$  
### For fo-pooling
$$Z = \tanh(W_Z * X + b_Z)$$  
$$F = \sigma(W_F * X + b_F)$$  
$$O = \sigma(W_O * X + b_O)$$  
$$c_t = f_t \circ c_{t-1} + (1 - f_t) \circ z_t$$  
$$h_t = o_t \circ c_t$$  
### For ifo-pooling
$$Z = \tanh(W_Z * X + b_Z)$$  
$$F = \sigma(W_F * X + b_F)$$  
$$O = \sigma(W_O * X + b_O)$$  
$$I = \sigma(W_I * X + b_I)$$  

## Note
1. The calculation expression in the original SRUv5 only includes recurrent activation, the Sigmoid function (***σ***). But in SRU v4-type, the last formula was expressed as $$h_t = r_t \circ g(c_t) + (1 - r_t) \circ x_t$$. We have retained the activation function here which defaults to '***tanh***', but we have added a new parameter for this layer called '***use activation***' which defaults to '***False***'.
2. By observing the last formula, it can be found that the feature dimensions of the input and output of SRU must be equal. Therefore, we added a judgment in the model to determine whether the input and output feature dimensions are equal. When the output dimension is set to be unequal to the input dimension, the last formula will become $$h_t = r_t \circ c_t + (1 - r_t) \circ (W_h \cdot x_t)$$ to ensure that the model can run smoothly and facilitate support for multi-layer bidirectional (Bi-) SRU.

## Using example:  
```python
>>> import numpy as np
>>> import tensorflow as tf
>>> import SRU_Layer_tf2101
>>> inputs = np.random.random((32, 10, 8))
>>> sru = SRU_Layer_tf2101.SRU(4, return_sequences=True, return_state=True)
>>> output = sru(inputs)
>>> output.shape
(32, 4)
>>> sru = SRU_Layer_tf2101.SRU(4, return_sequences=True, return_state=True)
>>> whole_sequence_output, final_state = sru(inputs)
>>> whole_sequence_output.shape
(32, 10, 4)
>>> final_state.shape
(32, 4)
>>> bisru = tf.keras.layers.Bidirectional(SRU_Layer_tf2101.SRU(4, return_sequences=True), merge_mode="concat")
>>> output = bisru(inputs)
>>> output.shape
(32, 10, 8)
```  
  
## A little advertisement:
This code is part for the model compared in my article. If this code can provide assistance for your research or development work, you can cite my article in publications if convenient.  
Yao, S., Jing, C., He, X., He, Y., & Zhang, L. (2024). A TDFC-RNNs framework integrated temporal convolutional attention mechanism for InSAR surface deformation prediction: A case study in Beijing Plain. International Journal of Applied Earth Observation and Geoinformation, 134, 104199. https://doi.org/10.1016/j.jag.2024.104199
