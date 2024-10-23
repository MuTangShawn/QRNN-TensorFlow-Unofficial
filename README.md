# QRNN-TensorFlow-Unofficial
Unofficial Implementation of the Quasi Recurrent Neural Networks (Quasi-RNN/QRNN) in TensorFlow/Keras 2.10 Version.  
  
This code was written based on the official implementation of RNN layers such as ***LSTM*** and ***GRU*** in TensorFlow, but did not implement the related acceleration function of ***CuDNN*** library.  
  
The implementation of this model refers to the QRNN structure proposed in the original text.  
Bradbury, J., Merity, S., Xiong, C., & Socher, R. (2016). Quasi-recurrent neural networks. arXiv preprint arXiv:1611.01576. https://arxiv.org/abs/1611.01576  
  
## The calculation expression of QRNNCell
### For ***f-pooling***
$$Z = \tanh(W_Z * X + b_Z)$$  
$$F = \sigma(W_F * X + b_F)$$  
$$h_t = f_t \circ h_{t-1} + (1 - f_t) \circ z_t$$  
### For ***fo-pooling***
$$Z = \tanh(W_Z * X + b_Z)$$  
$$F = \sigma(W_F * X + b_F)$$  
$$O = \sigma(W_O * X + b_O)$$  
$$c_t = f_t \circ c_{t-1} + (1 - f_t) \circ z_t$$  
$$h_t = o_t \circ c_t$$  
### For ***ifo-pooling***
$$Z = \tanh(W_Z * X + b_Z)$$  
$$I = \sigma(W_I * X + b_I)$$  
$$F = \sigma(W_F * X + b_F)$$  
$$O = \sigma(W_O * X + b_O)$$  
$$c_t = f_t \circ c_{t-1} + i_t \circ z_t$$  
$$h_t = o_t \circ c_t$$  

## Note
1. In the original text, the calculation expression of QRNN does not include **bias** terms. Here, we have retained this function, but the ***use_bias*** parameter is set to ***False*** by default.
2. In this code, we have implemented the **3 pooling modes** recommended in the original article. The selection of these 3 modes is controlled by the '***pool_mode***' parameter, and there are **3 options** to choose from: '***f***', '***fo***', and '***ifo***', with '***fo***' being the default.
3. The implementation of QRNN here can be wrapped using the 'tf.keras.layers.Bidirectional' API. Please refer to the usage examples for details.

## Using example:  
```python
>>> import numpy as np
>>> import tensorflow as tf
>>> import QRNN_Layer_tf2101
>>> inputs = np.random.random((32, 10, 8))
>>> qrnn = QRNN_Layer_tf2101.QRNN(4, return_sequences=True, return_state=True)
>>> output = qrnn(inputs)
>>> output.shape
(32, 4)
>>> qrnn = QRNN_Layer_tf2101.QRNN(4, return_sequences=True, return_state=True)
>>> whole_sequence_output, final_state = qrnn(inputs)
>>> whole_sequence_output.shape
(32, 10, 4)
>>> final_state.shape
(32, 4)
>>> biqrnn = tf.keras.layers.Bidirectional(QRNN_Layer_tf2101.QRNN(4, return_sequences=True), merge_mode="concat")
>>> output = biqrnn(inputs)
>>> output.shape
(32, 10, 8)
```  
  
## A little advertisement:
This code is part for the model compared in my article. If this code can provide assistance for your research or development work, you can cite my article in publications if convenient.  
Yao, S., Jing, C., He, X., He, Y., & Zhang, L. (2024). A TDFC-RNNs framework integrated temporal convolutional attention mechanism for InSAR surface deformation prediction: A case study in Beijing Plain. International Journal of Applied Earth Observation and Geoinformation, 134, 104199. https://doi.org/10.1016/j.jag.2024.104199

## License
This project is licensed under the Mozilla Public License 2.0 (MPL 2.0) - see the [LICENSE](./LICENSE) file for details.
