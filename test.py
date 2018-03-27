from keras.layers import Input, merge
from keras.models import Model
import numpy as np

input_a = np.reshape([1, 2, 3], (1, 3))
input_b = np.reshape([4, 5, 6], (1, 3))

a = Input(shape=(3, ))
b = Input(shape=(3, ))

# concat = merge([a, b], mode='concat', concat_axis=-1)
dot = merge([a, b], mode='dot', dot_axes=1)
cos = merge([a, b], mode='cos', dot_axes=0)

# model_concat = Model(inputs=[a, b], outputs=concat)
model_dot = Model(inputs=[a, b], outputs=dot)
model_cos = Model(inputs=[a, b], outputs=cos)

# print(model_concat.predict([input_a, input_b]))
print(model_dot.predict([input_a, input_b]))
print(model_cos.predict([input_a, input_b]))
