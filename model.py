from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, SeparableConv2D
from keras.layers import Activation, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, AveragePooling2D
from keras.models import Model
from keras.layers.merge import concatenate

def _get_smallest_shape(layers):
  return min([layer.get_shape().as_list()[1] for layer in layers])

def _get_smallest_channels(layers):
  return min([layer.get_shape().as_list()[-1] for layer in layers])
      
def _resize_layers(layers, smallest_shape):
  res_layers = []
  for layer in layers:
    layer_shape = layer.get_shape().as_list()[1]
    if layer_shape != smallest_shape:
      ps = layer_shape // smallest_shape
      res_layer = AveragePooling2D(pool_size=(ps, ps))(layer)
      res_layers.append(res_layer)
    else:
      res_layers.append(layer)
  
  return res_layers

def _reduce_channels(conc, smallest_channels):
  return Convolution2D(smallest_channels, (1, 1), padding='same', use_bias=False, activation='relu')(conc)

def merge_layers(layers):
  smallest_shape = _get_smallest_shape(layers)
  smallest_channels = _get_smallest_channels(layers)
  
  resized_layers = _resize_layers(layers, smallest_shape)
  conc = concatenate(resized_layers)
  reduced = _reduce_channels(conc, smallest_channels)
  return reduced

def get_model(num_classes=10):
  inp = Input(shape=(32,32,3), name='input')

  sep1_r = SeparableConv2D(64, (5, 5), use_bias=False, name='sep1', padding='same', activation='relu')(inp)
  sep1 = BatchNormalization()(sep1_r)

  conv1_r = Convolution2D(64, (5, 5), use_bias=False, name='conv1', padding='same', activation='relu')(sep1)
  conv1 = BatchNormalization()(conv1_r)

  conv2_r = Convolution2D(64, (5, 5), use_bias=False, name='conv2', padding='same', activation='relu')(conv1)
  conv2 = BatchNormalization()(conv2_r)

  concat1_r = merge_layers([sep1, conv2])
  concat1 = BatchNormalization()(concat1_r)

  sep2_r = SeparableConv2D(64, (5, 5), use_bias=False, name='sep2', padding='same', activation='relu')(concat1)
  sep2 = BatchNormalization()(sep2_r)

  concat2_r = merge_layers([sep1, sep2])
  concat2 = BatchNormalization()(concat2_r)
  pool1 = MaxPooling2D(2, 2, name='pool1')(concat2)

  sep3_r = SeparableConv2D(128, (3, 3), use_bias=False, name='sep3', padding='same', activation='relu')(pool1)
  sep3 = BatchNormalization()(sep3_r)

  concat4_r = merge_layers([sep2, sep3])
  concat4 = BatchNormalization()(concat4_r)

  conv3_r = Convolution2D(128, (5, 5), use_bias=False, name='conv3', padding='same', activation='relu')(concat4)
  conv3 = BatchNormalization()(conv3_r)

  concat5_r = merge_layers([conv2, sep2, sep3, conv3])
  concat5 = BatchNormalization()(concat5_r)

  sep4_r = SeparableConv2D(128, (3, 3), use_bias=False, name='sep4', padding='same', activation='relu')(concat5)
  sep4 = BatchNormalization()(sep4_r)

  concat6_r = merge_layers([conv2, conv3, sep1, sep2, sep3, sep4])
  concat6 = BatchNormalization()(concat6_r)

  sep5_r = SeparableConv2D(128, (5, 5), use_bias=False, name='sep5', padding='same', activation='relu')(concat6)
  sep5 = BatchNormalization()(sep5_r)

  concat7_r = merge_layers([sep1, sep2, sep3, sep4, sep5])
  concat7 = BatchNormalization()(concat7_r)
  pool2 = MaxPooling2D(2, 2, name='pool2')(concat7)

  concat8_r = merge_layers([conv3, pool2])
  concat8 = BatchNormalization()(concat8_r)

  conv4_r = Convolution2D(256, (5, 5), use_bias=False, name='conv4', padding='same', activation='relu')(concat8)
  conv4 = BatchNormalization()(conv4_r)

  concat9_r = merge_layers([sep2, sep4, conv2, conv4])
  concat9 = BatchNormalization()(concat9_r)

  sep6_r = SeparableConv2D(256, (5, 5), use_bias=False, name='sep6', padding='same', activation='relu')(concat9)
  sep6 = BatchNormalization()(sep6_r)

  concat10_r = merge_layers([sep3, conv1, conv2, conv4, sep6])
  concat10 = BatchNormalization()(concat10_r)

  conv5_r = Convolution2D(256, (3, 3), use_bias=False, name='conv5', padding='same', activation='relu')(concat10)
  conv5 = BatchNormalization()(conv5_r)

  concat11_r = merge_layers([conv2, sep1, sep2, sep3, sep4, sep6, conv5])
  concat11 = BatchNormalization()(concat11_r)

  sep7_r = SeparableConv2D(256, (5, 5), use_bias=False, name='sep7', padding='same', activation='relu')(concat11)
  sep7 = BatchNormalization()(sep7_r)

  concat12_r = merge_layers([sep4, sep2, sep6, sep7])
  concat12 = BatchNormalization()(concat12_r)

  convf = Convolution2D(num_classes, (1, 1), activation='softmax', use_bias=False)(concat12)
  gap = GlobalAveragePooling2D()(convf)

  model = Model(inputs=[inp], outputs=[gap])


  # Compile the model
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  print(model.summary())

  return model

model = get_model()
