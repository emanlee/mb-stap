# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


def load_data_TF2(data_path):
    """
    Load all NTxdata_tf*.npy files and concatenate them into a single array.
    Also return the cumulative index list for each file.
    """
    xxdata_list = []
    count_set = [0]
    count_setx = 0

    npy_files = sorted([
        f for f in os.listdir(data_path)
        if f.startswith('NTxdata_tf') and f.endswith('.npy')
    ])

    print("Detected {} files:".format(len(npy_files)))
    for f in npy_files:
        print(" -", f)

    for f in npy_files:
        xdata = np.load(os.path.join(data_path, f))
        for k in range(xdata.shape[0]):
            xxdata_list.append(xdata[k, :, :, :])

        count_setx += xdata.shape[0]
        count_set.append(count_setx)

    return np.array(xxdata_list), count_set


def spatio_temporal_attention(x):
    """Spatio-temporal attention module."""

    # Spatial attention
    s_att = layers.Conv3D(
        1, (1, 3, 3),
        activation='sigmoid',
        padding='same'
    )(x)

    x = layers.Multiply()([x, s_att])

    # Temporal attention
    t_att = layers.Permute((2, 3, 1, 4))(x)

    t_att = layers.Conv3D(
        1, (3, 1, 1),
        activation='sigmoid',
        padding='same'
    )(t_att)

    t_att = layers.Permute((3, 1, 2, 4))(t_att)

    x = layers.Multiply()([x, t_att])

    return x


def ASPP_block(x, filters=32):
    """3D Atrous Spatial Pyramid Pooling module."""

    branch1 = layers.Conv3D(
        filters,
        (1, 1, 1),
        padding='same',
        activation='relu'
    )(x)

    branch2 = layers.Conv3D(
        filters,
        (3, 3, 3),
        padding='same',
        dilation_rate=(1, 5, 5),
        activation='relu'
    )(x)

    branch3 = layers.Conv3D(
        filters,
        (3, 3, 3),
        padding='same',
        dilation_rate=(1, 10, 10),
        activation='relu'
    )(x)

    branch4 = layers.GlobalAveragePooling3D()(x)
    branch4 = layers.Dense(filters)(branch4)
    branch4 = layers.Reshape((1, 1, 1, filters))(branch4)

    branch4 = layers.UpSampling3D(
        size=(x.shape[1], x.shape[2], x.shape[3])
    )(branch4)

    concat = layers.Concatenate(axis=-1)(
        [branch1, branch2, branch3, branch4]
    )

    out = layers.Conv3D(
        filters,
        (1, 1, 1),
        activation='relu'
    )(concat)

    return out


def MBConv3D(input_tensor,
             expand_ratio,
             output_channels,
             kernel_size,
             strides):
    """3D Mobile Inverted Bottleneck Convolution block."""

    input_channels = input_tensor.shape[-1]

    # Expansion phase
    x = layers.Conv3D(
        input_channels * expand_ratio,
        1,
        padding='same'
    )(input_tensor)

    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.swish)(x)

    # Depthwise 3D convolution
    x = layers.Conv3D(
        filters=x.shape[-1],
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        groups=x.shape[-1]
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.swish)(x)

    x = spatio_temporal_attention(x)

    # Projection phase
    x = layers.Conv3D(
        output_channels,
        1,
        padding='same'
    )(x)

    x = layers.BatchNormalization()(x)

    # Residual connection
    if input_channels == output_channels and strides == (1,1,1):
        x = layers.Add()([x, input_tensor])

    return x


def SpatioTemporalModel(input_shape, num_classes):
    """MB-STAP model for spatiotemporal feature learning."""

    inputs = layers.Input(shape=input_shape)

    x = layers.Conv3D(
        8,
        (3,3,3),
        padding='same',
        kernel_initializer='he_normal'
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.swish)(x)

    x = MBConv3D(
        x,
        expand_ratio=1,
        output_channels=16,
        kernel_size=(3,3,3),
        strides=(1,2,2)
    )

    x = MBConv3D(
        x,
        expand_ratio=1,
        output_channels=32,
        kernel_size=(3,3,3),
        strides=(1,2,2)
    )

    x = ASPP_block(x, filters=32)

    x = layers.GlobalAveragePooling3D()(x)

    x = layers.Dense(128, activation=tf.nn.swish)(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(
        num_classes,
        activation='softmax'
    )(x)

    return Model(inputs, outputs)


length_TF = int(sys.argv[1])
data_path = sys.argv[2]
num_classes = int(sys.argv[3])
model_path = sys.argv[4]

# Load input data
x_test, count_set = load_data_TF2(data_path)

print(x_test.shape, 'x_test samples')

# Build model
model = SpatioTemporalModel(
    x_test.shape[1:],
    num_classes
)

# Load trained weights
model.load_weights(model_path)

print('Load model and predict...')

# Run prediction
y_predict = model.predict(x_test)

# Save prediction results
save_dir = os.path.join(
    os.getcwd(),
    'predict_results'
)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

np.save(
    save_dir + '/y_predict.npy',
    y_predict
)

with open(save_dir + '/gene_index.txt','w') as f:
    for i in count_set:
        f.write(str(i)+'\n')