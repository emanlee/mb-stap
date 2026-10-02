from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling3D, Attention
from keras.callbacks import EarlyStopping,ModelCheckpoint
import numpy as np
import os,sys
import matplotlib
matplotlib.use('Agg')
from sklearn import metrics

from scipy import interp
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv3D, Conv1D, BatchNormalization, Activation, GlobalAveragePooling3D, Dense, Reshape, Multiply, Add, GlobalAveragePooling2D, Concatenate
import tensorflow as tf


import pandas as pd
from collections import defaultdict
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import seaborn as sns


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def calculate_metrics(y_true, y_pred, num_classes=3):
    """Extended metric calculation function. Returns a dictionary of metrics."""
    # Convert to class labels
    y_true_labels = np.argmax(y_true, axis=1) if num_classes > 2 else y_true.flatten()
    y_pred_labels = np.argmax(y_pred, axis=1) if num_classes > 2 else (y_pred > 0.5).astype(int)

    metrics_dict = {}

    # Classification report
    report = classification_report(
        y_true_labels, y_pred_labels,
        target_names=[f'Class_{i}' for i in range(num_classes)],
        output_dict=True
    )

    # AUC metrics
    if num_classes == 2:
        metrics_dict['AUROC'] = roc_auc_score(y_true, y_pred[:, 1])
        precision, recall, _ = precision_recall_curve(y_true.flatten(), y_pred[:, 1])
        metrics_dict['AUPRC'] = auc(recall, precision)
    else:
        # AUROC (One-vs-Rest)
        y_true_bin = label_binarize(y_true_labels, classes=range(num_classes))
        metrics_dict['AUROC_macro'] = roc_auc_score(
            y_true_bin, y_pred, average='macro', multi_class='ovr'
        )

        # AUPRC per class
        precision_dict, recall_dict, pr_auc_dict = {}, {}, {}
        for i in range(num_classes):
            precision_dict[i], recall_dict[i], _ = precision_recall_curve(
                y_true_bin[:, i], y_pred[:, i]
            )
            pr_auc_dict[i] = auc(recall_dict[i], precision_dict[i])
        metrics_dict['AUPRC_macro'] = np.mean(list(pr_auc_dict.values()))

    # Overall metrics
    metrics_dict.update({
        'Accuracy': report['accuracy'],
        'Precision_macro': report['macro avg']['precision'],
        'Recall_macro': report['macro avg']['recall'],
        'F1_macro': report['macro avg']['f1-score']
    })

    return metrics_dict, (precision_dict, recall_dict), report


def plot_radar(metrics_dict, num_classes, save_path):
    """Radar chart plotting function."""
    categories = ['Precision', 'Recall', 'F1']
    class_labels = [f'Class_{i}' for i in range(num_classes)]

    data = []
    for label in class_labels:
        if label in metrics_dict:
            data.append([
                metrics_dict[label]['precision'],
                metrics_dict[label]['recall'],
                metrics_dict[label]['f1-score']
            ])
        else:
            print(f"Warning: {label} not found in metrics_dict. Filled with zeros.")
            data.append([0.0, 0.0, 0.0])

    data_norm = np.array(data)
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    colors = ['#FF6384', '#36A2EB', '#FFCE56']
    for i, (values, color) in enumerate(zip(data_norm, colors)):
        values = values.tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2,
                label=class_labels[i].replace('_', ' '))
        ax.fill(angles, values, color=color, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_thetagrids(np.degrees(angles[:-1]),
                      [cat.upper() for cat in categories],
                      fontsize=12)

    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 5))

    plt.title('Classification Metrics Radar Chart',
              loc='center',
              fontdict={'fontsize': 14, 'fontweight': 'bold'})

    plt.legend(bbox_to_anchor=(1.25, 1),
               loc='upper right',
               borderaxespad=0.)

    plt.savefig(os.path.join(save_path, 'metrics_radar.pdf'),
                bbox_inches='tight',
                dpi=300)
    plt.close()

# ================================================================================================

def load_data(indel_list, data_path):
    import numpy as np

    xxdata_list = []
    yydata = []
    count_set = [0]
    count_setx = 0

    for i in indel_list:
        xdata = np.load(data_path + f'/NTxdata_tf{i}.npy')  # Load x data
        ydata = np.load(data_path + f'/ydata_tf{i}.npy')    # Load labels

        for k in range(len(ydata) // 3):
            # Read labels 0, 1, 2
            xxdata_list.append(xdata[3 * k, :, :, :, :])
            yydata.append(1)

            xxdata_list.append(xdata[3 * k + 1, :, :, :, :])
            yydata.append(2)

            xxdata_list.append(xdata[3 * k + 2, :, :, :, :])
            yydata.append(0)

        count_setx += len(ydata)
        count_set.append(count_setx)
        print(f'Processing {i}: {len(ydata)} samples')

    xxdata_array = np.array(xxdata_list)
    yydata_array = np.array(yydata).astype('int')

    print(xxdata_array.shape)
    return xxdata_array, yydata_array, count_set


data_augmentation = False
batch_size = 512
num_classes = 3
epochs = 20
length_TF = 36  # Number of TFs

model_name = 'keras_cnn_trained_model_shallow.h5'

whole_data_TF = [i for i in range(length_TF)]

data_path = '/home/yanke/GTRD_NT_8X8_4_mesc2'  # Path of generated 3D NEPDF files and labels
# data_path = '/home/yanke/GTRD_NT_8X8_5_hesc1'
# data_path = '/home/yanke/TDL-2025/GTRD_NT_8X8_6'
# data_path = '/home/yanke/GTRD_NT_8X8_9_mesc1'


def spatio_temporal_attention(x):
    """
    Spatio-Temporal Attention Module

    Args:
        x: Input 3D feature tensor
           shape [batch_size, depth, height, width, channels]

    Returns:
        Feature tensor after spatio-temporal attention weighting
    """

    # Spatial attention
    s_att = layers.Conv3D(1, (1, 3, 3), activation='sigmoid', padding='same')(x)
    x = layers.Multiply()([x, s_att])

    # Temporal attention
    t_att = layers.Permute((2, 3, 1, 4))(x)
    t_att = layers.Conv3D(1, (3, 1, 1), activation='sigmoid', padding='same')(t_att)
    t_att = layers.Permute((3, 1, 2, 4))(t_att)
    x = layers.Multiply()([x, t_att])

    return x

def ASPP_block(x, filters=32):
    """Improved ASPP module."""

    # Branch 1: 1×1×1 convolution
    branch1 = layers.Conv3D(
        filters, (1, 1, 1), padding='same', activation='relu'
    )(x)

    # Branch 2: 3×3×3 dilated convolution (1,5,5)
    branch2 = layers.Conv3D(
        filters, (3, 3, 3),
        padding='same',
        dilation_rate=(1, 5, 5),
        activation='relu'
    )(x)

    # Branch 3: 3×3×3 dilated convolution (1,10,10)
    branch3 = layers.Conv3D(
        filters, (3, 3, 3),
        padding='same',
        dilation_rate=(1, 10, 10),
        activation='relu'
    )(x)

    # Branch 4: global average pooling → channel projection → upsampling
    branch4 = layers.GlobalAveragePooling3D()(x)
    branch4 = layers.Dense(filters)(branch4)
    branch4 = layers.Reshape((1, 1, 1, filters))(branch4)
    branch4 = layers.UpSampling3D(
        size=(x.shape[1], x.shape[2], x.shape[3])
    )(branch4)

    # Multi-scale feature fusion
    concat = layers.Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
    out = layers.Conv3D(filters, (1, 1, 1), activation='relu')(concat)

    return out


def MBConv3D(input_tensor, expand_ratio, output_channels, kernel_size, strides):
    """Improved 3D MBConv module."""
    input_channels = input_tensor.shape[-1]

    # Expansion phase
    x = layers.Conv3D(input_channels * expand_ratio, 1, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.swish)(x)

    # Depthwise convolution (temporal dimension preserved)
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
    x = layers.Conv3D(output_channels, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Residual connection
    if input_channels == output_channels and strides == (1, 1, 1):
        x = layers.Add()([x, input_tensor])

    return x


def SpatioTemporalModel(xx):
    """Spatio-temporal feature fusion model."""

    input = layers.Input(xx.shape[1:])

    # Stem convolution
    x = layers.Conv3D(
        8, (3, 3, 3),
        strides=(1, 1, 1),
        padding='same',
        kernel_initializer='he_normal'
    )(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.swish)(x)

    # MBConv blocks
    x = MBConv3D(x, expand_ratio=1, output_channels=16,
                 kernel_size=(3, 3, 3), strides=(1, 2, 2))
    x = MBConv3D(x, expand_ratio=1, output_channels=32,
                 kernel_size=(3, 3, 3), strides=(1, 2, 2))

    # ASPP module
    x = ASPP_block(x, filters=32)

    # Global feature aggregation
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(128, activation=tf.nn.swish)(x)
    x = layers.Dropout(0.5)(x)

    # Output layer
    outputs = layers.Dense(3, activation='softmax')(x)

    return Model(input, outputs)



metrics_history = defaultdict(list)
all_y_test = []
all_y_pred = []

###################################################################################################################################
for test_indel in range(1,4): ################## three fold cross validation
    test_TF = [i for i in range (int(np.ceil((test_indel-1)*0.333333*length_TF)),int(np.ceil(test_indel*0.333333*length_TF)))]
    train_TF = [i for i in whole_data_TF if i not in test_TF]                                                                  #

    (x_train, y_train, count_set_train) = load_data(train_TF, data_path)
    (x_test, y_test, count_set) = load_data(test_TF, data_path)
    print(x_train.shape, 'x_train samples')
    print(x_test.shape, 'x_test samples')

############################### model
    save_dir = os.path.join(os.getcwd(),str(test_indel) + 'Xlr00001_3dse_simple_rate_55_1010+shi-time5' + str(epochs))
    if num_classes >2:
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
    print(y_train.shape, 'y_train samples')
    print(y_test.shape, 'y_test samples')


    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if num_classes <2:
        print ('no enough categories')
        sys.exit()
    elif num_classes ==2:
        model = SpatioTemporalModel(x_train)
        sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])
    else:
        model = SpatioTemporalModel(x_train)
        sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=600, verbose=0, mode='auto')
    checkpoint1 = ModelCheckpoint(filepath=save_dir + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                  verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    checkpoint2 = ModelCheckpoint(filepath=save_dir + '/weights.hdf5', monitor='val_accuracy', verbose=1,
                                  save_best_only=True, mode='auto', period=1)
    callbacks_list = [checkpoint2, early_stopping]
    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,validation_split=0.2,
                  shuffle=True, callbacks=callbacks_list)

    # Save model and weights
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    y_predict = model.predict(x_test)
    np.save(save_dir+'/end_y_test_3dse_simple_rate_55_1010+shi-time5.npy',y_test)
    np.save(save_dir+'/end_y_predict_3dse_simple_rate_55_1010+shi-time5.npy',y_predict)

    # =======================================================================================================
    # Compute extended metrics
    metrics_dict, (precision_dict, recall_dict), report = calculate_metrics(
        y_test, y_predict, num_classes
    )

    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(save_dir + '/classification_report_3dse_simple_rate_55_1010+shi-time5.csv')

    # Plot PR curve
    plt.figure(figsize=(10, 8))
    colors = cycle(['#ff0000', '#00ff00', '#0000ff'])

    if num_classes == 2:
        precision, recall, _ = precision_recall_curve(y_test.flatten(), y_predict[:, 1])
        plt.plot(recall, precision, color='b', label='PR Curve')
    else:
        for i, color in zip(range(num_classes), colors):
            plt.plot(
                recall_dict[i], precision_dict[i],
                color=color,
                label=f'Class {i} (AP={metrics_dict["AUPRC_macro"]:.2f})'
            )

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(save_dir + '/PR_curve_3dse_simple_rate_55_1010+shi-time5.pdf')

    # Plot radar chart
    if num_classes > 2:
        plot_radar(report, num_classes, save_dir)
    else:
        binary_report = {
            'Class_0': report['0'],
            'Class_1': report['1'],
            'accuracy': report['accuracy']
        }
        plot_radar(binary_report, num_classes, save_dir)

    # Save full metrics
    with open(save_dir + '/full_metrics_3dse_simple_rate_55_1010+shi-time5.txt', 'w') as f:
        for key, value in metrics_dict.items():
            f.write(f"{key}: {value}\n")

    # Print core metrics
    print('=' * 30 + 'Core Metrics Summary' + '=' * 30)
    print(f"Accuracy: {metrics_dict['Accuracy']:.4f}")
    print(f"AUROC (macro): {metrics_dict['AUROC_macro']:.4f}")
    print(f"AUPRC (macro): {metrics_dict['AUPRC_macro']:.4f}")
    print(f"F1 Score (macro): {metrics_dict['F1_macro']:.4f}")
