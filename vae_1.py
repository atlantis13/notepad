from __future__ import division, print_function, unicode_literals

#%%
from datetime import date


import time
import glob
import os
import numpy as np

from tqdm import tqdm
import librosa

import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

SHAPE_INPUT = 4800
PATH_BASE = r"F:\sinwooyoo\python\hist"
PATH_SESS = os.path.join(
    PATH_BASE, str(date.today()),
    str(time.strftime("%H%M%S", time.localtime())))
PATH_FIGS = os.path.join(PATH_SESS, r"figs")
PATH_LOGS = os.path.join(PATH_SESS, r"logs")
PATH_CKPT = os.path.join(PATH_SESS, r"ckpt")
os.makedirs(PATH_SESS, exist_ok=True)
os.makedirs(PATH_FIGS, exist_ok=True)


def reset_graph(seed=37):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PATH_FIGS, fig_id + ".svg")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='svg', dpi=600)


def load_wav(lst_file, block_size=2400, sr=48000, duration=10):
    # timestamp, start
    start_time = time.time()

    # listup files from given dir_wav
    n_files = len(lst_file)
    lst_size = [os.path.getsize(s) for s in lst_file]
    dct_file = {k: v for k, v in zip(lst_file, lst_size)}
    # lst_file = {k: v for k, v in sorted(lst_file.items(), key=lambda x: x[1])}
    lst_file = sorted(dct_file.items(), key=lambda x: x[1])

    wfs = np.empty((0, block_size), dtype=float)
    idx = 0
    _duration = 0
    _wfs = 0
    _offset = .0
    _duration = round(duration/n_files, 1)
    
    while(_wfs < duration):
        if(len(lst_file) == idx):
            idx = 0
            _offset += _duration

        if (len(lst_file) == 0):
            break

        try:
            _wf, _ = librosa.load(
                lst_file[idx][0],
                sr=sr,
                mono=False,
                offset=_offset,
                duration=_duration,
                res_type='kaiser_best')
            print(f"\tloaded:{0}, {_offset:3.3f}s ~ {_offset+(len(_wf)/sr):3.3f}s".format(
                os.path.basename(lst_file[idx][0])))

            # If it's stereo signal
            if(np.array(_wf.shape)[0] == 2):
                _wf = (_wf[0, :] + _wf[1, :]) / 2
                _wf = (_wf - _wf.min()) / (_wf.max() - _wf.min()) + 1e-16
            else:
                _wf = (_wf - _wf.min()) / (_wf.max() - _wf.min()) + 1e-16

            # sd.play(_wf, sr)

            _len_read = round(_wf.shape[0]/sr, 1)
            if(_len_read < int(_duration)):
                lst_file.pop(idx)
            else:
                idx += 1
            n_blocks = _wf.shape[0]//block_size
            _wf = _wf[:n_blocks*block_size]

            _wf = _wf.reshape(_wf.shape[0]//block_size, block_size)
            wfs = np.vstack((wfs, _wf))
            _wfs += _len_read

        except Exception as _e:
            print("\t", _e, f": file, {lst_file[idx]}")
            lst_file.pop(idx)
    _n_frame = int(sr/block_size*duration)
    if wfs.shape[0] > _n_frame:
        wfs = wfs[1:_n_frame+1, :]
    else:
        pass

    print(f"read in this class: {(wfs.shape[0]*wfs.shape[1]/sr):2.1f} seconds")

    end_time = time.time()
    print(f"took {end_time-start_time:.1f} seconds...\n")

    # return wfs_class, {"class_type":class_type, "len_total":len_total, "n_files":n_files, "power_wfs":np.power(wfs_in_dir, 2).mean()}
    return wfs




def build_data(path=None, block_size=2400, split=.9):
    if path==None:
        path_source = r"F:\sinwooyoo\data\input\waves\drone_wav_1\01-bebop2"
    else:
        pass


    print("Reading data list...")
    lst_class = [s for s in os.listdir(path_source) if os.path.isdir(os.path.join(path_source,s))]
    if len(lst_class)==0:
        lst_class = [path_source]
    features = np.empty((0, int(block_size)))
    labels = np.empty((0,1))
    labels_one_hot = np.empty((0, len(lst_class)))

    for id_name in tqdm(lst_class, desc="Loading wav files..."):
        paths_wav = glob.glob(os.path.join(path_source, id_name) + r'\*.wav')
        feature = np.vstack(load_wav(paths_wav, block_size=SHAPE_INPUT, duration=10/len(lst_class)))
        label = np.repeat(lst_class.index(id_name), repeats=len(feature)).reshape(len(feature), 1)
        # label_one_hot = one_hot(label, feature.shape[0], len(lst_class))

        features = np.vstack((feature, features))
        labels = np.vstack((label, labels))
        # labels_one_hot = np.row_stack((label_one_hot, labels_one_hot))
    np.random.shuffle(features)
    n_samples = features.shape[0]
    idx = int(n_samples*split)

    return feature[:idx], feature[idx:], labels[:idx], labels[idx:]

X_train, X_test, _, _ = build_data(block_size=SHAPE_INPUT)





# %%
import soundfile as sf
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as k
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import (
    Dense,
    Reshape,
    Conv1D,
    Conv1DTranspose,
    Concatenate,
    UpSampling1D,
    AveragePooling1D,
    Flatten,
    Dropout,
    Lambda,
    Add
)

tf.random.set_seed(777)


lst_latent = [2, 8, 32]
lst_filter_enc = [300, 300, 300, 300, 600, 1]
lst_kernel_enc = [2, 3, 5, 9, 17, 33]
lst_stride = [1, 1, 1, 1, 1, 1]
lst_dilations = [1, 2, 4, 8, 16, 32]

lst_filter_dec = lst_filter_enc[::-1]
lst_kernel_dec = [s+v-1 for s, v in zip(lst_kernel_enc[::-1], lst_stride[::-1])]
lst_stride_rev = lst_stride[::-1]


inputs = Input(shape=(SHAPE_INPUT, 1), name='input')

def _encoder(_x):
    for i, _ in enumerate(lst_filter_enc):
        _x = Conv1D(
            lst_filter_enc[i],
            lst_kernel_enc[i],
            strides=lst_stride[i],
            dilation_rate=lst_dilations[i],
            padding='valid',
            activation='LeakyReLU')(_x)
        _x = Dropout(rate=.2)(_x)
    return _x


def _decoder(_x):
    for i, _ in enumerate(lst_filter_dec):
        _x = tf.keras.layers.Conv1DTranspose(
            lst_filter_dec[i],
            lst_kernel_dec[i],
            strides=lst_stride_rev[i],
            dilation_rate=lst_dilations[i],
            padding='valid',
            activation='tanh')(_x)
    return _x

encoder1 = _encoder(inputs)
encoder2 = _encoder(inputs)
encoder3 = _encoder(inputs)
decoder1 = _decoder(encoder1)
decoder2 = _decoder(encoder2)
decoder3 = _decoder(encoder3)

merged = Add()([decoder1, decoder2, decoder3])
outputs = Dense(1)(merged)
# outputs = tf.keras.layers.BatchNormalization()(merged)

# X = tf.data.Dataset.from_tensor_slices(X).shuffle(200).batch(20)

model = Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=10e-3),
    loss='mse')
model.summary()
plot_model(
    model,
    to_file=str(os.path.join(PATH_FIGS, "model.png")), 
    show_shapes=True)

tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=PATH_LOGS,
    histogram_freq=1,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq='epoch',
    profile_batch=0,
    embeddings_freq=1,
    embeddings_metadata=None)

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(PATH_CKPT)

class CustomEarlyStopping(tf.keras.callbacks.EarlyStopping):
    def __init__(self, ratio=0.0,
                 patience=0, verbose=0):
        super(tf.keras.callbacks.EarlyStopping, self).__init__()

        self.ratio = ratio
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater

    def on_train_begin(self, logs=None):
        self.wait = 0  # Allow instances to be re-used

    def on_epoch_end(self, epoch, logs=None):
        current_val = logs.get('val_loss')
        current_train = logs.get('loss')
        if current_val is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)
        # If ratio current_loss / current_val_loss > self.ratio
        if self.monitor_op(np.divide(current_train,current_val),self.ratio):
            if(current_val>.0):
                self.wait = 0
                pass
            pass
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                pass
            self.wait += 1

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))
            pass

es_callback = CustomEarlyStopping(ratio=.9, patience=2, verbose=1)


class PerformancePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, model_name):
        self.x_test = x_test
        self.model_name = model_name
        pass

    def on_epoch_end(self, epoch, logs={}):
        idx = np.random.randint(self.x_test.shape[0])
        x_gen = self.model.predict(self.x_test[idx].reshape((-1, SHAPE_INPUT, 1)))
        x_gen = np.squeeze(x_gen)
        x_gen /= np.max(x_gen)
        # x_gen = tf.linalg.normalize(x_gen).numpy()
        x = self.x_test[idx]
        delta = x - x_gen
        sf.write(os.path.join(PATH_FIGS,str(epoch)+"x.wav"), x, 48000)
        sf.write(os.path.join(PATH_FIGS,str(epoch)+"y.wav"), x_gen, 48000)

        fig = plt.figure(figsize=(20,5))
        ax0 = fig.add_subplot(211)
        ax0.plot(x)
        ax0.plot(x_gen)
        ax1 = fig.add_subplot(212)
        ax1.plot(delta)

        plt.tight_layout()
        fig.figure.savefig(os.path.join(PATH_FIGS,str(epoch))+".png", format='png', dpi=600)
        plt.close('all')

        # delta = x_gen - self.x_test[:20].reshape((x_gen.shape))
        # print("\n delta:{0}".format(delta))

plt_callback = PerformancePlotCallback(X_test, "generated_signal")

model.fit(
    X_train, X_train, 
    batch_size=50,
    epochs=300,
    validation_split=.5,
    shuffle=True,
    verbose=1,
    callbacks=[tb_callback, plt_callback])



# %%
