# from __future__ import division, print_function, unicode_literals

#%%
from datetime import date
# from tabnanny import verbose
import time
import glob
import os
import io
import pickle
from tkinter import Y
from PIL import Image

# from tensorflow import keras
# import tensorflow_addons as tfa
# from tensorflow.keras import backend as k
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import (
    Dense, Reshape,
    Conv1D, Conv2D, Conv1DTranspose, Conv2DTranspose,
    Concatenate, UpSampling1D, AveragePooling1D,
    Flatten, Dropout, Lambda, Add, Multiply
)
import soundfile as sf
import tensorflow as tf


import numpy as np
from tqdm import tqdm
import librosa


import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


DIM = 2
BLOCK_SIZE = 48000
SAMPLE_RATE = 48000
PATH_BASE = r"F:\sinwooyoo\python\hist"
PATH_SESS = os.path.join(
    PATH_BASE, str(date.today()),
    str(time.strftime("%H%M%S", time.localtime())))
PATH_FIGS = os.path.join(PATH_SESS, r"figs")
PATH_LOGS = os.path.join(PATH_SESS, r"logs")
PATH_CKPT = os.path.join(PATH_SESS, r"ckpt")
PATH_PKL = os.path.join(PATH_BASE, r"pkl")
os.makedirs(PATH_SESS, exist_ok=True)
os.makedirs(PATH_FIGS, exist_ok=True)
os.makedirs(PATH_PKL, exist_ok=True)
os.makedirs(PATH_CKPT, exist_ok=True)

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
    # lst_file = {k: v for k, v in sorted(lst_file.items(), key=lambda _x: _x[1])}
    lst_file = sorted(dct_file.items(), key=lambda _x: _x[1])

    wfs = np.empty((0, block_size), dtype=float)
    idx = 0
    _duration = 0
    _wfs = 0
    _offset = .0
    _duration = round(duration/n_files, 1)
    while _wfs < duration:
        if len(lst_file) == idx:
            idx = 0
            _offset += _duration

        if len(lst_file) == 0:
            break

        try:
            _wf, _ = librosa.load(
                lst_file[idx][0],
                sr=sr,
                mono=False,
                offset=_offset,
                duration=_duration,
                res_type='kaiser_best')
            _sec = _offset+(len(_wf)/sr) if len(_wf.shape)==1 else _offset+(_wf.shape[1]/sr)
            print(f"\tloaded:{os.path.basename(lst_file[idx][0]):s}, \
                {_offset:3.2f}s ~ {_sec:3.2f}s")

            # If it's stereo signal
            if np.array(_wf.shape)[0] == 2:
                _wf = (_wf[0, :] + _wf[1, :]) / 2
                _wf = (_wf - _wf.min()) / (_wf.max() - _wf.min()) + 1e-16
            else:
                _wf = (_wf - _wf.min()) / (_wf.max() - _wf.min()) + 1e-16

            # sd.play(_wf, sr)

            _len_read = round(_wf.shape[0]/sr, 1)
            if _len_read < int(_duration):
                lst_file.pop(idx)
            else:
                idx += 1
            n_blocks = _wf.shape[0]//block_size
            _wf = _wf[:n_blocks*block_size]

            _wf = _wf.reshape(_wf.shape[0]//block_size, block_size)
            wfs = np.vstack((wfs, _wf))
            _wfs += _len_read

        except IndexError as _e:
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

    # return wfs_class, {"class_type":class_type, "len_total":len_total,\
    # "n_files":n_files, "power_wfs":np.power(wfs_in_dir, 2).mean()}
    return wfs


def save_load(val=None, opt="load"):
    try:
        if opt == "save":
            loc = os.path.join(PATH_SESS, "dataset.pkl")
            file = open(loc, 'wb')
            pickle.dump(val, file)
            file.close()
            print(f"data saved:{loc}")

        elif opt == "load":
            file = open(val, 'rb')
            dataset = pickle.load(file)
            file.close()
            return dataset
        else:
            print('Invalid saveLoad option')
    except ValueError:
        return None


def build_data(path=None, block_size=2400, split=.8, dim=1):

    sample_rate = SAMPLE_RATE

    try:
        with open(PATH_PKL+r"\features.pkl", 'rb') as f:
            features = pickle.load(f)
        with open(PATH_PKL+r"\labels.pkl", 'rb') as f:
            labels = pickle.load(f)

        np.random.shuffle(features)
        n_samples = features.shape[0]
        idx = int(n_samples*split)
        if dim==1:
            print(f"total: {features.shape[0]*features.shape[1]/sample_rate} seconds")
            print(f"{n_samples} sample records, splitted: train-1~{idx}, test-{idx}~{n_samples}")
        elif dim==2:
            print(f"total: {features.shape[0]}x{features.shape[1]}x{features.shape[2]}, \
                {features.shape[0]*features.shape[1]*features.shape[2]/sample_rate} seconds.")
            print(f"{n_samples} sample records, splitted: train-1~{idx}, test-{idx}~{n_samples}")

        return features[:idx], features[idx:], labels[:idx], labels[idx:]
    except Exception as _e:
        print("issue from reading existed file:{_e}, building new one...")
        
        if path is None:
            path_source = r"F:\sinwooyoo\data\input\waves\drone_wav_1"
        else:
            pass


        print("Reading data list...")
        lst_class = [s for s in os.listdir(path_source) \
            if os.path.isdir(os.path.join(path_source,s))]
        if len(lst_class)==0:
            lst_class = [path_source]
        if dim==1:
            features = np.empty((0, int(block_size)))
        elif dim==2:
            features = np.empty((0, 240, 200))
        labels = np.empty((0,1))
        # labels_one_hot = np.empty((0, len(lst_class)))

        for id_name in tqdm(lst_class, desc="Loading wav files..."):
            paths_wav = glob.glob(os.path.join(path_source, id_name) + r'\*.wav')
            if len(paths_wav) == 0:
                pass
            else:
                if dim==1:
                    feature = np.vstack(load_wav(paths_wav, block_size=BLOCK_SIZE, duration=60))
                    label = np.repeat(lst_class.index(id_name), \
                        repeats=len(feature)).reshape(len(feature), 1)
                    # label_one_hot = one_hot(label, feature.shape[0], len(lst_class))

                    features = np.vstack((feature, features))
                    labels = np.vstack((label, labels))
                    # labels_one_hot = np.row_stack((label_one_hot, labels_one_hot))
                elif dim==2:
                    feature = load_wav(paths_wav, block_size=BLOCK_SIZE, duration=60)\
                        .reshape((-1, 240, 200))
                    label = np.repeat(lst_class.index(id_name), repeats=len(feature))\
                        .reshape(len(feature), 1)

                    features = np.vstack((feature, features))
                    labels = np.vstack((label, labels))

        with open(PATH_PKL+r"\features.pkl", 'wb') as f:
            pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)
        with open(PATH_PKL+r"\labels.pkl", 'wb') as f:
            pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)

        ## need to fix for labels permutation
        np.random.shuffle(features)
        n_samples = features.shape[0]
        idx = int(n_samples*split)
        if dim==1:
            print(f"total: {features.shape[0]*features.shape[1]/sample_rate} seconds")
            print(f"{n_samples} sample records, splitted: train-1~{idx}, test-{idx}~{n_samples}")
        elif dim==2:
            print(f"total: {features.shape[0]}x{features.shape[1]}x{features.shape[2]}, \
                {features.shape[0]*feature.shape[1]*feature.shape[2]/sample_rate} seconds.")
            print(f"{n_samples} sample records, splitted: train-1~{idx}, test-{idx}~{n_samples}")

        return features[:idx], features[idx:], labels[:idx], labels[idx:]

X_train, X_test, _, _ = build_data(block_size=BLOCK_SIZE, dim=2)


# %%
tf.random.set_seed(777)

LEARNING_RATE = 1e-4

lst_latent = [2, 8, 32]
lst_filter_enc_1d = [128, 128, 128, 128, 128]
lst_kernel_enc_1d = [64, 128, 256, 512, 1024]
lst_stride_enc_1d = [1, 2, 4, 8, 16]
lst_dilation_enc_1d = [1, 2, 4, 8, 16]
lst_filter_dec_1d = lst_filter_enc_1d[::-1]
lst_kernel_dec_1d = lst_kernel_enc_1d[::-1]
lst_stride_dec_1d = lst_stride_enc_1d[:0:-1]
lst_dilation_dec_1d = lst_dilation_enc_1d[:0:-1]

lst_filter_enc_2d = [128, 128, 128]
lst_kernel_enc_2d = [4, 4, 4]
lst_stride_enc_2d = [1, 2, 4]
lst_dilation_enc_2d = [1, 2, 4]

lst_filter_dec_2d = lst_filter_enc_2d
lst_kernel_dec_2d = lst_kernel_enc_2d[::-1]
lst_stride_dec_2d = lst_stride_enc_2d[::-1]
lst_dilation_dec_2d = lst_dilation_enc_2d[::-1]


if DIM == 1:
    inputs = Input(shape=(BLOCK_SIZE, 1), name='input')
elif DIM == 2:
    inputs = Input(shape=(240, 200, 1), name='input')

def _encoder(_x, dim=1):
    if dim==1:
        for i, _ in enumerate(lst_stride_enc_1d):
            _x = Conv1D(
                lst_filter_enc_1d[i],
                lst_kernel_enc_1d[i],
                dilation=lst_stride_enc_2d[i],
                padding='causal',
                activation='relu')(_x)
    elif dim==2:
        for i, _ in enumerate(lst_stride_enc_2d):
            _x = Conv2D(
                filters=lst_filter_enc_2d[i],
                kernel_size=lst_kernel_enc_2d[i],
                strides=lst_stride_enc_2d[i],
                padding='same',
                activation='relu')(_x)
    return _x


def _decoder(_x, dim=1):
    if dim==1:
        for i, _ in enumerate(lst_stride_dec_1d):
            _x = Conv1DTranspose(
                lst_filter_dec_1d[i],
                lst_kernel_dec_1d[i],
                dilation=lst_stride_dec_2d[i],
                padding='same',
                activation='relu')(_x)
    elif dim==2:
        for i, _ in enumerate(lst_stride_dec_2d):
            _x = Conv2DTranspose(
                lst_filter_dec_2d[i],
                lst_kernel_enc_2d[i],
                strides=lst_stride_dec_2d[i],
                padding='same',
                activation='relu')(_x)
    return _x

encoder1 = _encoder(inputs, dim=2)
encoder2 = _encoder(inputs, dim=2)
encoder3 = _encoder(inputs, dim=2)

decoder1 = _decoder(encoder1, dim=2)
decoder2 = _decoder(encoder2, dim=2)
decoder3 = _decoder(encoder3, dim=2)

merged = Add()([decoder1*.3, decoder2*.3, decoder3*.3])
# scaled = Multiply()([tf.constant(np.ones((BLOCK_SIZE,5))*.333), merged])
outputs = Dense(1)(merged)
# outputs = tf.keras.layers.BatchNormalization()(merged)

# _x = tf.data.Dataset.from_tensor_slices(_x).shuffle(200).batch(20)

model = Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
model.summary()
plot_model(model, to_file=str(os.path.join(PATH_FIGS, "model.png")), show_shapes=True)

class CustomCallback(tf.keras.callbacks.Callback):

    writer = tf.summary.create_file_writer(PATH_FIGS)

    def __init__(self, ratio=0.0, patience=5, verbose=2, x_test=X_test):
        self.x_test = x_test
        self.ratio = ratio
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater

    def on_train_begin(self, logs=None):
        self.wait = 0  # Allow instances to be re-used

    def make_plot(self, _y=None, title=""):
        if _y is None:
            print("No data")
            return
        plt.ioff()
        fig = plt.figure(figsize=(10,5))
        _x = np.linspace(0,BLOCK_SIZE/SAMPLE_RATE, SAMPLE_RATE)
        _ax = fig.add_axes([.1, .15, .8, .8])
        _ax.grid(True, linestyle='-.', linewidth=1)
        _ax.set(xlabel='Time (ms)', ylabel='Energy (normalized)', title=title)
        _ax.plot(_x, _y.reshape(SAMPLE_RATE,-1), linewidth=1)
        # _ax.tick_params(axis='x', direction='in', length=3, pad=2, labelsize=14, top=True)
        # _ax.tick_params(axis='y', direction='inout', length=10, pad=2, labelsize=12, width=2)
        plot = io.BytesIO()
        _ax.figure.savefig(plot, format='png')
        plot.seek(0)
        plot = tf.image.decode_png(plot.getvalue(), channels=4)
        plot = tf.expand_dims(plot, 0)

        with tf.summary.create_file_writer(PATH_LOGS).as_default():
            tf.summary.image(title, plot, step=self.params.get('steps'))

        plt.close("all")


    def make_image(self, _y=None, title=""):
        """
        Convert an numpy representation image to Image protobuf.
        Copied from https://github.com/lanpa/tensorboard-pytorch/
        """

        if _y is None:
            print("No data")
            return

        image = Image.fromarray((_y*255).astype(np.uint8))

        output = io.BytesIO()
        image.save(output, format='png')
        output.close()

        with tf.summary.create_file_writer(PATH_LOGS).as_default():
            tf.summary.image(title, _y.reshape((-1,240,200,1)), step=self.params.get('steps'))
    
    
    def on_epoch_end(self, epoch, logs=None):
        current_val = logs.get('val_loss')
        current_train = logs.get('loss')
        if current_val is None:
            warnings.warn(f'Early stopping requires {self.monitor:s} available!', RuntimeWarning)
        # If ratio current_loss / current_val_loss > self.ratio
        if self.monitor_op(np.divide(current_train,current_val),self.ratio):
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.wait += 1
        idx = np.random.randint(self.x_test.shape[0])
        _x = self.x_test[idx]
        if DIM==1:
            _x_gen = self.model.predict(_x.reshape((-1, BLOCK_SIZE, 1)))
            _x_gen = np.squeeze(_x_gen)
            _x_gen /= np.max(_x_gen)
            _delta = _x - _x_gen
            
            sf.write(os.path.join(PATH_LOGS,str(epoch)+"_x.wav"), _x, 48000)
            sf.write(os.path.join(PATH_LOGS,str(epoch)+"_y.wav"), _x_gen, 48000)
            
            self.make_plot(_y=_x, title=str(epoch)+"#x_test")
            self.make_plot(_y=_x_gen, title=str(epoch)+"#predicted")
            self.make_plot(_y=_delta, title=str(epoch)+"#x_test")

        elif DIM==2:
            _x_gen = self.model.predict(_x.reshape((-1, 240, 200, 1)))
            _x_gen = _x_gen.reshape(240, 200)
            _delta = _x - _x_gen
            # _x_gen = np.squeeze(_x_gen)
            # _x_gen /= np.max(_x_gen)
            sf.write(os.path.join(PATH_LOGS,str(epoch)+"_x.wav"), _x.reshape((SAMPLE_RATE, -1)), 48000)
            sf.write(os.path.join(PATH_LOGS,str(epoch)+"_y.wav"), _x_gen.reshape((SAMPLE_RATE, -1)), 48000)
            sf.write(os.path.join(PATH_LOGS,str(epoch)+"_d.wav"), _delta.reshape((SAMPLE_RATE, -1)), 48000)

            self.make_image(_y=_x, title=str(epoch)+"#x_test(240x200)")
            self.make_image(_y=_x_gen, title=str(epoch)+"#predicted(240x200)")
            self.make_image(_y=_delta, title=str(epoch)+"#delta(240x200)")
            
            self.make_plot(_y=_x, title=str(epoch)+"#x_test")
            self.make_plot(_y=_x_gen, title=str(epoch)+"#predicted")
            self.make_plot(_y=_delta, title=str(epoch)+"#x_test")


    def on_train_end(self, logs=None):
        if (self.stopped_epoch > 0) and (self.verbose > 0):
            print(f'Epoch {self.stopped_epoch:05d}: early stopping')


tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=PATH_LOGS,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    write_steps_per_second=False,
    update_freq='epoch',
    profile_batch=0,
    embeddings_freq=1,
    embeddings_metadata=None)
reduceLR = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=.8,
    patience=4)
cs_callback = CustomCallback(ratio=.9, patience=10, verbose=2, x_test=X_test)


model.fit(
    X_train, X_train,
    batch_size=20,
    epochs=1000,
    validation_split=.4,
    shuffle=True,
    verbose=2,
    use_multiprocessing=True,
    callbacks=[tb_callback, cs_callback, reduceLR])



# %%
