#!/usr/bin/env python
# coding: utf-8

# ## Basic Steps
# 
# 1. Run a baseline training script to build a speech commands model.
# 2. Add in your custom word to the training and test/validation sets.
#    - Modify labels, shape of your output tensor in the model.
#    - Make sure that feature extractor for the model aligns with the feature extractor 
#      used in the arduino code.
# 3. Re-train model. => TF Model using floating-point numbers, that recognizes Google word and custom word.
# 4. Quantize the model and convert to TFlite. => keyword_model.tflite file
# 5. Convert tflite to .c file, using xxd => model_data.cc
# 6. Replace contents of existing micro_features_model.cpp with output of xxd.
# 
# All of the above steps are done in this notebook for the commands 'left', 'right'.
# 
# 7. In micro_speech.ino, modify micro_op_resolver (around line 80) to add any necessary operations (DIFF_FROM_LECTURE)
# 8. In micro_features_model_settings.h, modify kSilenceIndex and kUnknownIndex, depending on 
# where you have them in commands.  
#   - Commands = ['left', 'right', '_silence', '_unknown'] => kSilenceIndex=2, kUnknownIndex=3
# 9. In micro_features_model_settings.cpp, modify kCategoryLabels to correspond to commands in this script.
# 10. In micro_features_micro_model_settings.h, set kFeatureSliceDurationMs, kFeatureSliceStrideMs to match what is passed to microfrontend as window_size, window_step, respectively.
# 11. Rebuild Arduino program, run it, recognize the two target words.
# 12. Experiment with model architecture, training parameters/methods, augmentation, more data-gathering, etc.
# 

# You can download the data set with this command line (or just a browser):
# 
# `wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz `

# In[1]:


def in_notebook():
  """
  Test if we are in a python script vs jupyter notebook.
  Returns True if called in a jupyter notebook, false if called in a standard python file
  taken from https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
  """
  import __main__ as main
  return not hasattr(main, '__file__')


# In[2]:


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import Input, layers
from tensorflow.keras import models
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
print(tf.__version__)

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if in_notebook():
  from tqdm.notebook import tqdm
else:
  from tqdm import tqdm 

import csv
import os, glob, pathlib, time, random

from datetime import datetime as dt
import platform


# In[3]:


# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


# In[4]:


APPLE_SILICON = platform.processor() == 'arm'
# Define range of 16-bit integers
i16min = -2**15
i16max = 2**15-1
fsamp = 16000 # sampling rate
wave_length_ms = 1000 # 1000 => 1 sec of audio
wave_length_samps = int(wave_length_ms*fsamp/1000)

# you can change these next three
window_size_ms=64 
window_step_ms=48
num_filters = 32
batch_size = 32
# Training-only: small random time shift via symmetric pad + crop (~30 ms at 16 kHz).
time_shift_max_samples = 480
use_microfrontend = True # recommended, but you can use another feature extractor if you like

## uncomment exactly one of these 
dataset = 'mini-speech'
# dataset = 'full-speech-files' # use the full speech commands stored as files 

commands = ['backward', 'spottieottie'] ## Change this line for your custom keywords

# limit the instances of each command in the training set to simulate limited data
limit_positive_samples = False
# When limit_positive_samples is True, at most this many train_files per command are kept.
max_wavs_0 = 100
max_wavs_1 = 100

silence_str = "_silence"  # label for <no speech detected>
unknown_str = "_unknown"  # label for <speech detected but not one of the target words>
EPOCHS = 25

# TPR/FPR reporting table: target is both classes above TPR min and below FPR max
TARGET_TPR_MIN = 0.75
TARGET_FPR_MAX = 0.2
# Per-run file: tpr_fpr_runs_{_fig_stem}.csv (set after training with same stem as figures)

try:
    _SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = pathlib.Path.cwd()

print(f"FFT window length = {int(window_size_ms * fsamp / 1000)}")

might_be = {True:"IS", False:"IS NOT"} # useful for formatting conditional sentences


# Apply the frontend to an example signal.

# In[44]:


if dataset == 'mini-speech':
  data_root = pathlib.Path(os.getenv("HOME")) / "data"
  data_root.mkdir(parents=True, exist_ok=True)
  # `training/`'s parent / `dataset` — same tree as record_audio / preprocess; not cwd-relative.
  data_dir = _SCRIPT_DIR.parent / "dataset"
  if not data_dir.exists():
    tf.keras.utils.get_file(
        "mini_speech_commands.zip",
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True,
        cache_dir=str(data_root),
        cache_subdir=".",
    )
  # Keras may extract to mini_speech_commands_extracted/mini_speech_commands
  if not data_dir.exists():
    alt = data_root / "mini_speech_commands_extracted" / "mini_speech_commands"
    if alt.is_dir():
      data_dir = alt
  # commands = np.array(tf.io.gfile.listdir(str(data_dir))) # if you want to use all the command words
  # commands = commands[commands != 'README.md']
elif dataset == 'full-speech-files':
  data_dir = pathlib.Path(os.path.join(os.getenv("HOME"), 'data', 'speech_commands_v0.02'))
  if not data_dir.exists():
    raise RuntimeError("Either download the speech commands files to {data_dir} or change this code to where you have them")
else:
  raise RuntimeError('dataset should either be "mini-speech" or "full-speech-files"')


# In[7]:


label_list = commands.copy()
label_list.insert(0, silence_str)
label_list.insert(1, unknown_str)
print('label_list:', label_list)


# In[8]:


if dataset in ['mini-speech', 'full-speech-files']:
    # filenames = tf.io.gfile.glob(str(data_dir) + '/*/*.wav') 
    # filenames = tf.io.gfile.glob(str(data_dir) + os.sep + '*' + '/' + '*.wav') 
    filenames = glob.glob(os.path.join(str(data_dir), '*', '*.wav')) 
  
    # with the next commented-out line, you can choose only files for words in label_list
    # filenames = tf.concat([tf.io.gfile.glob(str(data_dir) + '/' + cmd + '/*') for cmd in label_list], 0)
    random.shuffle(filenames)
    num_samples = len(filenames)
    print('Number of total examples:', num_samples)
    # print('Number of examples per label:',
    #       len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
    print('Example file tensor:', filenames[0])


# In[9]:


# Not really necessary, but just look at a few of the files to make sure that 
# they're the correct files, shuffled, etc.

# for i in range(10):
#     print(filenames[i].numpy().decode('utf8'))


# In[10]:


if dataset == 'mini-speech':
  print('Using mini-speech')
  num_train_files = int(0.8*num_samples) 
  num_val_files = int(0.1*num_samples) 
  num_test_files = num_samples - num_train_files - num_val_files
  train_files = filenames[:num_train_files]
  val_files = filenames[num_train_files: num_train_files + num_val_files]
  test_files = filenames[-num_test_files:]
elif dataset == 'full-speech-files':  
  # the full speech-commands set lists which files are to be used
  # as test and validation data; train with everything else
  fname_val_files = os.path.join(data_dir, 'validation_list.txt')
  with open(fname_val_files) as fpi_val:
    val_files = fpi_val.read().splitlines()
  # validation_list.txt only lists partial paths
  val_files = [os.path.join(data_dir, fn) for fn in val_files]
  fname_test_files = os.path.join(data_dir, 'testing_list.txt')
  with open(fname_test_files) as fpi_tst:
    test_files = fpi_tst.read().splitlines()
  # testing_list.txt only lists partial paths
  test_files = [os.path.join(data_dir, fn).rstrip() for fn in test_files]    
    
  if os.sep != '/': 
    # the files validation_list.txt and testing_list.txt use '/' as path separator
    # if we're on a windows machine, replace the '/' with the correct separator
    val_files = [fn.replace('/', os.sep) for fn in val_files]
    test_files = [fn.replace('/', os.sep) for fn in test_files] 

  # convert the TF tensor filenames into an array of strings so we can use basic python constructs
  # train_files = [f.decode('utf8') for f in filenames.numpy()]

  
  # don't train with the _background_noise_ files; exclude when directory name starts with '_'
  train_files = [f for f in filenames if f.split(os.sep)[-2][0] != '_']
  # validation and test files are listed explicitly in *_list.txt; train with everything else
  train_files = list(set(train_files) - set(test_files) - set(val_files))
   
elif dataset == 'full-speech-ds':  
    print("Using full-speech-ds. This is in progress.  Good luck!")
else:
  raise ValueError("dataset must be either full-speech-files, full-speech-ds or mini-speech")
print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))


# In[11]:


## Remove some of the target words if we're experimenting with limited data

if limit_positive_samples:
  if dataset not in ['mini-speech', 'full-speech-files']:
    raise RuntimeError("Right now, limit_positive_samples is only implemented if drawing the data from files")
  num_files_cmd0 = 0
  num_files_cmd1 = 0
  # elements of train_files look like this:
  # '/path/to/data/speech_commands_0_2_root/right/196e84b7_nohash_0.wav'
  # so if we split on '/' (or '\' in windows), the 2nd to last element is the label
  for idx,f in enumerate(train_files):
    if f.split(os.sep)[-2] == commands[0]:
      if num_files_cmd0 >= max_wavs_0:
        train_files.pop(idx)
      else:
        num_files_cmd0 += 1
    elif f.split(os.sep)[-2] == commands[1]:
      if num_files_cmd1 >= max_wavs_1:
        train_files.pop(idx)
      else:
        num_files_cmd1 += 1


# In[12]:


print(train_files[:5])
print(val_files[:5])
print(test_files[:5])


# In[13]:


def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)


# In[14]:


def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  in_set = tf.reduce_any(parts[-2] == label_list)
  label = tf.cond(in_set, lambda: parts[-2], lambda: tf.constant(unknown_str))
  # print(f"parts[-2] = {parts[-2]}, in_set = {in_set}, label = {label}")
  # Note: You'll use indexing here instead of tuple unpacking to enable this 
  # to work in a TensorFlow graph.
  return  label # parts[-2]


# In[15]:


def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label


# In[16]:


def get_spectrogram(waveform):
  # Truncate longer clips, then pad shorter ones so all clips are wave_length_samps.
  waveform = waveform[:wave_length_samps]
  pad_len = wave_length_samps - tf.shape(waveform)[0]
  zero_padding = tf.zeros([pad_len], dtype=tf.int16)
  waveform = tf.cast(0.5*waveform*(i16max-i16min), tf.int16)  # scale float [-1,+1]=>INT16
  equal_length = tf.concat([waveform, zero_padding], 0)
  ## Make sure these labels correspond to those used in micro_features_micro_features_generator.cpp
  spectrogram = frontend_op.audio_microfrontend(equal_length, sample_rate=fsamp, num_channels=num_filters,
                                    window_size=window_size_ms, window_step=window_step_ms)
  return spectrogram


# Function to convert each waveform in a set into a spectrogram, then convert those
# back into a dataset using `from_tensor_slices`.  (We should be able to use 
# `wav_ds.map(get_spectrogram_and_label_id)`, but there is a problem with that process).
#    

# In[17]:


def create_silence_dataset(num_waves, samples_per_wave, rms_noise_range=[0.01,0.2], silent_label=silence_str):
    # create num_waves waveforms of white gaussian noise, with rms level drawn from rms_noise_range
    # to act as the "silence" dataset
    rng = np.random.default_rng()
    rms_noise_levels = rng.uniform(low=rms_noise_range[0], high=rms_noise_range[1], size=num_waves)
    rand_waves = np.zeros((num_waves, samples_per_wave), dtype=np.float32) # pre-allocate memory
    for i in range(num_waves):
        rand_waves[i,:] = rms_noise_levels[i]*rng.standard_normal(samples_per_wave)
    labels = [silent_label]*num_waves
    return tf.data.Dataset.from_tensor_slices((rand_waves, labels))  


# In[18]:


def wavds2specds(waveform_ds, verbose=True):
  wav, label = next(waveform_ds.as_numpy_iterator())
  one_spec = get_spectrogram(wav)
  one_spec = tf.expand_dims(one_spec, axis=0)  # add a 'batch' dimension at the front
  one_spec = tf.expand_dims(one_spec, axis=-1) # add a singleton 'channel' dimension at the back    

  num_waves = 0 # count the waveforms so we can allocate the memory
  for wav, label in waveform_ds:
    num_waves += 1
  print(f"About to create spectrograms from {num_waves} waves")
  spec_shape = (num_waves,) + one_spec.shape[1:] 
  spec_grams = np.nan * np.zeros(spec_shape)  # allocate memory
  labels = np.nan * np.zeros(num_waves)
  idx = 0
  for wav, label in waveform_ds:    
    if verbose and idx % 250 == 0:
      print(f"\r {idx} wavs processed", end='')
    spectrogram = get_spectrogram(wav)
    # TF conv layer expect inputs structured as 4D (batch_size, height, width, channels)
    # the microfrontend returns 2D tensors (freq, time), so we need to 
    spectrogram = tf.expand_dims(spectrogram, axis=0)  # add a 'batch' dimension at the front
    spectrogram = tf.expand_dims(spectrogram, axis=-1) # add a singleton 'channel' dimension at the back
    spec_grams[idx, ...] = spectrogram
    new_label = label.numpy().decode('utf8')
    new_label_id = np.argmax(new_label == np.array(label_list))    
    labels[idx] = new_label_id # for numeric labels
    # labels.append(new_label) # for string labels
    idx += 1
  labels = np.array(labels, dtype=int)
  output_ds = tf.data.Dataset.from_tensor_slices((spec_grams, labels))  
  return output_ds


# In[19]:


AUTOTUNE = tf.data.experimental.AUTOTUNE
num_train_files = len(train_files)
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
train_ds = wavds2specds(waveform_ds)


# In[20]:


def copy_with_noise(ds_input, rms_level=0.25):
  rng = tf.random.Generator.from_seed(1234)
  wave_shape = tf.constant((wave_length_samps,))
  def add_noise(waveform, label):
    noise = rms_level*rng.normal(shape=wave_shape)
    waveform = waveform[:wave_length_samps]
    pad_len = wave_length_samps - tf.shape(waveform)[0]
    zero_padding = tf.zeros([pad_len], dtype=tf.float32)
    waveform = tf.concat([waveform, zero_padding], 0)    
    noisy_wave = waveform + noise
    return noisy_wave, label

  return ds_input.map(add_noise)


# In[21]:


def pad_16000(waveform, label):
    waveform = waveform[:wave_length_samps]
    pad_len = wave_length_samps - tf.shape(waveform)[0]
    zero_padding = tf.zeros([pad_len], dtype=tf.float32)
    waveform = tf.concat([waveform, zero_padding], 0)        
    return waveform, label


# In[22]:


def count_labels(dataset):
    counts = {}
    for sample in dataset: # sample will be a tuple: (input, label) or (input, label, weight)
        lbl = sample[1]
        if lbl.dtype == tf.string:
            label = lbl.numpy().decode('utf-8')
        else:
            label = lbl.numpy()
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    return counts


# In[23]:


def is_batched(ds):
    ## This is probably not very robust
    try:
        ds.unbatch()  # does not actually change ds. For that we would ds=ds.unbatch()
    except:
        return False # we'll assume that the error on unbatching is because the ds is not batched.
    else:
        return True  # if we were able to unbatch it then it must have been batched (??)


# In[24]:


def _make_time_shift_map_fn(max_shift_samples):
  """Pad/crop in time for a small random shift (use for training augmentation only)."""

  def _fn(waveform, label):
    orig_len = tf.shape(waveform)[0]
    pad = max_shift_samples
    w = tf.pad(waveform, [[pad, pad]], mode="CONSTANT")
    delta = tf.random.uniform([], 0, 2 * pad + 1, dtype=tf.int32)
    w = tf.slice(w, [delta], [orig_len])
    return w, label

  return _fn


# Collect what we did to generate the training dataset into a 
# function, so we can repeat with the validation and test sets.
def preprocess_dataset(
    files, num_silent=None, noisy_reps_of_known=None, time_shift_max_samples=0
):
  """
  preprocess_dataset(files, num_silent=None, noisy_reps_of_known=None)
  files -- list of files
  noisy_reps_of_known either None, or a list of rms noise levels
      For every target word in the data set, 1 copy will be created with each level 
      of noise added to it.  So [0.1, 0.2] will add 2x noisy copies of the target words 
  time_shift_max_samples -- if > 0, apply random time shift (pad/crop) on waveforms.
  """
  if num_silent is None:
    num_silent = int(0.2*len(files))+1
  print(f"Processing {len(files)} files")
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  waveform_ds = files_ds.map(get_waveform_and_label)
  if time_shift_max_samples and time_shift_max_samples > 0:
    waveform_ds = waveform_ds.map(
        _make_time_shift_map_fn(int(time_shift_max_samples)),
        num_parallel_calls=AUTOTUNE,
    )
  if noisy_reps_of_known is not None:
    # create a few copies of only the target words to balance the distribution
    # create a tmp dataset with only the target words
    ds_only_cmds = waveform_ds.filter(lambda w,l: tf.reduce_any(l == commands))
    for noise_level in noisy_reps_of_known:
       waveform_ds = waveform_ds.concatenate(copy_with_noise(ds_only_cmds, rms_level=noise_level))
  if num_silent > 0:
    silent_wave_ds = create_silence_dataset(num_silent, wave_length_samps, 
                                            rms_noise_range=[0.01,0.2], 
                                            silent_label=silence_str)
    waveform_ds = waveform_ds.concatenate(silent_wave_ds)
  print(f"Added {num_silent} silent wavs and ?? noisy wavs")
  num_waves = 0
  output_ds = wavds2specds(waveform_ds)
  return output_ds


# In[25]:


print(f"We have {len(train_files)}/{len(val_files)}/{len(test_files)} training/validation/test files")


# In[26]:


# train_files = train_files[0:10000] for a quick test run


# In[27]:


t0 = time.time()
# train_ds is already done
with tf.device('/CPU:0'): # needed on M1 mac
    train_ds = preprocess_dataset(
        train_files,
        noisy_reps_of_known=[0.05, 0.1, 0.15, 0.2],
        time_shift_max_samples=time_shift_max_samples,
    )

val_ds = preprocess_dataset(val_files)
test_ds_unbatched = preprocess_dataset(test_files)
train_ds_unbatched = train_ds
val_ds_unbatched = val_ds
t1 = time.time()
print(f"Took time: {t1-t0}")

print("Train label counts:", count_labels(train_ds_unbatched))
print("Val label counts:", count_labels(val_ds_unbatched))
print("Test label counts:", count_labels(test_ds_unbatched))

rep_samples = []
for x, y in train_ds_unbatched.take(100):
  rep_samples.append(x.numpy())
rep_samples = np.array(rep_samples, dtype=np.float32)
np.save(_SCRIPT_DIR / "rep_data.npy", rep_samples)
print("Saved representative data to", _SCRIPT_DIR / "rep_data.npy", rep_samples.shape)


# In[28]:


train_ds = train_ds.shuffle(int(len(train_files) * 1.2))


# In[29]:


if not is_batched(train_ds):
    train_ds = train_ds.batch(batch_size)
if not is_batched(val_ds):
    val_ds = val_ds.batch(batch_size)
test_ds_batched = test_ds_unbatched.batch(batch_size)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds_batched = test_ds_batched.cache().prefetch(AUTOTUNE)


# In[30]:


for spectrogram, _ in train_ds.take(1):
  # take(1) takes 1 *batch*, so we have to select the first 
  # spectrogram from it, hence the [0]
  input_shape = spectrogram.shape[1:]  
print('Input shape:', input_shape)
num_labels = len(label_list)


# In[31]:


def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_labels),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model


def compute_sparsity(model):
    """Fraction of kernel weights with |w| < 1% of max |w| in that layer (Conv / Dense only)."""
    sparsity_results = {}
    for layer in model.layers:
        if not hasattr(layer, "kernel"):
            continue
        w = layer.kernel.numpy()
        max_val = np.max(np.abs(w))
        if max_val == 0:
            sparsity_results[layer.name] = 1.0
            continue
        threshold = 0.01 * max_val
        near_zero = np.sum(np.abs(w) < threshold)
        sparsity_results[layer.name] = near_zero / w.size
    return sparsity_results


def print_sparsity(sparsity_dict, title):
    print(f"\n{title}")
    for layer, value in sparsity_dict.items():
        print(f"  {layer}: {value:.4f}")


# In[32]:


print('Input shape:', input_shape)
model = build_model(input_shape)
model.summary()            


# In[33]:


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
    ),
]
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

weight_sparsity_metrics = compute_sparsity(model)
print_sparsity(
    weight_sparsity_metrics,
    "Weight near-zero fraction (end of training; |w| < 1% of layer max |w|)",
)


# In[34]:


date_str = dt.now().strftime("%d%b%Y_%H%M%S").lower()
_lim = f"_lim{max_wavs_0}_{max_wavs_1}" if limit_positive_samples else ""
_fig_stem = f"{date_str}{_lim}"
training_curves_fpath = f"training_curves_{_fig_stem}.png"
confusion_matrix_fpath = f"confusion_matrix_{_fig_stem}.png"
tpr_fpr_runs_fpath = f"tpr_fpr_runs_{_fig_stem}.csv"
print(f"Completed training at {date_str}")
print(
    f"Figure outputs this run: {training_curves_fpath}, {confusion_matrix_fpath}, "
    f"{tpr_fpr_runs_fpath}"
)

final_train_acc = float(history.history["accuracy"][-1])
final_val_acc = float(history.history["val_accuracy"][-1])
print("\n=== Training (final epoch) ===")
print(f"Training accuracy (final):   {final_train_acc:.4f} ({100*final_train_acc:.2f}%)")
print(f"Validation accuracy (final): {final_val_acc:.4f} ({100*final_val_acc:.2f}%)")

metrics = history.history
fig, axes = plt.subplots(2, 1, figsize=(8, 6))
axes[0].semilogy(history.epoch, metrics['loss'], label='training')
axes[0].semilogy(history.epoch, metrics['val_loss'], label='validation')
axes[0].legend()
axes[0].set_ylabel('Loss')
axes[0].set_xlabel('Epoch')
axes[1].plot(history.epoch, metrics['accuracy'], label='training')
axes[1].plot(history.epoch, metrics['val_accuracy'], label='validation')
axes[1].legend()
axes[1].set_ylabel('Accuracy')
axes[1].set_xlabel('Epoch')
plt.tight_layout()
plt.savefig(training_curves_fpath, dpi=150, bbox_inches='tight')
print(f"Saved training curves to {training_curves_fpath}")
plt.close(fig)  # plt.show() would block here until the window is closed; skip so the rest of the script runs


# In[35]:


# ## Freezing a PASS run (deployment)
# This script overwrites the same kws_model.keras / kws_model.txt on each run. After a TARGET=PASS
# you care about, run:  python training/freeze_passing_run.py
# (from repo root) or:  python freeze_passing_run.py
# (from training/). That copies artifacts into training/frozen_runs/<...>/ and updates
# training/frozen_runs/CURRENT_CANDIDATE.* — use that frozen .keras for quantization, not
# a later ad-hoc training/ copy.

model_file_name = "kws_model.keras"
print(f"Saving model to {model_file_name}")
model.save(model_file_name, overwrite=True)
print("Evaluating on test set and building confusion matrix…")


# In[36]:


## Measure test-set accuracy manually and get values for confusion matrix
test_audio = []
test_labels = []
for audio, label in test_ds_unbatched:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

model_out = model.predict(test_audio)
y_pred = np.argmax(model_out, axis=1)
y_true = test_labels
np.save(_SCRIPT_DIR / "test_audio.npy", test_audio)
np.save(_SCRIPT_DIR / "y_true.npy", y_true)

pred_counts = {
    label_list[i]: int(np.sum(y_pred == i)) for i in range(len(label_list))
}
print("Predicted label counts:", pred_counts)

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.1%}')


# In[37]:


## Measure test-set accuracy with the keras built-in function
test_loss, test_acc = model.evaluate(test_ds_batched, verbose=2)
print(f"\n=== Test set (Keras evaluate) ===")
print(f"Test accuracy: {float(test_acc):.4f} ({100*float(test_acc):.2f}%)")


# In[38]:


confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
cm = confusion_mtx.numpy()
n_classes = len(label_list)
fig_cm = plt.figure(figsize=(6, 6))
sns.heatmap(cm, xticklabels=label_list, yticklabels=label_list,
            annot=True, fmt='g')
plt.gca().invert_yaxis()  # flip so origin is at bottom left
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.title('Confusion matrix (test set)')
plt.tight_layout()
plt.savefig(confusion_matrix_fpath, dpi=150, bbox_inches='tight')
print(f"Saved confusion matrix to {confusion_matrix_fpath}")
plt.close(fig_cm)


# In[ ]:





# In[39]:


tpr = np.nan * np.zeros(n_classes)
fpr = np.nan * np.zeros(n_classes)
for i in range(n_classes):
    row_sum = np.sum(cm[i, :])
    col_sum = np.sum(cm[:, i])
    total = np.sum(cm)
    tpr[i] = cm[i, i] / row_sum if row_sum > 0 else 0.0
    neg = total - row_sum
    fpr[i] = (col_sum - cm[i, i]) / neg if neg > 0 else 0.0
    print(f"TPR / FPR for '{label_list[i]:9}' = {tpr[i]:.4f} / {fpr[i]:.4f}")

idx_left = label_list.index(commands[0])
idx_right = label_list.index(commands[1])
tpr_left = float(tpr[idx_left])
fpr_left = float(fpr[idx_left])
tpr_right = float(tpr[idx_right])
fpr_right = float(fpr[idx_right])

target_ok = (
    tpr_left > TARGET_TPR_MIN
    and tpr_right > TARGET_TPR_MIN
    and fpr_left < TARGET_FPR_MAX
    and fpr_right < TARGET_FPR_MAX
)
target_str = "PASS" if target_ok else "FAIL"
n_train_samples = len(train_files)

print("\n=== Left / right metrics (every run) ===")
print(f"TPR_left  = {tpr_left:.4f}")
print(f"FPR_left  = {fpr_left:.4f}")
print(f"TPR_right = {tpr_right:.4f}")
print(f"FPR_right = {fpr_right:.4f}")

print("\n=== TPR/FPR table (this run) ===")
print(
    f"Target: TPR > {TARGET_TPR_MIN} and FPR < {TARGET_FPR_MAX} for both "
    f"{commands[0]} and {commands[1]}"
)
print(
    f"{'Samples':>10}  {'TPR_left':>10}  {'FPR_left':>10}  "
    f"{'TPR_right':>10}  {'FPR_right':>10}  {'TARGET':>8}"
)
print(
    f"{n_train_samples:10d}  {tpr_left:10.4f}  {fpr_left:10.4f}  "
    f"{tpr_right:10.4f}  {fpr_right:10.4f}  {target_str:>8}"
)

_runs_csv_path = pathlib.Path(tpr_fpr_runs_fpath)
with open(_runs_csv_path, "w", newline="") as _cf:
    _w = csv.writer(_cf)
    _w.writerow(
        [
            "timestamp",
            "samples_train",
            "max_wavs_0",
            "max_wavs_1",
            "TPR_left",
            "FPR_left",
            "TPR_right",
            "FPR_right",
            "TARGET",
        ]
    )
    _w.writerow(
        [
            date_str,
            n_train_samples,
            max_wavs_0 if limit_positive_samples else "",
            max_wavs_1 if limit_positive_samples else "",
            f"{tpr_left:.6f}",
            f"{fpr_left:.6f}",
            f"{tpr_right:.6f}",
            f"{fpr_right:.6f}",
            target_str,
        ]
    )
print(f"\nWrote this run to {_runs_csv_path.resolve()}")

_log_dir = _runs_csv_path.parent
_csv_logs = list(_log_dir.glob("tpr_fpr_runs_*.csv"))
_legacy = _log_dir / "tpr_fpr_runs.csv"
if _legacy.is_file():
    _csv_logs.append(_legacy)
_csv_logs = sorted(_csv_logs, key=lambda p: p.stat().st_mtime)
_first_pass = None
for _fp in _csv_logs:
    with open(_fp, newline="") as _cf:
        for _row in csv.DictReader(_cf):
            if _row.get("TARGET") == "PASS":
                _first_pass = (_fp, _row)
                break
    if _first_pass is not None:
        break
if _first_pass is not None:
    _fp, _r = _first_pass
    print(
        f"First PASS in log files (searching tpr_fpr_runs_*.csv): file={_fp.name}, "
        f"timestamp={_r.get('timestamp')}, Samples={_r.get('samples_train')}, "
        f"TPR_left={_r.get('TPR_left')}, FPR_left={_r.get('FPR_left')}, "
        f"TPR_right={_r.get('TPR_right')}, FPR_right={_r.get('FPR_right')}"
    )
else:
    print(
        f"No PASS yet in tpr_fpr_runs_*.csv under {_log_dir.resolve()} "
        f"(TPR > {TARGET_TPR_MIN}, FPR < {TARGET_FPR_MAX} for both classes)."
    )


# In[40]:


info_file_name = model_file_name.split('.')[0] + '.txt'
with open(info_file_name, 'w') as fpo:
    fpo.write(f"i16min            = {i16min           }\n")
    fpo.write(f"i16max            = {i16max           }\n")
    fpo.write(f"fsamp             = {fsamp            }\n")
    fpo.write(f"wave_length_ms    = {wave_length_ms   }\n")
    fpo.write(f"wave_length_samps = {wave_length_samps}\n")
    fpo.write(f"window_size_ms    = {window_size_ms   }\n")
    fpo.write(f"window_step_ms    = {window_step_ms   }\n")
    fpo.write(f"num_filters       = {num_filters      }\n")
    fpo.write(f"use_microfrontend = {use_microfrontend}\n")
    fpo.write(f"label_list        = {label_list}\n")
    fpo.write(f"spectrogram_shape = {spectrogram.numpy().shape}\n")
    fpo.write(f"Training accuracy (final) = {final_train_acc:.4f}\n")
    fpo.write(f"Validation accuracy (final) = {final_val_acc:.4f}\n")
    fpo.write(f"Test set accuracy = {test_acc:.4f}\n")
    fpo.write(f"training_curves_fpath = {training_curves_fpath}\n")
    fpo.write(f"confusion_matrix_fpath = {confusion_matrix_fpath}\n")
    fpo.write(
        "weight_sparsity: near-zero fraction per layer (|w| < 1% of that layer max |w|)\n"
    )
    for layer_name, frac in weight_sparsity_metrics.items():
        fpo.write(f"  sparsity_{layer_name} = {frac:.6f}\n")
    fpo.write(f"TPR_left = {tpr_left:.4f}\n")
    fpo.write(f"FPR_left = {fpr_left:.4f}\n")
    fpo.write(f"TPR_right = {tpr_right:.4f}\n")
    fpo.write(f"FPR_right = {fpr_right:.4f}\n")
    fpo.write(
        f"TARGET (TPR>{TARGET_TPR_MIN}, FPR<{TARGET_FPR_MAX} both classes) = {target_str}\n"
    )
    fpo.write(f"samples_train = {n_train_samples}\n")
    fpo.write(f"tpr_fpr_runs_fpath = {tpr_fpr_runs_fpath}\n")
    for i in range(n_classes):
        fpo.write(f"tpr_{label_list[i]:9} = {tpr[i]:.4f}\n")
        fpo.write(f"fpr_{label_list[i]:9} = {fpr[i]:.4f}\n")


# In[41]:


print(f"Wrote description to {info_file_name}")
with open(info_file_name) as fpi:
    print(fpi.read())


# In[42]:


print("\n=== Summary ===")
print(f"Training accuracy (final):   {final_train_acc:.4f}")
print(f"Validation accuracy (final): {final_val_acc:.4f}")
print(f"Test accuracy:                {float(test_acc):.4f}")
print(
    "Per-class TPR/FPR, TPR/FPR table, TARGET, and "
    f"{tpr_fpr_runs_fpath} printed above."
)
print_sparsity(
    weight_sparsity_metrics,
    "Weight near-zero fraction (same metrics as end of training)",
)
print(f"Figures: {training_curves_fpath}, {confusion_matrix_fpath}")

