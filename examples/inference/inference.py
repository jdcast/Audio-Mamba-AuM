#!/usr/bin/env python
# coding: utf-8

# # Example Inference Notebook

# This notebook contains an example of how to use a trained model to make predictions on a test set. 
# There are three main relevant folders:
# - ``data/``: This folder contains samples the model will be tested on. Currently, it contains 5 samples from the evaluation set of the VGGSound dataset.
# - ``datafiles/``: This folder contains information about the labels and the filepaths of the samples.
# - ``models/``: This folder contains a trained model.
# 
# The notebook firstly downloads the [AudioSet pretrained VGGSound model](https://drive.google.com/file/d/1spsJXncpEXHKmIvDcB7ddkcgrzARpEeK/view?usp=drive_link) under the ``models/`` folder. Then, it loads the model and does a single sample or batch inference on the samples under the ``data/`` folder.

# ## Imports and Setup

# In[1]:


# Convenient for importing modules from the parent directory
import sys
sys.path.append("../../")


# In[2]:


import os
import torch
import torchaudio
import numpy as np
import src.models as models
from src import dataloader
from src.utilities.stats import calculate_stats
from IPython.display import Audio, display
import csv
import warnings

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# In[3]:


# Install the gdown package to download a model from the gdrive link
# get_ipython().system('pip install gdown')

import gdown

# model link and save path information
gdrive_link = 'https://drive.google.com/file/d/1spsJXncpEXHKmIvDcB7ddkcgrzARpEeK/view?usp=drive_link'
model_base_path = 'models/'
model_name = 'aum-base_audioset-vggsound.pth'
model_path = model_base_path + model_name

# check if the directory exists
if not os.path.exists(model_base_path):
    os.makedirs(model_base_path)

# make a downloadable link and downlaod the model
model_link = f'https://drive.google.com/uc?id={gdrive_link.split("/")[-2]}'
# gdown.download(model_link, model_path, quiet=False)

print(f'Model downloaded and saved to {model_path}')


# ## Model and Data

# In[4]:


# Arguments about the data
data_args = Namespace(
    num_mel_bins = 128,
    target_length = 1024,
    mean = -5.0767093,
    std = 4.4533687,
)

# Arguments about the model
model_args = Namespace(
    model_type = 'base',
    n_classes = 309,
    imagenet_pretrain = False,
    imagenet_pretrain_path = None,
    aum_pretrain = True,
    aum_pretrain_path = 'models/aum-base_audioset-vggsound.pth',
    aum_variant = 'Fo-Bi',
    device = 'cuda',
)


# In[5]:


# Initilize the model

# Embedding dimension
if 'base' in model_args.model_type:
    embed_dim = 768
elif 'small' in model_args.model_type:
    embed_dim = 384
elif 'tiny' in model_args.model_type:
    embed_dim = 192

# AuM block type
bimamba_type = {
    'Fo-Fo': 'none', 
    'Fo-Bi': 'v1', 
    'Bi-Bi': 'v2'
}.get(
    model_args.aum_variant, 
    None
)

# Create the model
AuM = models.AudioMamba(
    spectrogram_size=(data_args.num_mel_bins, data_args.target_length),
    patch_size=(16, 16),
    strides=(16, 16),
    embed_dim=embed_dim,
    num_classes=model_args.n_classes,
    imagenet_pretrain=model_args.imagenet_pretrain,
    imagenet_pretrain_path=model_args.imagenet_pretrain_path,
    aum_pretrain=model_args.aum_pretrain,
    aum_pretrain_path=model_args.aum_pretrain_path,
    bimamba_type=bimamba_type,
)

# Prepare the model for inference
AuM.to(model_args.device)
AuM.eval()


# ## Single Sample Inference

# In[6]:


# Arguments about the dataset
single_sample_dataset_args = Namespace(
    # sample_path = 'data/sample0.wav',
    sample_path = '/home/jdcast/wav_training_data/TK/wav/1678008359.170411002002.wav',
    sample_label = '/m/vggsd273',
    label_csv = 'datafiles/class_labels_indices.csv',
)

# For indexing and storing labels
index_dict, label_dict = {}, {}
with open(single_sample_dataset_args.label_csv, 'r') as f:
    csv_reader = csv.DictReader(f)
    line_count = 0
    for row in csv_reader:
        index_dict[row['mid']] = row['index']
        label_dict[row['index']] = row['display_name']
        line_count += 1


# In[8]:


# Play the audio
audio_path = single_sample_dataset_args.sample_path
display(Audio(filename=audio_path))


# In[9]:


# Obtain the waveform and normalize
waveform, sr = torchaudio.load(audio_path)
waveform = waveform - waveform.mean()

# Extract the features
fbank = torchaudio.compliance.kaldi.fbank(
    waveform, 
    htk_compat=True, 
    sample_frequency=sr, 
    use_energy=False,
    window_type='hanning', 
    num_mel_bins=data_args.num_mel_bins,
    dither=0.0, 
    frame_shift=10
)

# Compute the padding length
n_frames = fbank.shape[0]
p = data_args.target_length - n_frames

# cut or pad
if p > 0:
    m = torch.nn.ZeroPad2d((0, 0, 0, p))
    fbank = m(fbank)
elif p < 0:
    fbank = fbank[0:data_args.target_length, :]


# initialize the label
label_indices = np.zeros(model_args.n_classes)
label_indices[int(index_dict[single_sample_dataset_args.sample_label])] = 1.0
label_indices = torch.FloatTensor(label_indices)

# Normalize the features
fbank = (fbank - data_args.mean) / (data_args.std * 2)

# Add batch dimension
fbank = fbank.unsqueeze(0)
label_indices = label_indices.unsqueeze(0)


# In[10]:


# Move to device
fbank = fbank.to(model_args.device)
label_indices = label_indices.to(model_args.device)

# Forward pass
with torch.no_grad():
    output = AuM(fbank)

# The prediction
output = torch.sigmoid(output)
output = output.cpu().numpy()


# In[11]:


# The top 5 predictions
top5 = np.argsort(output[0])[-5:][::-1]

print('The top 5 predictions are:')
for i in top5:
    print(label_dict[str(i)], output[0][i])

# The actual label
actual_label = label_dict[ 
    index_dict[
        single_sample_dataset_args.sample_label
    ]
] 

print(f'\nThe actual label: {actual_label}')


# ## Batch Inference

# In[11]:


# Arguments about the dataset
batch_dataset_args = Namespace(
    data_val = 'datafiles/eval.json',
    label_csv = 'datafiles/class_labels_indices.csv',
    main_metric = 'Acc',
    batch_size = 2,
    dataset = 'vggsound-eval-subset'
)


# In[12]:


# Dataloader configuration
val_audio_conf = {
    'num_mel_bins': data_args.num_mel_bins, 
    'target_length': data_args.target_length, 
    'freqm': 0, 
    'timem': 0, 
    'mixup': 0, 
    'dataset': batch_dataset_args.dataset, 
    'mode': 'evaluation', 
    'mean': data_args.mean, 
    'std': data_args.std, 
    'noise': False
}

# Create the dataloader
val_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(
        batch_dataset_args.data_val, 
        label_csv=batch_dataset_args.label_csv, 
        audio_conf=val_audio_conf
    ),
    batch_size=batch_dataset_args.batch_size, 
    shuffle=False, 
    pin_memory=True
)


# In[13]:


# Model predictions and targets
predictions = []
targets = []

# The batch inference
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        audio_input, labels, path = batch
        audio_input = audio_input.to(model_args.device)
        
        audio_output = AuM(audio_input)
        audio_output = torch.sigmoid(audio_output)
        
        prediction = audio_output.detach()
        predictions.append(prediction)
        
        targets.append(labels)


# In[14]:


# Concatenate the predictions and targets
predictions = torch.cat(predictions, dim=0)
predictions = predictions.cpu().numpy()

targets = torch.cat(targets, dim=0)
targets = targets.cpu().numpy()

# Suppress the warnings of sklearn regarding recall / precision
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    stats = calculate_stats(predictions, targets, skips={'auc'}) # Skipping AUC since not our focus

# Measure performance according to the main metric
if batch_dataset_args.main_metric == 'mAP':
    perf = np.mean([stat['AP'] for stat in stats])
elif batch_dataset_args.main_metric == 'Acc':
    perf = stats[0]['acc']

print(f'{batch_dataset_args.main_metric}: {perf:.4f}')

