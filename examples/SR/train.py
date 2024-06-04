#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime
import ipykernel
from csbdeep.io import load_training_data
from csbdeep.utils import axes_dict, plot_some,plot_history
import matplotlib.pyplot as plt
from actin_tubules_sim.models import DFCAN
from actin_tubules_sim.loss import mse_ssim
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks


# In[2]:


root_dir ='/home/asarker/SVI/Images/Thesis_argha/actin_tubles_sim/Microtubules/'
train_data_file = f'{root_dir}/Train/SR/microtubule_sr_training_data.npz'
log_dir = "logs/fitSR/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# In[3]:


(X,Y), (X_val,Y_val), axes = load_training_data(train_data_file, validation_split=0.1, verbose=True)

c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]
X = tf.squeeze(X, axis=-1)
X_val = tf.squeeze(X_val, axis=-1)
Y = tf.squeeze(Y, axis=-1)
Y_val = tf.squeeze(Y_val, axis=-1)
X = tf.transpose(X, perm=[0, 2, 3, 1])
X_val = tf.transpose(X_val, perm=[0, 2, 3, 1])

Y = tf.transpose(Y, perm=[0, 2, 3, 1])

Y_val = tf.transpose(Y_val, perm=[0, 2, 3, 1])




# In[4]:


X.shape,Y.shape,X_val.shape


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(12,5))
plot_some(tf.transpose(X_val[:5], perm=[0, 3, 1, 2]),Y_val[:5])
plt.suptitle('5 example validation patches (top row: source, bottom row: target)')


# In[6]:


init_lr = 1e-4
batch_size = 3
epochs = 10
beta_1=0.9
beta_2=0.999
scale_gt = 2.0

total_data,  height, width, channels= X.shape


# In[7]:


Trainingmodel = DFCAN((height, width, channels), scale=scale_gt)
optimizer = Adam(learning_rate=init_lr, beta_1=beta_1, beta_2=beta_2)
Trainingmodel.compile(loss=mse_ssim, optimizer=optimizer)
Trainingmodel.summary()

tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
lrate = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
hrate = callbacks.History()


# ## Training
# Training the model will likely take some time. We recommend to monitor the progress with [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard), which allows you to inspect the losses during training.
# 
# You can start TensorBoard from the current working directory with `tensorboard --logdir=.` Then connect to http://localhost:6006/ with your browser.

# In[8]:


history = Trainingmodel.fit(X, Y, batch_size=batch_size,
                               epochs=epochs, validation_data=(X_val, Y_val), shuffle=True,
                               callbacks=[lrate, hrate, tensorboard_callback])


# In[9]:


print(sorted(list(history.history.keys())))
plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss'])


# In[10]:


plt.figure(figsize=(12,7))
_P = Trainingmodel.predict(X_val[:5])

plot_some(tf.transpose(X_val[:5], perm=[0, 3, 1, 2]),Y_val[:5],_P,pmax=99.5)
plt.suptitle('5 example validation patches\n'      
             'top row: input (source),  '          
             'middle row: target (ground truth),  '
             'bottom row: predicted from source')


# In[11]:


Trainingmodel.save(root_dir)


# In[ ]:




