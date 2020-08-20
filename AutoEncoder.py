'''
AutoEncoder For Denoising


'''
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose
import matplotlib.pyplot as plt
import pathlib
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



# define leaky ReLU function
def lrelu(x, alpha=0.1):
    return tf.math.maximum(alpha*x, x)

data_dir = "../input/retinal-dataset/Images/NOISY"

data_path = pathlib.Path(data_dir)

print('Database Directory')
print(data_path)

image_count = len(list(data_path.glob('*.png')))
print(image_count)

retinal_dataset = sorted(list(data_path.glob('*')))

img_height, img_width = 605, 700
batch_size = 8


#Preprocessing
training_data = []
def read_image(path):
    for img in path:
        im_path = str(img)
        img_array = cv2.imread(im_path)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_array = cv2.resize(img_array, dsize=(576,544), interpolation=cv2.INTER_AREA)
        training_data.append(img_array)

read_image(retinal_dataset) 

print(len(training_data))


X = np.array(training_data)
print(X.dtype)
X = X.astype('float32') / 255.0
print(X.dtype)

from sklearn.model_selection import train_test_split

train_imgs, test_imgs = train_test_split(X, test_size =0.2)


# plot image example from training images
plt.imshow(train_imgs[1])
plt.show()

print('Encoder Intialisation')
encoder = Sequential([
    # convolution
    Conv2D(
        filters=64,
        kernel_size=(3,3),
        strides=(1,1),
        padding='SAME',
        use_bias=True,
        activation=lrelu,
        name='conv1'
    ),
    # the input size is 28x28x32
    MaxPooling2D(
        pool_size=(2,2),
        strides=(2,2),
        name='pool1'
    ),
    # the input size is 14x14x32
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        strides=(1,1),
        padding='SAME',
        use_bias=True,
        activation=lrelu,
        name='conv2'
    ),
    # the input size is 14x14x32
    MaxPooling2D(
        pool_size=(2,2),
        strides=(2,2),
        name='encoding'
    ),
    # the output size is 7x7x32
        # the input size is 14x14x32
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        strides=(1,1),
        padding='SAME',
        use_bias=True,
        activation=lrelu,
        name='conv3'
    ),
    # the input size is 14x14x32
    MaxPooling2D(
        pool_size=(2,2),
        strides=(2,2),
        name='encoding1'
    )
    
])

print('Decoder Intialisation')
# describe decoder pipeline
decoder = tf.keras.Sequential([
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        strides=(1,1),
        name='conv3',
        padding='SAME',
        use_bias=True,
        activation=lrelu
    ),
    # updampling, the input size is 7x7x32
    Conv2DTranspose(
        filters=32,
        kernel_size=3,
        padding='same',
        strides=2,
        name='upsample1'
    ),
    # upsampling, the input size is 14x14x32
    Conv2DTranspose(
        filters=32,
        kernel_size=3,
        padding='same',
        strides=2,
        name='upsample2'
    ),
        # updampling, the input size is 7x7x32
    Conv2DTranspose(
        filters=64,
        kernel_size=3,
        padding='same',
        strides=2,
        name='upsample3'
    ),
    # the input size is 28x28x32
    Conv2D(
        filters=3,
        kernel_size=(3,3),
        strides=(1,1),
        name='logits',
        padding='SAME',
        use_bias=True
    )    
])

print('Initialising Model')

# model class definition
class EncoderDecoderModel(tf.keras.Model):
    def __init__(self, is_sigmoid=True):
        super(EncoderDecoderModel, self).__init__()
        # assign encoder sequence
        self._encoder = encoder
        # assign decoder sequence 
        self._decoder = decoder
        self._is_sigmoid = is_sigmoid
        
    # forward pass
    def call(self, x):
        x = self._encoder(x)
        decoded = self._decoder(x)
        if self._is_sigmoid:
            decoded = tf.keras.activations.sigmoid(decoded)
        return decoded

print('Initialising AutoEncoder')    
encoder_decoder_model = EncoderDecoderModel()
# training loop params
num_epochs = 100
batch_size_to_set = 8

# training process params
learning_rate = 1e-3
# default number of workers for training process
num_workers = 2
print('Compiling AutoEncoder')   
# initialize the training configurations such as optimizer, loss function and accuracy metrics
encoder_decoder_model.compile(
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
    loss=tf.keras.losses.MeanAbsoluteError(),
    metrics=None
)

print('Fit AutoEncoder')   
results = encoder_decoder_model.fit(
    train_imgs,
    train_imgs,
    epochs=num_epochs,
    batch_size=batch_size_to_set,
    validation_data=(test_imgs, test_imgs),
    workers=num_workers,
    shuffle=True
)
encoder_decoder_model.summary()
print('Complete!')  
# funstion for train and val losses visualizations
def plot_losses(results):
    plt.plot(results.history['loss'], 'bo', label='Training loss')
    plt.plot(results.history['val_loss'], 'r', label='Validation loss')
    plt.title('Training and validation loss',fontsize=14)
    plt.xlabel('Epochs ',fontsize=14)
    plt.ylabel('Loss',fontsize=14)
    plt.legend()
    plt.show()
    plt.close()
# visualize train and val losses
plot_losses(results)
## Call Saver to save the model and re-use it later during evaluation
encoder_decoder_model.save("retinal_model")




#restored_keras_model = tf.keras.models.load_model('../input/output/retinal_model')
#results = restored_keras_model.fit(train_imgs, epochs=20)
#plot_losses(results)

import numpy as np
predictions = np.clip(encoder_decoder_model.predict(test_imgs), 0.0, 1.0)
plt.imshow(test_imgs[0])
plt.show()
plt.imshow(predictions[0])
plt.show()
test = cv2.cvtColor(test_imgs[0], cv2.COLOR_RGB2BGR)
pred = cv2.cvtColor(predictions[0], cv2.COLOR_RGB2BGR)
cv2.imwrite("test.png", 255*test)
cv2.imwrite("pred.png", 255*pred)
        
