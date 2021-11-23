import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.callbacks import ModelCheckpoint

from model import model
from data import X, y


print("Enter directory to save model weights:")
dir_name = input()

filepath = dir_name + "/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(
        filepath, monitor='loss', 
        verbose=1, save_best_only=True, 
        mode='min'
        )
callbacks_list = [checkpoint]

# fit the data
model.fit(X, y, epochs=1, batch_size=128, callbacks=callbacks_list)
