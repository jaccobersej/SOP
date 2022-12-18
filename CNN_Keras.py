import tensorflow as tf
import keras 
import numpy
from PIL import Image
import matplotlib.pyplot as plt
import pydot
import graphviz

train_ds = keras.utils.image_dataset_from_directory(
    directory='Data_collection/Data/Fewer_place_Train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 1280))
validation_ds = keras.utils.image_dataset_from_directory(
    directory='Data_collection/Data/Fewer_place_Test',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 1280))

#model = tf.keras.models.load_model('old_models/færdig_model_1 67 acc 30 val acc')

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, kernel_size=3, activation="relu", padding="same"),
                                    #tf.keras.layers.SpatialDropout2D(0.2),
                                    tf.keras.layers.Conv2D(16, kernel_size=3, activation="relu", padding="same"),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
                                    #tf.keras.layers.SpatialDropout2D(0.2),
                                    tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
                                    #tf.keras.layers.SpatialDropout2D(0.2),
                                    tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(128, kernel_size=3, activation="relu", padding="same"),
                                    #tf.keras.layers.SpatialDropout2D(0.2),
                                    tf.keras.layers.Conv2D(128, kernel_size=3, activation="relu", padding="same"),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(256, kernel_size=3, activation="relu", padding="same"),
                                    #tf.keras.layers.SpatialDropout2D(0.2),
                                    tf.keras.layers.Conv2D(256, kernel_size=3, activation="relu", padding="same"),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(512, kernel_size=3, activation="relu", padding="same"),
                                    tf.keras.layers.SpatialDropout2D(0.2),
                                    tf.keras.layers.Conv2D(512, kernel_size=3, activation="relu", padding="same"),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(768, activation="relu"),
                                    tf.keras.layers.Dropout(0.4),
                                    tf.keras.layers.Dense(768, activation="relu"),
                                    tf.keras.layers.Dropout(0.4),
                                    tf.keras.layers.Dense(13, activation="softmax")])

def scheduler(epoch, lr):
    decay_rate = tf.math.exp(-0.3)
    decay_step = 2
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

opt = tf.keras.optimizers.Adam(learning_rate=0.00005)
model.build(input_shape=(None, 256, 1280, 3))
model.summary() 
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy(name='Cat_Accuracy')])
model_history = model.fit(train_ds, epochs=60, validation_data=validation_ds, callbacks=callback)

def Train_Val_Plot(acc, val_acc, loss, val_loss):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(" MODEL'S METRICS VISUALIZATION ")

    ax1.plot(range(1, len(acc) + 1), acc)
    ax1.plot(range(1, len(val_acc) + 1), val_acc)
    ax1.set_title('History of Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(['training', 'validation'])

    ax2.plot(range(1, len(loss) + 1), loss)
    ax2.plot(range(1, len(val_loss) + 1), val_loss)
    ax2.set_title('History of Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(['training', 'validation'])

    plt.show()

model.save('models/')

Train_Val_Plot(model_history.history['Cat_Accuracy'], model_history.history['val_Cat_Accuracy'], model_history.history['loss'], model_history.history['val_loss'])

#tf.keras.utils.plot_model(model, to_file='færdig_model_1.jpg', show_shapes=True)