#Import libraries der skal bruges:
import tensorflow as tf
import keras 
import numpy
from PIL import Image
import matplotlib.pyplot as plt
import pydot
import graphviz


#Load datasæt. Først bliver det datasæt der skal trænes på loaded. Måden den skal skabe labels på og størrelsen af billederne er defineret.
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

#Mulighed for at loade en gammel model
#model = tf.keras.models.load_model('MODEL_PATH')

#Definer modellen (Hvilke lag der er tilstede og nogle af parametrene for den)
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
                                    tf.keras.layers.Dense(512, activation="relu"),
                                    tf.keras.layers.Dropout(0.4),
                                    tf.keras.layers.Dense(512, activation="relu"),
                                    tf.keras.layers.Dropout(0.4),
                                    tf.keras.layers.Dense(13, activation="softmax")])

#Hvis man gerne have at learning raten skal aftage efter en bestemt mængde træningsomgange (Epochs) kan man justere det her. Det hjælper mod overfitting
#Det hjælper også med at sikre at man finder et lokalt optimum for modellen.
"""def scheduler(epoch, lr):
    decay_rate = tf.math.exp(-0.3)
    decay_step = 2
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)"""

#Definer, hvilken optimizer der bliver brugt og sæt den globale start learning rate
opt = tf.keras.optimizers.Adam(learning_rate=0.00005)

#build modellen. Det gøres for at den er klar og muliggører at vi kan lave et summary af den
model.build(input_shape=(None, 256, 1280, 3))

#Printer et resume af modellen som viser, hvilke shapes dataen kommer i undervejs og hvor mange parametre modellen har
model.summary() 

#Fortæl modellen, hvilken optimizer og loss den skal bruge. Derudover fortæller man den også, hvilken metrics den skal holde styr på.
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy(name='Cat_Accuracy')])

#Træn modellen på train_ds (Trænings datasættet) 60 gange (Epochs=60) og valider efter, hver omgang med validation_ds (validations/test datasættet)
#Det returnerer så en history, som vi kan bruge til at plotte en graf af de forskellige metrics, som modellen har holdt styr på.
model_history = model.fit(train_ds, epochs=60, validation_data=validation_ds)#, callbacks=callback)

#Funktion der plotter de forskellige metrics, som modellen har haft styr på.
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

#Gem modellen sådan at man kan bruge den samme færdige model på et andet tidspunkt uden at skulle træne den igen.
model.save('models/')

#Kald funktionen der plotter metrics for at plotte metrics
Train_Val_Plot(model_history.history['Cat_Accuracy'], model_history.history['val_Cat_Accuracy'], model_history.history['loss'], model_history.history['val_loss'])
