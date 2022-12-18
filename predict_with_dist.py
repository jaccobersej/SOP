import tensorflow as tf
import numpy as np
from PIL import Image
import pygeohash as pgh
import glob
import matplotlib.pyplot as plt

dirpath = "Data_collection/Data/fewer_place_test/" #"C:/Users/jacob/Documents/3DDU2 intelligente systemer/Geoguessr/Data_collection/Data/testing_data_for_test"
labels = glob.glob(dirpath + "/*")
labels = [label[len(dirpath):][1:3] for label in labels]
dirpath = "C:/Users/jacob/Documents/3DDU2 intelligente systemer/Geoguessr/Data_collection/Data/fewer_place_test" #"C:/Users/jacob/Documents/3DDU2 intelligente systemer/Geoguessr/Data_collection/Data/testing_data_for_test"
test_data = glob.glob(dirpath + "/*/*.jpg")
labels = glob.glob(dirpath + "/*")
labels = [label[len(dirpath):][1:3] for label in labels]
test_data_labels = [x[len(dirpath):][1:3] for x in test_data]
print(test_data_labels)
for index, label in enumerate(test_data_labels):
    test_data_labels[index] = labels.index(label)
    
GuessDists = []
model = tf.keras.models.load_model('old_models/Færdig_model_3 93 acc 30 val acc')

for i in range(len(test_data)): #int(len(Train_data)/32) #len(Train_data)
    singleImage = Image.open(test_data[i])
    predThis =  np.array([np.resize(np.array(singleImage), (256, 1280, 3))])
    true_label = labels[test_data_labels[i]]
    prediction = model.predict(predThis)
    predLabel = labels[np.argmax(prediction)][-2:]
    dist = pgh.geohash_haversine_distance(predLabel, true_label)
    GuessDists.append(dist/1000)


categories = ['10.000+', '10.000-2500', '2500-750', '750-200', '200-0']
counts = [0, 0, 0, 0, 0]
for value in GuessDists:
    if value > 10000:
        counts[0] += 1
    elif value > 2500:
        counts[1] += 1
    elif value > 750:
        counts[2] += 1
    elif value > 200:
        counts[3] += 1
    elif value >= 0:
        counts[4] += 1

print('10.000+: ', counts[0]/np.array(counts).sum(), '10.000-2500: ', counts[1]/np.array(counts).sum(), '2500-750: ', counts[2]/np.array(counts).sum(), '750-200: ', counts[3]/np.array(counts).sum(), '200-0', counts[4]/np.array(counts).sum())
        
plot, ax = plt.subplots()

ax.bar(categories, counts)
ax.set_ylabel('Occurences')
ax.set_title('Distances in km by occurences')

plt.show()
    
"""img = Image.open('Data_collection/Data/Fewer_place_Test/u0/img_45.14114761352539,10.034805297851562.jpg')
predThis = np.array([np.resize(np.array(img), (256, 1280, 3))])

true_label = 'u0'

print(prediction)
print(np.argmax(prediction[0]))
print(prediction[0][np.argmax(prediction[0])])
#print(np.argsort(prediction)[0])
#print(np.argsort(prediction)[0][0])
predLabel = labels[np.argmax(prediction)][-2:]
print(predLabel)

dist = pgh.geohash_approximate_distance(predLabel, true_label)
haverDist = pgh.geohash_haversine_distance(predLabel, true_label)
print('approx dist=', dist)
print('approx dist/1000', dist/1000)
print('haverDist=', haverDist)
print('haverDist/1000 =', haverDist/1000)"""