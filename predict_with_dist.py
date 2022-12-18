"""
Det her script gætter på positionen af alle de billeder, som det neurale netværk ikke er blevet trænet på og viser et plot,
af hvor mange lå inde for nogle bestemte afstands kategorier: ['10.000+', '10.000-2500', '2500-750', '750-200', '200-0']
"""

#Import nødvendige libraries
import tensorflow as tf
import numpy as np
from PIL import Image
import pygeohash as pgh
import glob
import matplotlib.pyplot as plt

#Saml file paths til alle test billeder og gem indexet af hvilke labels de har i et andet array så vi kan checke det senere for at se om vi gættede rigtigt.
dirpath = "C:/Users/jacob/Documents/3DDU2 intelligente systemer/Geoguessr/Data_collection/Data/fewer_place_test"
test_data = glob.glob(dirpath + "/*/*.jpg")
labels = glob.glob(dirpath + "/*")
labels = [label[len(dirpath):][1:3] for label in labels]
test_data_labels = [x[len(dirpath):][1:3] for x in test_data]
for index, label in enumerate(test_data_labels):
    test_data_labels[index] = labels.index(label)
   
#Tom liste, hvor vi kan samle alle de afstande der er mellem det rigtige gæt og det gæt modellen laver.
GuessDists = []

#Load den færdige model som skal bruges til at gætte med via filepath til den mappe, hvor modellens "Saved_model.pb" ligger. 
model = tf.keras.models.load_model('old_models/Færdig_model_3 93 acc 30 val acc')

#For hvert billede i listen af test billeder: Gæt, hvor de og udregn haversine afstanden mellem gættet og den rigtige placering i km.
for i in range(len(test_data)):
    singleImage = Image.open(test_data[i])
    predThis =  np.array([np.resize(np.array(singleImage), (256, 1280, 3))])
    true_label = labels[test_data_labels[i]]
    prediction = model.predict(predThis)
    predLabel = labels[np.argmax(prediction)][-2:]
    dist = pgh.geohash_haversine_distance(predLabel, true_label)
    GuessDists.append(dist/1000) #Delt med 1000 da pgh.geohash_haversine_distance() returnerer sine svar i meter

#Definer de forskellige afstands kategorier
categories = ['10.000+', '10.000-2500', '2500-750', '750-200', '200-0']
counts = [0, 0, 0, 0, 0]

#Tæl hvor mange af gættene tilhører de forskellige kategorier
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

#Print, hvor mange procent tilhørte de forskellige kategorier
print('10.000+: ', counts[0]/np.array(counts).sum(), '10.000-2500: ', counts[1]/np.array(counts).sum(), '2500-750: ', counts[2]/np.array(counts).sum(), '750-200: ', counts[3]/np.array(counts).sum(), '200-0', counts[4]/np.array(counts).sum())
        
#Plot afstandene som søjlediagram
plot, ax = plt.subplots()
ax.bar(categories, counts)
ax.set_ylabel('Occurences')
ax.set_title('Distances in km by occurences')

plt.show()
