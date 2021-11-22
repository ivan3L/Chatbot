import tkinter

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import json
import random
import pickle
from tkinter import *

#Solo descargar una vez:
#nltk.download('punkt')

#Leemos respuestas del .json
with open("answers.json", encoding='utf-8') as file:
    datos = json.load(file)

palabras = []
tags = []
auxX = []
auxY = []

#Usamos librería nltk para utilizar lenguaje natural y separar palabras
for content in datos["content"]:
    for patterns in content["patterns"]:
        #Agregamos a lista de palabras clave
        auxWord = nltk.word_tokenize(patterns)
        #Agregamos a lista de frases
        palabras.extend(auxWord)
        #Agregamos a lista de palabras
        auxX.append(auxWord)
        #Agregamos a lista de tags
        auxY.append(content["tag"])

        #Añadimos el tag si es que no esta
        if content["tag"] not in tags:
            tags.append(content["tag"])

palabras = [stemmer.stem(w.lower()) for w in palabras if w!="?"]
palabras = sorted(list(set(palabras)))
tags = sorted(tags)

entrenamiento = []
salida = []

salidaVacia = [0 for _ in range(len(tags))]

for x, documento in enumerate(auxX):
    cubeta = []
    auxPalabra = [stemmer.stem(w.lower()) for w in documento]
    for w in palabras:
        if w in auxPalabra:
            cubeta.append(1)
        else:
            cubeta.append(0)
    filaSalida = salidaVacia[:]
    filaSalida[tags.index(auxY[x])] = 1
    entrenamiento.append(cubeta)
    salida.append(filaSalida)

entrenamiento = numpy.array(entrenamiento)
salida = numpy.array(salida)

with open("variables.pickle","wb") as pickleFile:
    pickle.dump((palabras,tags,entrenamiento,salida),pickleFile)

tensorflow.compat.v1.reset_default_graph()

#Entrenamos red neuronal
red = tflearn.input_data(shape = [None,len(entrenamiento[0])])
#2 hidden layers
red = tflearn.fully_connected(red,10)
red = tflearn.fully_connected(red,10)
red = tflearn.fully_connected(red,len(salida[0]),activation="softmax")
red = tflearn.regression(red)

modelo = tflearn.DNN(red)

modelo.fit(entrenamiento,salida,n_epoch=1000,batch_size=10,show_metric=True)
modelo.save("modelo.tflearn")

def botAnswer(entrada):
    cubeta = [0 for _ in range(len(palabras))]
    entradaProcesada = nltk.word_tokenize(entrada)
    entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]
    for palabraIndividual in entradaProcesada:
        for i,palabra in enumerate(palabras):
            if palabra == palabraIndividual:
                cubeta[i] = 1
    resultados = modelo.predict([numpy.array(cubeta)])
    #print(resultados)
    resultadosIndices = numpy.argmax(resultados)
    tag = tags[resultadosIndices]

    for tagAux in datos["content"]:
        if tagAux["tag"] == tag:
            respuesta = tagAux["answers"]

    #print("Bot: ", random.choice(respuesta))
    return random.choice(respuesta)

def send():
    input = e.get()
    rpta = botAnswer(input)
    txt.insert(END,"\n"+"Tú: "+input)
    e.delete(0,END)
    txt.insert(END,"\n"+"Washi Bot: "+rpta)

root = Tk()
txt=Text(root)
txt.grid(row=0,column=0,columnspan=2)
e = Entry(root,width=100)
send = Button(root, text = "Send", command = send).grid(row=1,column=1)
e.grid(row=1,column=0)
root.title("Washi Bot")
root.mainloop()