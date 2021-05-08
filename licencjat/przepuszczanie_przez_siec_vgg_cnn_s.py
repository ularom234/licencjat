import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import os


#ustawianie CPU czu CPU i GPU
caffe.set_mode_cpu()

#wczytwyanie sieci
sciezka_modelu="/home/ula/Pulpit/strKomp/ula/Dokumenty/licencjatNI/machineLearning/NowyFolder/models/vgg_cnn_s"

net = caffe.Net(sciezka_modelu+"/"+"VGG_CNN_S_deploy.prototxt",
                sciezka_modelu+"/"+"VGG_CNN_S.caffemodel",
                caffe.TEST)

print "Wczytano siec"
print "Wymiar wejscia: ",net.blobs["data"].data.shape

#Wczytwanie adresu datasetu

sciezka_obrazow="/home/ula/Pulpit/CSVM/rot90kol"
#sciezka_obrazow="/home/ula/Dokumenty/licencjatNI/machineLearning/folder"
lista_obrazow=os.listdir(sciezka_obrazow)
lista_obrazow.sort()
print "Liczba obrazow w datasecie: ",len(lista_obrazow)


#Konfigurowanie preproccesingu
transformer=caffe.io.Transformer({"data": net.blobs["data"].data.shape})
transformer.set_mean("data",np.load(sciezka_modelu+"/"+"VGG_mean.npy").mean(1).mean(1))
#transformer.set_mean("data",np.load("/home/ula/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy").mean(1).mean(1))
transformer.set_transpose("data",(2,0,1))
transformer.set_channel_swap("data",(2,1,0))
transformer.set_raw_scale("data",255.0)

print "Zakonczono konfigurowanie preproccesingu"
# wymiar 1000 lub 4096 w zaleznisci od wartswy
#tworzenie macierzy daych
X=np.zeros(shape=(len(lista_obrazow),4096))


#Petla przepuszczajaca obrazow
counter=0

for obraz in lista_obrazow:
    net.blobs["data"].reshape(1,3,224,224)
    im=caffe.io.load_image(sciezka_obrazow+"/"+obraz)
    net.blobs["data"].data[...]=transformer.preprocess("data",im)
    net.forward()
    #print net.blobs["data"].data.mean()
    warstwa=net.blobs["fc6"].data
    X[counter]=warstwa
    print "Srednia warstwy: ",np.mean(warstwa)
    print "Maksimum warstwy: ",np.max(warstwa)
    counter+=1
    print counter," obrazow zrobionych"

print X[:5,:5]

print np.mean(X[765])

#zapisywanie macierzy danych
np.save("vgg_cnn_s_fc6",X)
        


    
