#!/usr/bin/python
# -*- coding: ascii -*-
import os, sys
import numpy as np
#import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.svm import SVC
from sklearn.cross_validation import KFold, cross_val_score
from scipy import stats
import xlwt

#skalowanie danych wejsciowych:

def sigmoid_scaler2(A):
    B = np.divide(np.ones(A.shape),(1.0 + np.exp((-1.0) * A)))
    return B

def testKolmogorowaSmimrnowa(X,test_index,y,gdzie1,gdzie0):
    true_test=np.append(gdzie1[test_index], gdzie0[test_index])
    y_test=y[true_test].astype(int)
    notest=np.setdiff1d(np.arange(y.size),true_test) #indeksy obrazow nie wzietych do testu
    nn=np.zeros((X.shape[1], 2))
    for jq in np.arange(X.shape[1]):
        sk, p = stats.ks_2samp(X[np.intersect1d(gdzie1,notest), jq], X[np.intersect1d(gdzie0, notest), jq])
        nn[jq, 0]=p
        nn[jq, 1]=jq
    nn=np.array(sorted(nn, key=lambda m: m[0], reverse=False))
    ind=nn[:,1].astype(int)
    #f.write( str(nn[0,0]*100)+"%"+","+ str(nn[ile, 0]*100)+"%"+ "\n")
    return ind

def testSrednich(X,test_index,y,gdzie1,gdzie0):
    true_test=np.append(gdzie1[test_index], gdzie0[test_index])
    y_test=y[true_test].astype(int)
    notest=np.setdiff1d(np.arange(y.size),true_test) #indeksy obrazow nie wzietych do testu
    nn=[[iq,jq] for iq,jq in zip(np.abs(np.mean(X[np.intersect1d(gdzie1,notest)], axis=0)
            -np.mean(X[np.intersect1d(gdzie0, notest)], axis=0)), np.arange(X.shape[1]))]
    nn=np.array(sorted(nn, key=lambda m: m[0], reverse=True))
    ind=nn[:,1].astype(int)
    return ind


def petlaCrossWalidacji(D,y,folds,ile,test,ratio_test=1,kolejna=0): 
    """D to lista krotek numery i ratio (ilosc 0 w stosunku do 1. zawsze bierze wszystkie 1 i jesli R=1 tyle samo zer, jesli r>1 wiecej zer niz jedynek, r<1 mniej zer od jedynek)
    y tablica etykiet
    folds int ile folderow
    ile int ile parametrow
    test funkcja np.srednie i kolmogorow
    """
    tytul=["vgg_cnn_s_fc6.npy","vgg_cnn_s_fc6_kolor.npy","vgg_cnn_s_fc6_rot90kol.npy","vgg_cnn_s_fc6_gray.npy","vgg_cnn_s_fc6_jasnosc.npy","vgg_cnn_s_fc6_ostrosc.npy","vgg_cnn_s_fc6_kontrast.npy","vgg_cnn_s_fc6_left_right.npy",
    "vgg_cnn_s_fc6_kolory_random.npy","vgg_cnn_s_fc6_rot180kol.npy","vgg_cnn_s_fc6_rot270kol.npy","vgg_cnn_s_fc6_rotacja90.npy","vgg_cnn_s_fc6_rotacja180.npy","vgg_cnn_s_fc6_rotacja270.npy","vgg_cnn_s_fc6_top_botton.npy"]
 
    global s0
    gdzie1=np.where(y==1)[0]
    gdzie0=np.where(y==0)[0]
    np.random.seed(s0) 
    np.random.shuffle(gdzie1)
    np.random.shuffle(gdzie0)
    kf = KFold(gdzie1.size,n_folds=folds)

    X = []
    R = []
    uzyte = []
    for i in D:
        X.append(np.load(tytul[i[0]]))
        R.append(i[1])
        uzyte.append(tytul[i[0]])

   
    #glowna petla cross-walidacyjna,
    nazwy=["Wymiar wejscia", "Wymiar etykiet", "tyle jedynek", "tyle zer", "folds", "ile", "test",'seed']
    wartosci=[str(np.shape(X[0])), str(np.shape(y)), float(gdzie1.size), float(gdzie0.size), float(folds), float(ile), test.__name__ ,int(s0)]
    for i in range(len(nazwy)):
        sheet1.write(i, 0 +4*kolejna, nazwy[i])
        sheet1.write(i, 1 +4*kolejna, wartosci[i])
    n = len(nazwy)
    for i in range(len(uzyte)):
        n += 1
        sheet1.write(n, 0 +4*kolejna, "do testu")
        sheet1.write(n, 1 +4*kolejna, uzyte[i])
        sheet1.write(n, 2 +4*kolejna, R[i])

    bestiF1 = 0
    bestF1 = 0
    bestF1std = 0
    bestacc = 0
    bestaccstd = 0
    bestrec = 0
    bestrecstd = 0
    bestpres = 0
    bestpresstd = 0
    n = 16

    for k in xrange(0,9):
        #inicjalizacja trafnosci, precyzji, recall i f1 score
        acc = np.zeros(folds)
        pres = np.zeros(folds)
        rec = np.zeros(folds)
        f1 = np.zeros(folds)
        counter = 0
    
        for train_index,test_index in kf:

            #! wybor parametrow najlepiej roznicujacych dwie klasy
            ind=test(X[0],test_index,y,gdzie1,gdzie0)
            #f.write(test.__name__ + "\n")
            
            for i in range(len(X)):
                X_scaled=(sigmoid_scaler2(X[i][:,ind[:ile]]))            
                true_train=np.append(gdzie1[train_index], gdzie0[train_index[:int(min(1,R[i])*len(train_index))]])
                if R[i]>1:
                    true_train=np.append(true_train, gdzie0[gdzie1.size:gdzie1.size+int((R[i]-1)*train_index.size)])
                if i==0:
                    X_train=X_scaled[true_train]
                    y_train=y[true_train]
                    X_scaled0=X_scaled
                else:
                    X_train=np.append(X_train, X_scaled[true_train], axis=0)
                    y_train=np.append(y_train,y[true_train])                    

            #ind2=np.arange(y_train.size)
            #np.random.shuffle(ind2)
            #X_train=X_train[ind2]
            #y_train=y_train[ind2]

            true_test=np.append(gdzie1[test_index[:int(ratio_test*len(train_index))]], gdzie0[test_index])
            #? Na ktorych danych testujemy: oryginalnych czy dosypanych?
            X_test=X_scaled0[true_test]
            y_test=y[true_test].astype(int)
    
            #! Wartosc C warto zwiekszac logarytmicznie
            C1=10**(float(k)/3)
            svm=SVC(kernel='rbf', C=C1)
            svm.fit(X_train,list(y_train))
            params=svm.get_params()
    
            pred=svm.predict(X_test)
            acc[counter] = accuracy_score(y_test,pred)
            pres[counter] = precision_score(y_test,pred)
            rec[counter] = recall_score(y_test,pred)
            f1[counter] = f1_score(y_test,pred)
            counter += 1
   
        #nazwy2=["numer","C","treningowych","testowych","treningowych jedynek"]
        #nazwy3=["Finalowa trafnosc SVMa","Finalowe F1 score SVMa", "Finalowe recall SVMa","Finalowe precision SVMa"]
        #wartosci2=[float(k),float(C1),str(X_train.shape),str(X_test.shape),float(np.sum(y_train))]
        #wartosci3=[float(np.mean(acc)),float(np.mean(f1)),float(np.mean(rec)),float(np.mean(pres))]
        ##odchyleniewartosci3=[float(np.std(acc)),float(np.std(f1)),float(np.std(rec)),float(np.std(pres))]
        #n+=3
        #for i in range(len(nazwy2)):
        #    n+=1
        #    sheet1.write(n, 0 +4*kolejna, nazwy2[i])
        #    sheet1.write(n, 1 +4*kolejna, wartosci2[i])  
        #for i in range(len(nazwy3)):
        #    n+=1
        #    sheet1.write(n, 0 +4*kolejna, nazwy3[i])
        #    sheet1.write(n, 1 +4*kolejna, wartosci3[i])
        #    sheet1.write(n, 2 +4*kolejna, odchyleniewartosci3[i])   
        #    

        if np.mean(f1) > bestF1:
            bestC = C1
            bestF1 = np.mean(f1)
            bestF1std = np.std(f1)
            bestacc = np.mean(acc)
            bestaccstd = np.std(acc)
            bestrec = np.mean(rec)
            bestrecstd = np.std(rec)
            bestpres = np.mean(pres)
            bestpresstd = np.std(pres)
            

    nazwy4=['bestC','bestF1', 'bestacc', 'bestrec', 'bestpres']
    wartosci4=[float(bestC),float(bestF1), float(bestacc),float(bestrec), float(bestpres)]
    wartoscistd4=['',float(bestF1std), float(bestaccstd),float(bestrecstd), float(bestpresstd)]
    n+=2
    for i in range(len(nazwy4)):
        n+=1
        sheet1.write(n, 0 +4*kolejna, nazwy4[i])
        sheet1.write(n, 1 +4*kolejna, wartosci4[i])
        sheet1.write(n, 2 +4*kolejna, wartoscistd4[i])


D=[(0,1)]
#oryginalne +rotzkol
D1=[(0,1),(2,1),(9,1),(10,1)]
#oryginalne+rot
D2=[(0,1),(11,1),(12,1),(13,1)]
#oryginalne + jasnosc + ostrosc + kontrast + kolor
D3=[(0,1),(1,1),(4,1),(5,1),(6,1)]

'''
D1=[(0,4),(2,0),(9,0),(10,0)]
#oryginalne+rot
D2=[(0,4),(11,0),(12,0),(13,0)]
#oryginalne + jasnosc + ostrosc + kontrast + kolor
D3=[(0,4),(1,0),(4,0),(5,0),(6,0)]
'''
y=np.load("etykiety_eksp1.npy")
ile=2000
folds=5

#s0=np.random.randint(1000)
s0 = 1000

book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("oryginalne_1dodaatkowy")
petlaCrossWalidacji(D,y,folds,ile,testSrednich,ratio_test=1,kolejna = 1)
petlaCrossWalidacji(D,y,folds,ile,testKolmogorowaSmimrnowa,ratio_test=1)
print('zakonczono')
for i in range(1,15):
    Di=[(0,1),(i,1)]
    petlaCrossWalidacji(Di,y,folds,ile,testKolmogorowaSmimrnowa,ratio_test=1,kolejna = i+1)
    petlaCrossWalidacji(Di,y,folds,ile,testSrednich,ratio_test=1,kolejna = i + 16)
    print('zakonczono ', i)


sheet1 = book.add_sheet("oryginalne_kilka")
petlaCrossWalidacji(D,y,folds,ile,testKolmogorowaSmimrnowa,ratio_test=1)
petlaCrossWalidacji(D,y,folds,ile,testSrednich,ratio_test=1,kolejna = 1)
print('zakonczono')

petlaCrossWalidacji(D1,y,folds,ile,testKolmogorowaSmimrnowa,ratio_test=1,kolejna = 2)
petlaCrossWalidacji(D1,y,folds,ile,testSrednich,ratio_test=1,kolejna = 3)
print('zakonczono')

petlaCrossWalidacji(D2,y,folds,ile,testKolmogorowaSmimrnowa,ratio_test=1,kolejna = 4)
petlaCrossWalidacji(D2,y,folds,ile,testSrednich,ratio_test=1,kolejna = 5)
print('zakonczono')

petlaCrossWalidacji(D3,y,folds,ile,testKolmogorowaSmimrnowa,ratio_test=1,kolejna = 6)
petlaCrossWalidacji(D3,y,folds,ile,testSrednich,ratio_test=1,kolejna = 7)
print('zakonczono')

#book.save("svm.xls")



#book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("ile")
ile_bedzie=[500, 1000, 1500, 2000, 3000, 4096]
Dkilka=[D, D1, D2, D3]
for j in range(len(Dkilka)):
    for i in range(len(ile_bedzie)):
        petlaCrossWalidacji(Dkilka[j],y,folds,ile_bedzie[i],testKolmogorowaSmimrnowa,ratio_test=1,kolejna = 12*j+(i+6))
        petlaCrossWalidacji(Dkilka[j],y,folds,ile_bedzie[i],testSrednich,ratio_test=1,kolejna = 12*j+i)
        print ("zakonczono")

nazwa = str(s0)
nazwa2 = "test %s.xls" %(nazwa)
book.save(nazwa2)


