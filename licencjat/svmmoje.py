import numpy as np
#import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.svm import SVC
from sklearn.cross_validation import KFold, cross_val_score
from scipy import stats


#skalowanie danych wejsciowych:

def sigmoid_scaler2(A):
    B = np.divide(np.ones(A.shape),(1.0 + np.exp((-1.0) * A)))
    return B

def testKolmogorowaSmimrnowa(test_index,y,gdzie1,gdzie0):
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
    print str(nn[0,0]*100)+"%", str(nn[ile, 0]*100)+"%"
    return ind

def testSrednich(test_index,y,gdzie1,gdzie0):
    true_test=np.append(gdzie1[test_index], gdzie0[test_index])
    y_test=y[true_test].astype(int)
    notest=np.setdiff1d(np.arange(y.size),true_test) #indeksy obrazow nie wzietych do testu
    nn=[[iq,jq] for iq,jq in zip(np.abs(np.mean(X[np.intersect1d(gdzie1,notest)], axis=0)
            -np.mean(X[np.intersect1d(gdzie0, notest)], axis=0)), np.arange(X.shape[1]))]
    nn=np.array(sorted(nn, key=lambda m: m[0], reverse=True))
    ind=nn[:,1].astype(int)
    return ind

#wczytywanie danych wejsciowych i wyjsciowych

#katalog="/dmj/fizmed/jginter/CNN/Czerniak1/eksperyment1/"
katalog=""
X=np.load(katalog+"vgg_cnn_s_fc6.npy")
#? Jakie chcemy dodac dodatkowe obrazy?
X1=np.load(katalog+"vgg_cnn_s_fc6_kolor.npy")
X2=np.load(katalog+"vgg_cnn_s_fc6_rot90kol.npy")
X3=np.load(katalog+"vgg_cnn_s_fc6_gray.npy")
X4=np.load(katalog+"vgg_cnn_s_fc6_jasnosc.npy")
X5=np.load(katalog+"vgg_cnn_s_fc6_ostrosc.npy")
X6=np.load(katalog+"vgg_cnn_s_fc6_kontrast.npy")
X7=np.load(katalog+"vgg_cnn_s_fc6_left_right.npy")
X8=np.load(katalog+"vgg_cnn_s_fc6_kolory_random.npy")
X9=np.load(katalog+"vgg_cnn_s_fc6_rot180kol.npy")
X10=np.load(katalog+"vgg_cnn_s_fc6_rot270kol.npy")
X11=np.load(katalog+"vgg_cnn_s_fc6_rotacja90.npy")
X12=np.load(katalog+"vgg_cnn_s_fc6_rotacja180.npy")
X13=np.load(katalog+"vgg_cnn_s_fc6_rotacja270.npy")
X14=np.load(katalog+"vgg_cnn_s_fc6_top_botton.npy")

y=np.load("etykiety_eksp1.npy")

#?
#od numeru 1163 (504) do 9860 (662) dziwne obrazy, same zera - Czy usunac z obliczen?
#X=np.append(X[0:503], X[662:], axis=0)
#X1=np.append(X[0:503], X1[662:], axis=0)
#X2=np.append(X[0:503], X2[662:], axis=0)
#X3=np.append(X[0:503], X3[662:], axis=0)
#y=np.append(y[0:503], y[662:])

print "Wymiar wejscia: ",np.shape(X)
print "Wymiar etykiet: ",np.shape(y)


#? ile parametrow wybrac? Uwaga! wcale nie im wiecej tym lepiej.
ile=2000
print "Tyle parametrow zostawiono:", ile


#cross validation
folds=5 #na ile podzbiorow podzielic dane
gdzie1=np.where(y==1)[0]
gdzie0=np.where(y==0)[0]
np.random.shuffle(gdzie1)
np.random.shuffle(gdzie0)

print "tyle jedynek", gdzie1.size
print "tyle zer", gdzie0.size


def petlaCrossWalidacji(X,X1,X2,X3,y,gdzie1,gdzie0,folds,ile,test):
    kf=KFold(gdzie1.size,n_folds=folds)
    """D to lista krotek numery i ratio (ilosc 0 w stosunku do 1. zawsze bierze wszystkie 1 i jesli R=1 tyle samo zer, jesli r>1 wiecej zer niz jedynek, r<1 mniej zer od jedynek)
    y tablica etykiet
    folds int ile folderow
    ile int ile parametrow
    test funkcja np.srednie i kolmogorow
    """

    #glowna petla cross-walidacyjna
    print "Rozpoczynanie petli cross walidacyjnej"
    print "test: ", test
    bestiF1=0
    bestF1=0
    for i in xrange(0,9):
        #inicjalizacja trafnosci, precyzji, recall i f1 score
        acc=np.zeros(folds)
        pres=np.zeros(folds)
        rec=np.zeros(folds)
        f1=np.zeros(folds)
        counter=0
    
        for train_index,test_index in kf:

            #! wybor parametrow najlepiej roznicujacych dwie klasy
            if test==0:
                ind=testKolmogorowaSmimrnowa(test_index,y,gdzie1,gdzie0)
            if test==1:            
                ind=testSrednich(test_index,y,gdzie1,gdzie0)
    
            X_scaled = sigmoid_scaler2(X[:,ind[:ile]])
            X1_scaled = sigmoid_scaler2(X1[:,ind[:ile]])
            X2_scaled = sigmoid_scaler2(X2[:,ind[:ile]])
            X3_scaled = sigmoid_scaler2(X3[:,ind[:ile]])

            true_train=np.append(gdzie1[train_index], gdzie0[train_index])
            #? Jakie obrazy oryginalne dosypujemy?
            #true_train=np.append(true_train, gdzie0[gdzie1.size:gdzie1.size+int(0.2*train_index.size)])

            X_train=X_scaled[true_train]
            y_train=y[true_train]        

            #? Jakie obrazy przerobione dosypujemy? Jesli zadne to wpisac []
            tt1=[] #np.append(gdzie1[train_index], gdzie0[train_index+train_index.size*3])
            tt2=[] #np.append(gdzie1[train_index], gdzie0[train_index+train_index.size*2])
            tt3=[] #np.append(gdzie1[train_index], gdzie0[train_index]) #+train_index.size]) #gdzie1[train_index]
        
            X_train=np.append(X_train, X1_scaled[tt1], axis=0)
            X_train=np.append(X_train, X2_scaled[tt2], axis=0)
            X_train=np.append(X_train, X3_scaled[tt3], axis=0)
            y_train=np.append(y_train, np.append(y[tt1], np.append(y[tt2], y[tt3])))
        
            ind2=np.arange(y_train.size)
            np.random.shuffle(ind2)
            X_train=X_train[ind2]
            y_train=y_train[ind2]

            true_test=np.append(gdzie1[test_index], gdzie0[test_index])
            #? Na ktorych danych testujemy: oryginalnych czy dosypanych?
            X_test=X_scaled[true_test]
            y_test=y[true_test].astype(int)
    
            #! Wartosc C warto zwiekszac logarytmicznie
            C1=10**(float(i)/3)
            svm=SVC(kernel='rbf', C=C1)
            svm.fit(X_train,list(y_train))
            params=svm.get_params()
    
            pred=svm.predict(X_test)
            acc[counter] = accuracy_score(y_test,pred)
            pres[counter] = precision_score(y_test,pred)
            rec[counter] = recall_score(y_test,pred)
            f1[counter] = f1_score(y_test,pred)
            counter += 1
    
        print i*1, "C=", C1
        print "treningowych: ", X_train.shape, "testowych: ", X_test.shape
        print "treningowych jedynek:", np.sum(y_train)
        print "Finalowa trafnosc SVMa: ", np.mean(acc), "+/-", np.std(acc)
        print "Finalowe F1 score SVMa: ", np.mean(f1), "+/-", np.std(f1)
        print "Finalowe recall SVMa: ", np.mean(rec), "+/-", np.std(rec)
        print "Finalowe precision SVMa: ", np.mean(pres), "+/-", np.std(pres)
        print "---"
        if np.mean(f1)>bestF1:
            bestiF1=i*1
            bestF1=np.mean(f1)
    
    print 'rbf'
    print bestiF1
    print bestF1

#petlaCrossWalidacji(X,X1,X2,X3,y,gdzie1,gdzie0,folds,ile,0)

petlaCrossWalidacji(X,X1,X2,X3,y,gdzie1,gdzie0,folds,ile,1)
