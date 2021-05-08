# licencjat

Część programów użytych do stworzenia pracy licencjackiej.
 Program napisano w języku Python 2.7. Biblioteki, któreużyto to NumPy, Caffe, scikit-learn, Image.

W niniejszej pracy jako dane eksperymentalne użyto 900 zdjęć uzyskanych w dermatosko-pie, a wykonanych aparatem fotograficznym zmian barwnikowych skóry w formacie JPEG.Obrazy te były podzielone na zmiany barwnikowe złośliwe (173 zdjęcia) oraz zmiany barwni-kowe niezłośliwe (727 zdjęcia). Obrazy zostały pobrane z International Skin Imaging Colla-boration, ISIC: http://isdis.net/isic- pro-ject/.

Dodatkowo w pracy użyto wytrenowaną wcześniej sieć konwolucyjną udostępnioną przezBVLC (Berkeley Vision and Learning Center). Modele pochodziły z Image Large Scale VisualRecognition Challenge, ILSVRC. Sieci te zostały wytrenowane na bazie danych z ImageNet,15
zawierające fotografie z 1000 różnych obiektów [ Olga  Russakovsky,  Jia  Deng,  Hao  Su,  Jonathan  Krause,  Sanjeev  Satheesh,  Sean  Ma,Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg,and Li Fei-Fei.  Imagenet large scale visual recognition challenge.International Journalof Computer Vision, 115(3):211–252, 2015.]. Model sieci wykorzystany w niniejszejpracy  nazywa  się  VGG-CNN-S,  i  został  wytworzony  na  potrzeby  ILSVRSC  2012  [19]  Siećzbudowana jest z 5 warstw konwolucyjnych - CONV i 3 warstw w pełni połączonych - FC.Dwie  warstwy  FC  o  długości  4096  i  jednej  o  długości  1000,  która  stanowi  wyjście  z  sieci

W niniejszej pracy sieci konwolucyjne zostały użyte do parametryzacji cech obrazów, a klasyfikator maszyny wektorów nośnych do rozdzielenia danych na odpowiadające im klasy

SVM pozwala na uczenie za pomocą zbioru mniej licznego odliczby parametrów parametrów cech obrazu.

zastosowano także sztuczne powiększanie obrazów uczących. (Data augmentation)
