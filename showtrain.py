import os
import numpy as np
from PIL import Image               # Pillow, для чтения картинок


nx = 28; ny = 28    # размер букв (ну, мало ли)
imgs = []           # сюда будем сохранять тренировочные картинки
ami = []            # тут будет инфа о том, что на тренировочной картинке нарисовано

# перебор всех файлов в тренировочной папке
for root, dirs, files in os.walk("train"):
    for f in files:
        a = f.split('.')[0]
        f = 'train/' + f
        print('> чтение', f)
        img = np.asarray(Image.open(f).convert('L'))    # загрузка изображения
        print('тут букв:', int(len(img[0]) / nx * len(img) / ny), ': будут записаны как', a)
        z = 0
        for i in np.arange(0,len(img[0]) / nx):         # разделение изображения на картинки с буквами
            if (z >= 4000):                             # лично у меня не хватает компа, чтобы обучить его на всех данных, поэтому нужно выставить ограничение на количество букв
                break
            i = int(i)
            for j in np.arange(0,len(img) / ny):
                j = int(j)
                imgs.append(img[i:i + nx,j:j + ny].ravel().tolist())    # двумерный массив преобразуем в одномерный, потому что не получается передать двумерные данные в модель (хотя так питон работает ещё медленнее)
                ami.append(a)
                z = z + 1


# разделение данных на тестовые и основные
from sklearn.model_selection import train_test_split as tts
xtrain, xtest, ytrain, ytest = tts(imgs, ami, test_size=0.2, random_state=0)

# модель случайного леса (у меня получилось на ней лучше всего)
from sklearn.ensemble import RandomForestClassifier as rfc
model = rfc()                                        # аргументы для модели, для большей точности: n_estimators=1000, random_state=0 (но это невозможно долго)
print('>> крайне долгое обучение модели...')
model.fit(xtrain,ytrain)
print('>> модель обучена, отчёт на нерусском:')
ypred = model.predict(xtest)                        # предсказания

# выводы по модели, точность
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(ytest,ypred))
print('точность на своих же данных:', accuracy_score(ytest,ypred))
