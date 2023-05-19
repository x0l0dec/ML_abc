import os
import numpy as np
from PIL import Image               # Pillow, для чтения картинок
import pandas as pd
import random as rand


nx = 28; ny = 28    # размер букв (ну, мало ли)
ss = 'out.json'     # выходной файл

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
            if (z >= 6000):                             # лично у меня не хватает компа, чтобы обучать его на всех данных, поэтому нужно выставить некое ограничение на количество букв
                break
            i = int(i)
            for j in np.arange(0,len(img) / ny):
                j = int(j)
                imgs.append(img[i:i + nx,j:j + ny].ravel().tolist())    # двумерный массив преобразуем в одномерный, потому что не получается передать двумерные данные в модель (хотя так питон работает ещё медленнее)
                ami.append(a)
                z = z + 1


testimgs = []        # сюда будем сохранять тестовые картинки
testami = []

# перебор всех файлов в тестовой папке
for root, dirs, files in os.walk("test"):
    for f in files:
        testami.append(f.split('.')[0])
        f = 'test/' + f
        print('> чтение', f)
        img = np.asarray(Image.open(f).convert('L'))    # загрузка изображения
        testimgs.append(img.ravel().tolist())           # т.к. изображения отдельные, то их и не нужно разбивать на отдельные картинки с буквами, как для тренировочной папки


# модель случайного леса (у меня получилось на ней лучше всего)
from sklearn.ensemble import RandomForestClassifier as rfc
model = rfc()                                        # аргументы для модели, для большей точности: n_estimators=1000, random_state=0 (но это невозможно долго)
print('>> крайне долгое обучение модели...')
model.fit(imgs,ami)
print('>> модель обучена, распознаём тестовые данные')
ypred = model.predict(testimgs)                     # пытаемся предсказать по тестовым буквам
print('>> тестовые данные распознаны, они будут записаны в', ss)


import json     # для ответа нужен специфический формат

out = dict(sorted({testami[z] : ypred[z] for z in range(len(ypred))}.items(), key=lambda i: int(i[0]))) # не спрашивай
with open(ss, 'w') as f:    # запись ответа в файл
    json.dump(out, f)
