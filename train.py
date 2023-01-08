from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from target import make_target
from pathlib import Path
from sklearn.pipeline import Pipeline
import argparse
import joblib
import io
import re
import random

random.seed(4)


# Функция чтения файла
def read_file(file_path):
    file = []
    f = io.open(Path(file_path), 'r', encoding='utf-8')
    for string in f:
        string = string.strip()
        if string != '':
            file.append(string)

    return file


# препроцессинг файла
def preprocess(file):
    # Удаляем всё, что находится внутри квадратных и фигурных скобок
    file = [re.sub(r"\[.*\]|\{.*\}", "", string) for string in file]

    # Удаляем знаки пунктуации
    file = [re.sub(r'[^\w\s]', '', string) for string in file if
            string.count('"""') == string.count("'''") == string.count('#') == 0]

    # Переводим всё в нижний регистр
    file = [string.lower() for string in file]

    # Удаляем нестандартные буквы, которые не входят в ascii
    file = [''.join([ch for ch in string if ch.isascii()]).strip() for string in file]

    # Убираем пустые строки
    file = [string for string in file if string != '']
    return ' '.join(file)


# Функция создания предобработанного датасета с таргетами
def make_train_df(indir1, indir2, indir3):
    # создание файла с путями до файлов для обучения и их меток
    make_target(indir1, indir2, indir3)
    X = []
    y = []
    # Чтение файлов и их предобработка
    with open('target.txt', 'r') as f:
        for string in f.readlines():
            file, target = string.strip().split(',')
            X.append(preprocess(read_file(file)))
            y.append(target)

    # Перемешивание семплов для более равномерного обучения
    random.shuffle(X)
    random.shuffle(y)

    return X, y


def train(indir1, indir2, indir3, model_name):
    X, y = make_train_df(indir1, indir2, indir3)

    # Пайплайн из countvectorizer'a, tfidftransformer'a, и классификатора(SGD, выдающего вероятности классов,
    # в нашем случае вероятность плагиата)
    model = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('sgd', SGDClassifier(loss='modified_huber', penalty='l2',
                              alpha=1e-3, random_state=42,
                              max_iter=5, tol=None))])

    # Обучение модели
    model.fit(X, y)

    # Accuracy на трейне - 0.95
    # predicted = model.predict(X)
    # print(sum(predicted == y) / len(y))

    print(f'Model trained,model_mame - {model_name}')

    # Сохранение модели
    joblib.dump(model, model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Files to process')
    parser.add_argument('indir1', type=str, help='Input dir for files')
    parser.add_argument('indir2', type=str, help='Input dir for files')
    parser.add_argument('indir3', type=str, help='Input dir for files')
    parser.add_argument('--model', type=str, help='Model weights path')
    args = parser.parse_args()

    train(args.indir1, args.indir2, args.indir3, args.model)
