from os import listdir
from os.path import isfile, join


# Создается файл с целевой переменной каждого файла, где 0 - оригинал, 1 - плагиат
def make_target(indir1, indir2, indir3):
    with open('target.txt', 'w') as target:
        for file in [indir1, indir2, indir3]:
            for f in listdir(file):
                if isfile(join(file, f)):
                    target.write(f"{file +'/'+ f},{0 if file == indir1 else 1}" + '\n')
