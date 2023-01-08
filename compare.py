from pathlib import Path
from train import read_file, preprocess
import argparse
import joblib


# Основный блок, где читаются файлы для сравнения и записывается их результат
def main(input_dir, output_dir, model):
    assert model is not None, "Model weights path wasn't given"

    # Чтение путей до тестовых файлов
    files_to_check = [files.split()[1] for files in read_file(Path(input_dir))]

    # загрузка предобученной модели
    model = joblib.load(open(Path(model), "rb"))

    # Запись предиктов в файл
    with open(Path(output_dir), 'w') as out:
        for file in files_to_check:
            out.write(str(model.predict_proba([preprocess(read_file(file))])[0][0]))
            out.write('\n')

    print(f'Model predicted,score file - {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Files to process')
    parser.add_argument('indir', type=str, help='Input dir for files')
    parser.add_argument('outdir', type=str, help='Output dir for score file')
    parser.add_argument('--model', type=str, help='Model weights path')
    args = parser.parse_args()

    main(args.indir, args.outdir, args.model)
