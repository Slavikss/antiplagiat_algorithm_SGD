## Algorithm for plagiary detection based on bag of words, tfidf, SGD classifier

#### Example of train: `python3 train.py files plagiat1 plagiat2 --model model.pkl`, where:

1. `train.py` - script for model train

2. `files` - directory with original files

3. `plagiat1(2)` - directories with plagiary files

4. `model.pkl` - model's weights filename after train



#### Example of detection: `python3 compare.py input.txt scores.txt --model model.pkl `,where:

1. `compare.py` - script for comparing files

2. `input.txt` - text file with files to compare in format

   "`original/original.py plagiat/plagiat.py`"

3. `scores.txt` - text file with plagiary probability

4. `model.pkl` - model's weights




The task wasa made for <ins>Tinkoff's deep learning </ins> course





