import os
import numpy as np
from PIL import Image

def global_epoch(file_path, update=None):
    if not update:
        try:
            f = open(file_path, 'r', encoding='utf8')
            val = int(f.read())
            f.close()
            return val
        except:
            f = open(file_path, 'w', encoding='utf8')
            f.write(str(0))
            f.close()
            return 0
    else:
        f = open(file_path, 'w', encoding='utf8')
        f.write(str(update))
        f.close()

def load_dataset(data_dir, folder, size, annotation_file, generate_sentiment_to_index=False):
    sentiment = {}
    if generate_sentiment_to_index:
        sentiment2index = {}
        index = 0
    for row in annotation_file.readlines():
        row = row.strip().split()
        sentiment[row[0]] = row[1]
        if generate_sentiment_to_index:
            if row[1] not in sentiment2index:
                sentiment2index[row[1]] = index
                index += 1
    annotation_file.close()
    dataset = []
    img_iterator = []
    for img in os.listdir(data_dir + folder):
        im = Image.open(os.path.join(data_dir + folder, img))
        im = np.asarray(im)

        im = im.astype('float32')
        im = im / 255.0
        im = im - 0.5
        im = im * 2.0

        dataset.append(im.reshape(size))
        img_iterator.append(img)
    if generate_sentiment_to_index:
        return dataset, img_iterator, sentiment, sentiment2index
    return dataset, img_iterator, sentiment

def load_fer2013(data_dir):
    batch = []
    label = []
    dataset = open(data_dir + 'fer2013.csv', 'r')
    for row in dataset.readlines()[1:]:
        row = row.strip().split(',')
        label.append(int(row[0]))
        img = np.array(np.fromstring(row[1], np.uint8, sep=' '))
        shape = int(np.sqrt(img.shape[0]))
        img = img.astype('float32')
        img = img.reshape((shape, shape, 1))

        img = img / 255.0
        img = img - 0.5
        img = img * 2.0

        batch.append(img)

    return np.array(batch), np.array(label)


