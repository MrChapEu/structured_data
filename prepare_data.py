from scipy.misc import imread, imresize
import os
import numpy as np
from keras.utils.np_utils import to_categorical

def predict_batch(folder, img_size=None):
    img_list = []
    label_list = []

    for filename in os.listdir(folder):
        try:
            if filename.endswith(".jpg"):
                img = imread(os.path.join(folder,filename))
                lb = filename.split('_')[1]
                lb = lb.lower()
            
            if img_size:
                img = imresize(img,img_size)

            img = img.astype('float32')
            
            # convert image to greyscale
            img = img.sum(axis=2) / 3.
            img /= np.std(img)
            
            img = img[:, :, np.newaxis]
            
            img_list.append(img)
            label_list.append(lb)
        except:
            continue
    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        raise ValueError(
            'when both img_size and crop_size are None, all images '
            'in image_paths must have the same shapes.')

    #batch = preprocess_input(img_batch)
    return img_batch, label_list 
            #model.predict(img_batch)


YOUR_FOLDER_NAME = "Synth90k"

print("Images formatting")
X, labels = predict_batch(YOUR_FOLDER_NAME, (32, 100))

alphabet = "0123456789abcdefghijklmnopqrstuvwxyz "
maxLen = 23
def word_to_labels(w):
    w_to_int = []
    letters = list(w) + [" "]*(maxLen - len(w))
    for letter in letters :
        w_to_int.append(alphabet.index(letter))
    return np.array(w_to_int)

Y = []
for word in labels:
    Y.append(word_to_labels(word))
Y = np.array(Y)
Y = Y.T#la première ligne contient toutes les premières letters, la second toutes les secondes lettres ...

output_labels = []
for l in Y : 
    output_labels.append(to_categorical(l,37))


print("record")

data = {"X":X,"y":output_labels}

np.save("data",data)

print(y[0][0])
print(labels[0])

