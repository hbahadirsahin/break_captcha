from captcha.image import ImageCaptcha
from scipy import misc, ndimage
import numpy as np
from matplotlib import pyplot as plt


def generate_data(n=1000, max_digs=1, width=160):
    capgen = ImageCaptcha()
    data = []
    target=[]
    for i in range(n):
        x = np.random.randint(0, 10**max_digs)
        img = misc.imread(capgen.generate(str(x)))
        if 160 <= img.shape[1]:
            img = np.mean(img,axis=2)[:,:width]
            img = ndimage.zoom(img, 0.8)
            data.append(img.flatten())
            target.append(label_binarizer(x))
    return np.array(data), np.array(target)


def label_binarizer(label):
    res = np.zeros(5*10)

    for i, c in enumerate(str(label)):
        index = i * 10 + (ord(c) - 48)
        res[index] = 1
    return res

def pos_to_label(label):
    digit_pos = label.nonzero()[0]
    res = list()
    for i, c in enumerate(digit_pos):
        char_at_pos = i
        char_idx = c % 10
        char_code = char_idx
        res.append(str(char_code))
    return int("".join(res))

# data, target = generate_data(n=1, max_digs=5)
# plt.imshow(data[0].reshape(48,-1))
# plt.show()


