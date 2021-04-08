import os
from PIL import Image
import numpy as np
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
encoder.fit([[0],[1]])

data = []
paths = []
result = []

for r, d, f in os.walk(r'./dataset/yes'):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for r, d, f in os.walk(r'./dataset/no'):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128, 128))
    img = np.array(img)
    if(img.shape == (128, 128, 3)):
        data.append(np.array(img))
        result.append(encoder.transform([[0]]).toarray())

data = np.array(data)
result = np.array(result)
result = result.reshape(139, 2)
print('Result Shape:',result.shape)
print('Data Shape:',data.shape)
