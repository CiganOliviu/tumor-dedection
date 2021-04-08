import keras
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from join_data import data, result
from keras.callbacks import ModelCheckpoint

x_train, x_test, y_train, y_test = train_test_split(data, result, test_size=0.2, shuffle=True, random_state=0)

model = Sequential()

model.add(Conv2D(32, kernel_size=(2,2), input_shape=(128, 128, 3), padding='Same'))
model.add(Conv2D(32, kernel_size=(2,2), activation='relu', padding='Same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='Adamax')

checkpointer = ModelCheckpoint(filepath='save/model.h5', verbose=1, save_best_only=True)

epochs = 50
batch_size = 128

history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data= (x_test, y_test),
                    callbacks=[checkpointer],
                    verbose=2
                    )

model_json = model.to_json()
with open('save/model.json','w') as json_file:
    json_file.write(model_json)