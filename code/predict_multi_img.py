from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf


X = []
Y = []

train_dir = "./train"
categories = ["baemin", "dxmovie", "maple", "malgun"]
nb_classes = len(categories)


for idx, text in enumerate(categories):
    
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

    image_dir = train_dir + "/" + text    
    files = glob.glob(image_dir+"/*.png")   #확장자가 png인 파일만 추출
    print(text, " 파일 길이 : ", len(files))
    for img in files:
        image = Image.open(img)
        image = image.resize((64, 64))
        data = np.array(image)
        X.append(data)
        Y.append(label)

X = np.array(X)
Y = np.array(Y)

#1 0 0 0 이면 baemin
#0 1 0 0 이면 dxmovie 
#0 0 1 0 이면 maple
#0 0 0 1 이면 malgun

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=3)
SEED = 3
np.random.seed(SEED)
tf.random.set_seed(SEED)

X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255

model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape= X_train.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
   
model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
    
model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model_dir = './model'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
     
# model.summary()

### checkpoint = ModelCheckpoint(filepath = model_dir , monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=4)
model.compile(optimizer=Adam(lr=0.0002),loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=20, epochs=10, 
        validation_data=(X_test, Y_test), callbacks=[early_stopping])

print("정확도 : %.3f" % (model.evaluate(X_test, Y_test)[1]))
    
model.save('my_best_model.h5')

x_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()
