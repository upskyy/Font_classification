import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=60,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        shear_range=3.5,
        #zoom_range=0.2,
        #horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest")

img = load_img("C:\\Python Project\\font_project\\MALGUN\\malgun_가.png")  
x = img_to_array(img)  # (3, 150, 150) 크기의 NumPy 배열
x = x.reshape((1,) + x.shape)  # (1, 3, 150, 150) 크기의 NumPy 배열

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="z", save_prefix="malgun_가", save_format="png"):
    i += 1
    if i > 40:
        break  # 이미지 40장을 생성하고 마침
