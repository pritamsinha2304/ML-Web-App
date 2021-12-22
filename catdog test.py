import tensorflow as tf

model = tf.keras.models.load_model('CatDog.h5')
img = tf.keras.preprocessing.image.load_img('C:\\Users\\prita\\Desktop\\cd_test\\2.jpg', target_size=(256, 256))

img_arr = tf.keras.preprocessing.image.img_to_array(img)
# img_arr = img_arr/255
print(img_arr)
img_dim = tf.expand_dims(img_arr, 0)
print(img_dim)
score = model.predict(img_dim)
print(
    "This image is %.2f percent cat and %.2f percent dog."
    % (100 * (1 - score), 100 * score)
)
# print(score.argmax(axis=-1))
