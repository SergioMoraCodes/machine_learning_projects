import tensorflow as tf
import numpy as np
def predict_with_model(model,img_path):

    image = tf.io.read_file(img_path) #read the image as a png file
    image = tf.image.decode_png(image, channels=3) #converts the png-image to a tensor, dtype=uint8, specify channels to the color of image
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) # converts the tensor int values (0-255) to float32 (0-1)
    image = tf.image.resize(image, [60,60]) # (60,60,3) #resize according to your model
    image = tf.expand_dims(image, axis=0) # (1,60,60,3) #add the axis that the layers receives as the batch_size
    predictions = model.predict(image) #returns a list of probabilities of the image belonging to one of the classes
                                       # [0.003, 0.0005, 0.99, 0.000 ...]
    predictions = np.argmax(predictions) # 2, index of the maximun value
    return predictions


if __name__=='__main__':
    
    img_path = 'C:\\Users\\sergi\\Documents\\PROGRAMAS APRENDIZAJE\\tensorflow\\signs_recognition\\Test\\21\\09792.png'
    model = tf.keras.models.load_model('./models')
    predict = predict_with_model(model,img_path)

    print(f'prediction = {predict}')