
import tensorflow as tf
from gc import callbacks
from my_utils import split_data, order_test_set,create_generators
from deeplearning_models import street_signs
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

if __name__=='__main__':

    if False:
        path = 'C:\\Users\\sergi\Documents\\PROGRAMAS APRENDIZAJE\\tensorflow\\signs_recognition\\Train'
        path_train = 'C:\\Users\\sergi\\Documents\\PROGRAMAS APRENDIZAJE\\tensorflow\\signs_recognition\\_training\\train'
        path_val = 'C:\\Users\\sergi\\Documents\\PROGRAMAS APRENDIZAJE\\tensorflow\\signs_recognition\\_training\\val'

        split_data(path, path_train, path_val)

        path_to_images = 'C:\\Users\\sergi\\Documents\\PROGRAMAS APRENDIZAJE\\tensorflow\\signs_recognition\\Test'
        path_csv = 'C:\\Users\\sergi\\Documents\\PROGRAMAS APRENDIZAJE\\tensorflow\signs_recognition\\Test.csv'
        order_test_set(path_to_images, csv_path=path_csv)

    path_train = 'C:\\Users\\sergi\\Documents\\PROGRAMAS APRENDIZAJE\\tensorflow\\signs_recognition\\_training\\train'
    path_val = 'C:\\Users\\sergi\\Documents\\PROGRAMAS APRENDIZAJE\\tensorflow\\signs_recognition\\_training\\val'
    path_test = 'C:\\Users\\sergi\\Documents\\PROGRAMAS APRENDIZAJE\\tensorflow\\signs_recognition\\Test'
    batch_size = 64
    epochs = 15

    train_generator, val_generator, test_generator = create_generators(batch_size, path_train, path_val, path_test)
    nbr_classes = train_generator.num_classes
    TRAIN = False
    TEST = True
    if TRAIN:
        path_to_save_model = './models'
        check_saver = ModelCheckpoint(
            path_to_save_model,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )

        check_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=10
        )


        model = street_signs(nbr_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_generator,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_generator,
                callbacks=[check_saver,check_stop]

        )

    if TEST:
        model = tf.keras.models.load_model('./models')
        model.summary()

        print('Evaluating Validation Set : ')
        model.evaluate(val_generator)

        print('Evaluating Test Set : ')
        model.evaluate(test_generator)