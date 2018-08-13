import numpy as np
import os
import glob
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras import backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
from keras.models import model_from_json


image_width, image_height = 50, 50

MODEL_NAME = 'asl_convnet'

test = True


def load_data(data_path):
    x = []
    y = []
    dir_list = os.listdir(data_path)
    class_count = len(dir_list)
    for dir_name in dir_list:
        # print(dir_name)
        images = glob.glob(os.path.join(data_path, dir_name, '*.jpg'))
        for image_path in images:
            image = load_img(image_path)
            img_array = img_to_array(image)
            x.append(img_array)
            y.append(int(dir_name))
        # x.extend(images)
    return np.array(x), np.array(y), class_count


def build_model(class_count):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(50, 50, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_count, activation='softmax'))
    return model


def train_model(model, x, y, class_count):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=42)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.fit(
        x_train,
        to_categorical(y_train, class_count),
        epochs=20,
        # steps_per_epoch=200,
        validation_data=(x_test, to_categorical(y_test, class_count)),
        # validation_steps=80,
        batch_size=500
    )
    scores = model.evaluate(x_test, to_categorical(y_test, class_count), verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


# Export pb file of model for use with android
def export_model(model, input_node_names, output_node_name):
    saver = tf.train.Saver()
    tf.train.write_graph(K.get_session().graph_def, 'out/', MODEL_NAME + '_graph.pbtxt')
    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt',
                              None,
                              False,
                              'out/' + MODEL_NAME + '.chkp',
                              output_node_name,
                              "save/restore_all",
                              "save/Const:0",
                              'out/frozen_' + MODEL_NAME + '.pb',
                              True,
                              ""
                              )
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def,
                                                                         input_node_names,
                                                                         [output_node_name],
                                                                         tf.float32.as_datatype_enum)
    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("Model Saved!")


def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model")


def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model")
    return loaded_model


def load_test_images(path):
    image_paths = glob.glob(os.path.join(path, '*.jpg'))
    images = []
    labels = []
    for image_path in image_paths:
        image = load_img(image_path)
        img_array = img_to_array(image)
        images.append(img_array)
        label_with_dir = os.path.splitext(image_path)[0]
        label = label_with_dir.split("\\")[1]
        labels.append(label)
    return np.asarray(images), labels


if __name__ == '__main__':
    if not test:
        x, y, class_count = load_data('data/letters-blackwhite')
        model = build_model(class_count)
        train_model(model, x, y, class_count)
        # export_model(model, ["conv2d_1_input"], "dense_2/Softmax")
        save_model(model)
    else:
        test_x, test_y = load_test_images("test-images")
        model = load_model()
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        scores = model.evaluate(test_x, to_categorical(test, 26), verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))






