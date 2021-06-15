import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

from tkinter import Tk, filedialog, Label

# Parametry globalne
import_verbose = True  # True - dodatkowe informacje dotyczące importu danych są wyświetlane. False - brak komunikatów
batch_size = 50  # Liczba zdjęć w jednej procesowanej paczce danych wejściowych
image_size = (100, 100)  # Wielkość wczytywanego zdjęcia = 100 x 100 pilkseli
image_shape = (100, 100, 3)  # Format zdjęcia = wielkość zdjęcia x zapis koloru w RGB = 100 x 100 x 3


########################################################################################################################
#   Import danych treningowych i testowych wraz z etykietami z dysku lokalnego
########################################################################################################################
def import_data(main_folder):
    # Dane do szkolen (zdjęcia warzyw i owoców) przechowywane są odpowiednio w katalogu 'Training' i 'Test', które
    # mają tą samą strukturę pokazaną poniżej. Dla każdego gatunku owoca lub warzywa stworzony jest podkatalog,
    # którego nazwa będzie używana jako etykieta.

    # Training/Test
    # |________ Etykieta 1
    # |                  |____ Zdjecie 1
    # |                  |____ Zdjecie 2
    # |                  |____ ...
    # |                  |____ Zdjecie X
    # |________ Etykieta 2
    # |                  |____ Zdjecie 1
    # |                  |____ Zdjecie 2
    # |                  |____ ...
    # |                  |____ Zdjecie N
    # ...
    # |________ Etykieta M
    # |                  |____ Zdjecie 1
    # |                  |____ Zdjecie 2
    # |                  |____ ...
    # |                  |____ Zdjecie N

    # Tworzenie ścieżek do katalogów 'Training' i 'Test'
    train_image_folder = os.path.join(main_folder, 'Training')
    test_image_folder = os.path.join(main_folder, 'Test')

    # Import etykiet polega na wczytaniu wszystkich nazw podkatalogów z katalogu 'Training'
    labels = os.listdir(train_image_folder)
    print('Liczba wczytanych etykiet = ', len(labels))
    if import_verbose:
        print('Wczytane etykiety - początek')
        for i in range(0, len(labels)):
            print(i + 1, labels[i])
        print('Wczytane etykiety - koniec')

    # Import zdjęć i utworzenie zbioru treningowego i testowego

    # Kreator klasy ImageDataGenerator posiada wiele parametrów określających jak wczytywane zdjęcia mają być
    # przetwarzane np. obracanie, zmiana wielkości, zmiana nasycenia kolorami, itd.
    train_data_generator = ImageDataGenerator()
    test_data_generator = ImageDataGenerator()

    # Metoda Flow_from_directory klasy ImageDataGenerator umożliwia import zdjęć zapisanych w odpowiedniej strukturze
    # folderów (zobacz przykład struktury przedstawionej powyżej)
    train_set = train_data_generator.flow_from_directory(
        train_image_folder,  # Ścieżka do katalogu ze zdjęciami
        target_size=image_size,  # Wielkość wczytywanego zdjęcia
        class_mode='categorical',  # 'binary' jeżeli mamy tylko dwie etykiety, 'categorical' jeżeli jest wiele etykiet
        batch_size=batch_size,  # Liczba zdjęć w procesowanym podzbiorze wszystkich zdjęć
        shuffle=True,  # True - zdjęcia będą losowo mieszane w całym zbiorze
        classes=labels)  # Przypisanie etykiet do zdjęć

    test_set = test_data_generator.flow_from_directory(
        test_image_folder,
        target_size=image_size,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False,
        classes=labels)

    return train_set, test_set


########################################################################################################################
#   Utworzenie modelu numer 1.
########################################################################################################################
def create_model_1(input_shape, num_classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(500, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


########################################################################################################################
#   Utworzenie modelu numer 2.
########################################################################################################################
def create_model_2(input_shape, num_classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(2, 2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(2, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


########################################################################################################################
#   Wybór modelu, który ma zostać utworzony.
#   Docelowo będzie można wybrać jeden model z wielu różnych w celu ich porównania.
########################################################################################################################
def create_model(selected_model, input_shape, num_classes):
    # Wybór modelu do utworzenia
    if selected_model == 1:
        model = create_model_1(input_shape, num_classes)
        return model
    elif selected_model == 2:
        model = create_model_2(input_shape, num_classes)
        return model
    else:
        # Tutaj będą zdefiniowane pozostałe modele.
        return None


########################################################################################################################
# create a confusion matrix to visually represent incorrectly classified images
########################################################################################################################
def plot_confusion_matrix(y_true, y_pred, classes, out_path=""):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(40, 40))
    ax = sn.heatmap(df_cm, annot=True, square=True, fmt="d", linewidths=.2, cbar_kws={"shrink": 0.8})
    if out_path:
        plt.savefig(
            out_path + "/confusion_matrix.png")  # as in the plot_model_history, the matrix is saved in a file called "model_name_confusion_matrix.png"
    return ax


########################################################################################################################
# create 2 charts, one for accuracy, one for loss, to show the evolution of these two metrics during the training process
########################################################################################################################
def plot_model_history(model_history, out_path=""):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(
        np.arange(1, len(model_history.history['accuracy']) + 1))  # , len(model_history.history['accuracy']))
    axs[0].legend(['train'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1))  # , len(model_history.history['loss']))
    axs[1].legend(['train'], loc='best')
    # save the graph in a file called "acc.png" to be available for later; the model_name is provided when creating and training a model
    if out_path:
        plt.savefig(out_path + "/acc.png")
    plt.show()


########################################################################################################################
#   Trenowanie modelu.
########################################################################################################################
def train_model(model, model_folder, model_file, train_data, test_data, verbose=1, epochs=25):
    # Określenie zapisu modelu
    save_model = ModelCheckpoint(
        filepath=model_folder + "/" + model_file + '.h5',  # Lokalizacja i nazwa pliku
        monitor='loss',
        verbose=verbose,
        save_best_only=True,
        save_weights_only=False,  # True - zapis tylko wag, False - zapis całego modelu
        mode='min',
        save_freq='epoch')  # Częstotliwość zapisu modelu

    # Trenuj model sieci
    # with tf.device('/cpu:0'):
    history = model.fit(train_data,
                        epochs=epochs,
                        steps_per_epoch=(train_data.n // batch_size) + 1,
                        verbose=verbose,
                        callbacks=[save_model])

    model.load_weights(model_folder + "/" + model_file + '.h5')

    # Weryfikacja wytrenowanego modelu sieci na zbiorze treningowym i testowym
    train_data.reset()
    loss_t, accuracy_t = model.evaluate(train_data, steps=(train_data.n // batch_size) + 1, verbose=verbose)
    loss, accuracy = model.evaluate(test_data, steps=(test_data.n // batch_size) + 1, verbose=verbose)
    print("Train: accuracy = %f  ;  loss_v = %f" % (accuracy_t, loss_t))
    print("Test: accuracy = %f  ;  loss_v = %f" % (accuracy, loss))

    plot_model_history(history, out_path=model_folder)

    #  plot_model_history(history, out_path=model_out_dir)
    test_data.reset()
    y_pred = model.predict(test_data, steps=(test_data.n // batch_size) + 1, verbose=verbose)
    y_true = test_data.classes[test_data.index_array]
    plot_confusion_matrix(y_true, y_pred.argmax(axis=-1), test_data.class_indices, out_path=model_folder)
    class_report = classification_report(y_true, y_pred.argmax(axis=-1), target_names=test_data.class_indices)

    with open(model_folder + "/classification_report.txt", "w") as text_file:
        text_file.write("%s" % class_report)

    return None


########################################################################################################################
#   Funkcja główna.
########################################################################################################################
def main(selected_model, data_localization, retrain_model=True, verbose=1, epochs=25):
    # Utworzenie folderu dla modeli
    model_folder = os.path.join(data_localization, 'Models')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_file = 'model' + str(selected_model)

    model = None
    # Trenowanie sieci i zapis wyznaczonych wag albo odczyt wcześniej zapisanego modelu
    if retrain_model:
        # Import zdjęć z folderu dysku lokalnego
        train_data, test_data = import_data(data_localization)

        # Wybór i utworzenie modelu
        num_classes = train_data.num_classes  # Odczytynie liczby class, czyli liczby wczytanych etykiet
        model = create_model(selected_model, image_shape, num_classes)
        if model is not None:
            print(model.summary())
        else:
            print('\'Selected_model\' refers to unknown model number. Program is terminated.')
            return None
        train_model(model, model_folder, model_file, train_data, test_data, verbose, epochs)
        print('End of training.')
    else:
        if model:
            del model
        model = load_model(model_folder + '/' + model_file + '.h5')

    # Select picture
    selected_file = filedialog.askopenfile(initialdir=data_localization,
                                           mode="r",
                                           title='Select image',
                                           filetypes=[
                                               ('Image Files',
                                                ['.jpeg', '.jpg', '.png', '.gif', '.tiff', '.tif', '.bmp'])])

    # Import selected image as image to dislay in application
    image = tf.keras.preprocessing.image.load_img(selected_file.name, target_size=image_size)
    plt.imshow(image)
    plt.show()

    image = tf.keras.preprocessing.image.img_to_array(image)
    image_as_array = np.array([image])

    prediction = model.predict(image_as_array)
    print(np.argmax(prediction))
    test = prediction[0, 35]
    for i in range(0, prediction.size - 1):
        if prediction[0, i] > 0.01:
            print(i + 1, prediction[0, i].astype("float"))

    # # Import selected image as image to dislay in application
    # selected_image = Image.open(selected_file.name, mode='r')
    #
    # image = selected_image.resize((100,100))
    # img_arr = np.array(image)
    # plt.imshow(img_arr)
    # plt.show()
    # print(img_arr)
    # print(model.predict(img_arr))

    # print((model.predict(img_arr) > 0.5).astype("int32"))


########################################################################################################################
#   Wywołanie funkcji głównej z wybranym modelem sieci
########################################################################################################################

# selected_model = 1, 2, 3, ...
# numer modelu sieci, który ma być zbudowany

# data_localization
# Ścieżka do lokalnego katalogu z danymi

# retrain_model = True/False
# True - wykonaj trenowanie od początku, False - wczytaj wcześniej wytrenowany model

main(selected_model=2, data_localization="C:\AI\AI3\MiniProject1", retrain_model=True, verbose=1, epochs=5)
