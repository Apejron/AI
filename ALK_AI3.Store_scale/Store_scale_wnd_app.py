import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
import os
import datetime
from sklearn.metrics import confusion_matrix, classification_report
from tkinter import *
from tkinter import Tk, ttk, filedialog, messagebox, IntVar
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model
from PIL import Image, ImageTk

########################################################################################################################
# defaults
########################################################################################################################
# Globals - constant
default_data_location = 'C:/AI/AI3/MiniProject1'
image_size = (400, 400)  # width and height of the used images
image_shape = (400, 400, 3)  # the expected input shape for the trained models; since the images in the Fruit-360
# are 100 x 100 RGB images, this is the required input shape
batch_size = 5
validation_percent = 0.3
learning_rate = 0.1  # initial learning rate
learning_rate_reduction_factor = 0.5  # the factor used when reducing
# the learning rate -> learning_rate *= learning_rate_reduction_factor
min_learning_rate = 0.00001  # once the learning rate reaches this value, do not decrease it further
patience = 3  # how many epochs to wait before reducing the learning rate when the loss plateaus
verbose = 1  # controls the amount of logging done during training and testing: 0 - none, 1 - reports metrics after each batch, 2 - reports metrics after each epoch

# Globals - variable
train_set = None
validation_set = None
test_set = None
labels_list = None
model = None
current_image_file_for_prediction = 'Init_image.jpg'


########################################################################################################################
# change_data_location - open folder structure in current location, allow user to select new folder
########################################################################################################################
def change_data_location(current_location):
    selected_folder = filedialog.askdirectory(initialdir=current_location.get())
    if selected_folder:
        current_location.set(selected_folder)


########################################################################################################################
# Randomly changes hue and saturation of the image to simulate variable lighting conditions
########################################################################################################################
def augment_image(x):
    #    x = tf.image.random_saturation(x, 0.9, 1.2)
    #    x = tf.image.random_hue(x, 0.02)
    return x


########################################################################################################################
# import_data_from_folders - import labels, training, validation and test datasets
########################################################################################################################
def import_data_from_folders(import_scope, data_main_folder, validation_percent=0.2, batch_size=50):
    # Images are stored in two folders - training and test - which has structeres are presented below.
    # Each folder includes subfolders which names are labels of class. One subfolder includes numbers of images
    # represents one class.

    # Training/Test
    # |________ Label 1
    # |                  |____ Image 1
    # |                  |____ Image 2
    # |                  |____ ...
    # |                  |____ Image X
    # |________ Label 2
    # |                  |____ Image 1
    # |                  |____ Image 2
    # |                  |____ ...
    # |                  |____ Image N
    # ...
    # |________ Label M
    # |                  |____ Image 1
    # |                  |____ Image 2
    # |                  |____ ...
    # |                  |____ Image N

    # create paths to 'Training' and 'Test' folder
    train_image_folder = os.path.join(data_main_folder, 'Training')
    test_image_folder = os.path.join(data_main_folder, 'Test')

    # Import subfolders name as a list of labels
    labels = os.listdir(train_image_folder)

    # given the train and test folder paths and a validation to test ratio, this method creates three generators
    #  - the training generator uses (100 - validation_percent) of images from the train set
    #    it applies random horizontal and vertical flips for data augmentation and generates batches randomly
    #  - the validation generator uses the remaining validation_percent of images from the train set
    #    does not generate random batches, as the model is not trained on this data
    #    the accuracy and loss are monitored using the validation data so that the learning rate can be updated if the model hits a local optimum
    #  - the test generator uses the test set without any form of augmentation
    #    once the training process is done, the final values of accuracy and loss are calculated on this set

    if import_scope == 1:
        train_data_generator = ImageDataGenerator(
            width_shift_range=0.0,
            height_shift_range=0.0,
            zoom_range=0.0,
            horizontal_flip=True,
            vertical_flip=True,  # randomly flip images
            preprocessing_function=augment_image,
            validation_split=validation_percent)  # percentage indicating how much of the training set should be kept
        # for validation

        test_data_generator = ImageDataGenerator()

        train_set_gen = train_data_generator.flow_from_directory(
            train_image_folder,
            target_size=image_size,
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=True,
            subset='training',
            classes=labels)

        validation_set_gen = train_data_generator.flow_from_directory(
            train_image_folder,
            target_size=image_size,
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=False,
            subset='validation',
            classes=labels)

        test_set_gen = test_data_generator.flow_from_directory(
            test_image_folder,
            target_size=image_size,
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=False,
            subset=None,
            classes=labels)
    else:
        train_set_gen = None
        validation_set_gen = None
        test_set_gen = None

    return train_set_gen, validation_set_gen, test_set_gen, labels


########################################################################################################################
# import_data - reset global variables, call import_data_from_folders, update global variables
########################################################################################################################
def import_data(import_scope, data_main_folder, verbose):
    global train_set
    global validation_set
    global test_set
    global labels_list
    train_set = None
    validation_set = None
    test_set = None
    labels_list = None

    training_counter.set(0)
    validation_counter.set(0)
    test_counter.set(0)
    labels_counter.set(0)

    train_set_gen, validation_set_gen, test_set_gen, labels = import_data_from_folders(
        import_scope,
        data_main_folder,
        validation_percent,
        batch_size=batch_size)
    if train_set_gen:
        train_set = train_set_gen
        training_counter.set(train_set.n)

    if validation_set_gen:
        validation_set = validation_set_gen
        validation_counter.set(validation_set.n)

    if test_set_gen:
        test_set = test_set_gen
        test_counter.set(test_set.n)

    if labels:
        labels_list = labels
        labels_counter.set(len(labels_list))

    if verbose == 1:
        messagebox.showinfo("Data import", "Data import is finished.")


########################################################################################################################
# update_selected_model_label - refresh model description based on selected model
########################################################################################################################
def update_selected_model_label(current_model_index):
    if current_model_index == 0:
        selected_model_description.set('2 x Dense')
    elif current_model_index == 1:
        selected_model_description.set('2 x Conv2D, 2 x MaxPool2D, 2 x Dense')
    elif current_model_index == 2:
        selected_model_description.set('4 x Conv2D, 4 x MaxPool2D, 3 x Dense, 2 x Dropout')
    elif current_model_index == 3:
        selected_model_description.set('Inception V3 from Keras')
    elif current_model_index == 4:
        selected_model_description.set('NASNetLarge from Keras')
    else:
        selected_model_description.set('Unknown model')


########################################################################################################################
# update_selected_model_label - refresh model description based on selected model
########################################################################################################################
def model_combobox_changed(*args):
    global selected_model_var
    selected_model_var.set(training_model_combo.current())
    update_selected_model_label(training_model_combo.current())


########################################################################################################################
#   create_model_1 - simple
########################################################################################################################
def create_model_1(input_shape, num_classes):
    model = tf.keras.models.Sequential(name='Simple')
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


########################################################################################################################
#   create_model_2 - medium
########################################################################################################################
def create_model_2(input_shape, num_classes):
    model = tf.keras.models.Sequential(name='Medium')
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
#   create_model_3 - complex
########################################################################################################################
def create_model_3(input_shape, num_classes):
    model = tf.keras.models.Sequential(name='Complex')
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(16, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv1'))
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid', name='pool1'))
    model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv2'))
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid', name='pool2'))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv3'))
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid', name='pool3'))
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv4'))
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid', name='pool4'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu', name='fcl1'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu', name='fcl2'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions'))
    optimizer = Adadelta(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


########################################################################################################################
#   create_model_4 - InceptionV3
########################################################################################################################
def create_model_4(input_shape, num_classes):
    global image_shape
    image_shape = (299, 299, 3)

    model = tf.keras.applications.InceptionV3(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=image_shape,
        pooling=None,
        classes=num_classes,
        classifier_activation="softmax",
    )
    optimizer = Adadelta(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


########################################################################################################################
#   create_model_5 - NASNetLarge
########################################################################################################################
def create_model_5(input_shape, num_classes):
    global image_shape
    image_shape = (331, 331, 3)

    model = tf.keras.applications.NASNetLarge(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=num_classes,
        # classifier_activation="softmax",
    )
    optimizer = Adadelta(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


########################################################################################################################
#   create_model - based on selected model call function to create proper model
########################################################################################################################
def create_model(selected_model, input_shape, num_classes):
    if selected_model == 0:
        return create_model_1(input_shape, num_classes)
    elif selected_model == 1:
        return create_model_2(input_shape, num_classes)
    elif selected_model == 2:
        return create_model_3(input_shape, num_classes)
    elif selected_model == 3:
        return create_model_4(input_shape, num_classes)
    elif selected_model == 4:
        return create_model_5(input_shape, num_classes)
    else:
        print('Not defined model')
        return


########################################################################################################################
#   plot_model_history
########################################################################################################################
def plot_model_history(model_history, training_goal, model_folder="", selected_model=0):
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

    # save the graph in a file called "acc.png" to be available for later
    if training_goal == 1:
        history_file = model_folder + '/' + 'M' + str(selected_model + 1) + '_ACCU_training_result.png'
    else:
        history_file = model_folder + '/' + 'M' + str(selected_model + 1) + '_LOSS_training_result.png'

    if model_folder:
        plt.savefig(history_file)
    plt.show()


########################################################################################################################
# run_training
########################################################################################################################
def run_training(selected_model, data_localization, training_goal, input_shape, epochs=25, verbose_training=0):
    # check if data were imported
    if train_set is None or validation_set is None or test_set is None or labels_list is None:
        if verbose_training:
            messagebox.showinfo("Run training", "Data was not imported properly.\nPlease run import.")
        return None

    loaded_model_file_name_var.set('none')
    model = create_model(selected_model, input_shape, len(labels_list))
    if model:
        print(model.summary())

    # paths and files
    model_folder = os.path.join(data_localization, 'Models')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    if training_goal == 1:
        monitor = 'accuracy'
        mode = 'max'
        model_prefix = 'M' + str(selected_model + 1) + '_ACCU'
    else:
        monitor = 'loss'
        mode = 'min'
        model_prefix = 'M' + str(selected_model + 1) + '_LOSS'

    model_file = model_folder + '/' + model_prefix + '_model.h5'

    learning_rate_reduction = ReduceLROnPlateau(monitor=monitor, patience=patience, verbose=verbose,
                                                factor=learning_rate_reduction_factor, min_lr=min_learning_rate)
    save_model = ModelCheckpoint(filepath=model_file, monitor=monitor, verbose=verbose,
                                 save_best_only=True, save_weights_only=False, mode=mode, save_freq='epoch')

    history = model.fit(train_set,
                        epochs=epochs,
                        steps_per_epoch=(train_set.n // batch_size) + 1,
                        verbose=verbose,
                        callbacks=[learning_rate_reduction, save_model])

    model.load_weights(model_file)

    # Trained model verification
    train_set.reset()
    loss_training, accuracy_training = model.evaluate(train_set, steps=(train_set.n // batch_size) + 1, verbose=verbose)
    loss_test, accuracy_test = model.evaluate(test_set, steps=(test_set.n // batch_size) + 1, verbose=verbose)

    training_accuracy.set(accuracy_training)
    training_loss.set(loss_training)
    test_accuracy.set(accuracy_test)
    test_loss.set(loss_test)

    print("Train: accuracy = %f  ;  loss_v = %f" % (accuracy_training, loss_training))
    print("Test: accuracy = %f  ;  loss_v = %f" % (accuracy_test, loss_test))

    plot_model_history(history, training_goal, model_folder, selected_model)

    test_set.reset()
    y_pred = model.predict(test_set, steps=(test_set.n // batch_size) + 1, verbose=verbose)
    y_true = test_set.classes[test_set.index_array]
    class_report = classification_report(y_true, y_pred.argmax(axis=-1), target_names=labels_list, zero_division=0)

    classification_report_file = model_folder + '/' + model_prefix + '_classification_report.txt'
    with open(classification_report_file, "w") as text_file:
        text_file.write("%s" % class_report)

    loaded_model_file_name_var.set(os.path.basename(model_file))

    if verbose_training:
        messagebox.showinfo("Model training", "Model training is finished.")

    return


########################################################################################################################
#  load_model_from_file
########################################################################################################################
def load_model_from_file(data_main_folder):
    # Select model file
    selected_model_file = filedialog.askopenfile(initialdir=data_main_folder + '/Models',
                                                 mode="r",
                                                 title='Select model to import',
                                                 filetypes=[
                                                     ('H5 model files', ['.h5'])])

    if selected_model_file:
        global model
        model = load_model(selected_model_file.name)
        loaded_model_file_name_var.set(os.path.basename(selected_model_file.name))
        if labels_list is None:
            import_data(0, data_main_folder, 0)
        messagebox.showinfo("Prediction", "Model is loaded.")


########################################################################################################################
#  import_image_from_file
########################################################################################################################
def import_image_from_file(data_location):
    # Select picture
    selected_file = filedialog.askopenfile(initialdir=data_location,
                                           mode="r",
                                           title='Select image',
                                           filetypes=[
                                               ('Image Files',
                                                ['.jpeg', '.jpg', '.png', '.gif', '.tiff', '.tif', '.bmp'])])

    if selected_file:
        img = Image.open(selected_file.name, mode='r')
        img = img.resize((100, 100), Image.ANTIALIAS)
        photoImg = ImageTk.PhotoImage(img)
        imported_image_label.configure(height=100, width=100)
        imported_image_label.configure(image=photoImg)
        imported_image_label.image = photoImg

        global current_image_file_for_prediction
        current_image_file_for_prediction = selected_file.name

        prediction_result_textbox.config(state=NORMAL)
        prediction_result_textbox.delete(1.0, END)
        prediction_result_textbox.config(state=DISABLED)


########################################################################################################################
#   run_prediction
########################################################################################################################
def run_prediction():
    # clear previous prediction results
    prediction_result_textbox.config(state=NORMAL)
    prediction_result_textbox.delete(1.0, END)

    if model is None:
        messagebox.showinfo("Prediction", "Model is missing.\nPlease load one.")
        return

    if current_image_file_for_prediction:
        # Import selected image as image to dislay in application
        image = tf.keras.preprocessing.image.load_img(current_image_file_for_prediction, target_size=image_size)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image_as_array = np.array([image])

        prediction = model.predict(image_as_array)

        prediction_result_list = []
        for i in range(0, prediction.size):
            if prediction[0, i] > 0.0001:
                prediction_result_list.insert(0, [prediction[0, i], i])
        if len(prediction_result_list) > 0:
            prediction_result_list.sort(reverse=True)
            #    print(prediction_result_list[0][0], prediction_result_list[0][1])
            for p in range(0, min(4, len(prediction_result_list))):
                predicted_perc_f = 100 * prediction_result_list[p][0]
                predicted_perc = f'{"%.2f" % predicted_perc_f}% for '
                predicted_name = labels_list[prediction_result_list[p][1]]
                prediction_result_textbox.insert(INSERT, predicted_perc + predicted_name + '\n')

        prediction_result_textbox.config(state=DISABLED)


########################################################################################################################
# main function
########################################################################################################################
# def main():

# create root window
root = Tk()
root.title('Mini Project 1 - Store scale')
root.geometry('{}x{}'.format(919, 378))
root.resizable(0, 0)

# create all of the main containers
top_frame = Frame(root, bg='black', width=600, height=5, pady=0)
center_frame = Frame(root, width=600, height=300, padx=0, pady=0)
row_frame = Frame(root, width=600, height=5, bd=5, bg='black')
botom_frame = Frame(root, bg='black', width=600, height=50, pady=0)

top_frame.grid(row=0, sticky='ew')
center_frame.grid(row=1, sticky='nsew')
row_frame.grid(row=2, sticky='ew')
# botom_frame.grid(row=3, sticky='ew')

# create the center widgets - root.center_frame
column1_frame = Frame(center_frame, width=5, height=300, bd=5, bg='black')
data_frame = Frame(center_frame, width=200, height=300, bd=5)
column2_frame = Frame(center_frame, width=5, height=300, bd=5, bg='black')
training_frame = Frame(center_frame, width=200, height=300, bd=5)
column3_frame = Frame(center_frame, width=5, height=300, bd=5, bg='black')
predict_frame = Frame(center_frame, width=200, height=300, bd=5)
column4_frame = Frame(center_frame, width=5, height=300, bd=5, bg='black')

column1_frame.grid(row=0, column=0, sticky='ns')
data_frame.grid(row=0, column=1, sticky='ns')
column2_frame.grid(row=0, column=2, sticky='ns')
training_frame.grid(row=0, column=3, sticky='ns')  # 'nsew'
column3_frame.grid(row=0, column=4, sticky='ns')
predict_frame.grid(row=0, column=5, sticky='ns')
column4_frame.grid(row=0, column=6, sticky='ns')

########################################################################################################################
# data import frame
########################################################################################################################
# create the data frames - root.center_frame.data_frame
data_top_frame = Frame(data_frame, width=200, height=50)
data_center_frame = Frame(data_frame, width=200, height=400, pady=10)
data_bottom_frame = Frame(data_frame, width=200, height=50)

data_top_frame.rowconfigure(0, weight=1)

data_top_frame.grid(row=0, column=0)
data_center_frame.grid(row=1, column=0)
data_bottom_frame.grid(row=2, column=0)

# set data frame title
data_title_label = Label(data_top_frame, text='Data import', font=("Arial", 15))
data_title_label.grid()

# create data center frame widgets
# create data location label
data_location_label = Label(data_center_frame, text='Data location:')
data_location_label.grid(row=0, padx=10, sticky='w')

# create string variable for current data location label
current_data_location = StringVar()
current_data_location.set(default_data_location)

# create label for current data location
current_location_label = Label(data_center_frame, textvariable=current_data_location)
current_location_label.grid(row=1, column=0, padx=20, sticky='w', columnspan=3)

# create button to change current data location
location_change_button = ttk.Button(data_center_frame,
                                    text='Change',
                                    command=lambda: change_data_location(current_data_location))
location_change_button.grid(row=0, column=2, padx=10)

##################################################
# USUNĄC JAK ZNAJDZIE NORMALNY SPOSÓB NA ODSTĘP
##################################################
space = Label(data_center_frame, text='')
space.grid(row=2, sticky='w')

# create radio button label for import setting
data_import_setting_label = Label(data_center_frame, text='Data import setting:')
data_import_setting_label.grid(row=3, padx=10, sticky='w')

# create variable for import setting selection
selected_import_setting = IntVar()
selected_import_setting.set(1)

selected_import_radio_button_1 = Radiobutton(data_center_frame, text="Labels and all datasets",
                                             variable=selected_import_setting, value=1)
selected_import_radio_button_1.grid(row=4, padx=20, sticky='w')

selected_import_radio_button_2 = Radiobutton(data_center_frame, text="Labels only",
                                             variable=selected_import_setting, value=2)
selected_import_radio_button_2.grid(row=5, padx=20, sticky='w')

##################################################
# USUNĄC JAK ZNAJDZIE NORMALNY SPOSÓB NA ODSTĘP
##################################################
space = Label(data_center_frame, text='')
space.grid(row=6, sticky='w')

# create data import status label
data_import_status_label = Label(data_center_frame, text='Data import counters:')
data_import_status_label.grid(row=7, padx=10, sticky='w')

# create training dataset counter label
data_training_label = Label(data_center_frame, text='Training dataset:')
data_training_label.grid(row=8, padx=20, sticky='w')

# create training counter variable
# 3 global training_counter
training_counter = IntVar()
training_counter.set(0)

# create training counter label
training_counter_label = Label(data_center_frame, textvariable=training_counter)
training_counter_label.grid(row=8, column=1, padx=20, sticky='e', columnspan=2)

# create validation dataset counter label
data_validation_label = Label(data_center_frame, text='Validation dataset:')
data_validation_label.grid(row=9, padx=20, sticky='w')

# create validation counter variable
# 3 global validation_counter
validation_counter = IntVar()
validation_counter.set(0)

# create validation counter label
validation_counter_label = Label(data_center_frame, textvariable=validation_counter)
validation_counter_label.grid(row=9, column=1, padx=20, sticky='e', columnspan=2)

# create test dataset counter label
data_test_label = Label(data_center_frame, text='Test dataset:')
data_test_label.grid(row=10, padx=20, sticky='w')

# create test counter variable
# 3 global test_counter
test_counter = IntVar()
test_counter.set(0)

# create test counter label
test_counter_label = Label(data_center_frame, textvariable=test_counter)
test_counter_label.grid(row=10, column=1, padx=20, sticky='e', columnspan=2)

# create labels counter label
data_labels_label = Label(data_center_frame, text='Labels:')
data_labels_label.grid(row=11, padx=20, sticky='w')

# create labels counter variable
# 3 global labels_counter
labels_counter = IntVar()
labels_counter.set(0)

# create labels counter label
labels_counter_label = Label(data_center_frame, textvariable=labels_counter)
labels_counter_label.grid(row=11, column=1, padx=20, sticky='e', columnspan=2)

# create training, validation, test datasets and labels list variable
# global train_set
# global validation_set
# global test_set
# global labels_list
#
# train_set = None
# validation_set = None
# test_set = None
# labels_list = None

# create import button
import_button = ttk.Button(data_bottom_frame,
                           text='Import',
                           command=lambda: import_data(selected_import_setting.get(),
                                                       current_data_location.get(), 1))
import_button.grid(row=0, column=0, padx=10, pady=10)

########################################################################################################################
# model training frame
########################################################################################################################
# create the data frames - root.center_frame.data_frame
training_top_frame = Frame(training_frame, width=200, height=50)
training_center_frame = Frame(training_frame, width=200, height=400, pady=10)
training_bottom_frame = Frame(training_frame, width=200, height=50)

training_top_frame.rowconfigure(0, weight=1)

training_top_frame.grid(row=0, column=0)
training_center_frame.grid(row=1, column=0)
training_bottom_frame.grid(row=2, column=0)

# set training frame title
training_title_label = Label(training_top_frame, text='Model training', font=("Arial", 15))
training_title_label.grid()

# create training center frame widgets
# create training selection model label
model_selection_label = Label(training_center_frame, text='Select model:')
model_selection_label.grid(row=0, padx=10, sticky='w')

# create model combobox variable
selected_model_var = IntVar()
selected_model_var.set(0)

# create model selection combobox
training_model_combo = ttk.Combobox(training_center_frame)  # , textvariable=selected_model_var)
training_model_combo['values'] = ('1 - Simple', '2 - Medium', '3 - Complex', '4 - InceptionV3', '5 - NASNetLarge')
training_model_combo['state'] = 'readonly'
training_model_combo.current(selected_model_var.get())
training_model_combo.bind('<<ComboboxSelected>>', model_combobox_changed)
training_model_combo.grid(row=0, column=1, padx=10, sticky='w')

# create string variable for current data location label
selected_model_description = StringVar()
update_selected_model_label(training_model_combo.current())

# create label for description of selected model
selected_model_label = Label(training_center_frame, width=40, textvariable=selected_model_description, anchor='w')
selected_model_label.grid(row=1, column=0, padx=20, sticky='w', columnspan=2)

##################################################
# USUNĄC JAK ZNAJDZIE NORMALNY SPOSÓB NA ODSTĘP
##################################################
space = Label(training_center_frame, text='')
space.grid(row=2, sticky='w')

# create radio button label for training goal
training_goal_label = Label(training_center_frame, text='Training goal:')
training_goal_label.grid(row=3, padx=10, sticky='w')

# create variable for raining goal selection
selected_training_goal = IntVar()
selected_training_goal.set(1)

selected_training_goal_radio_button_1 = Radiobutton(training_center_frame, text='Maximize accuracy',
                                                    variable=selected_training_goal, value=1)
selected_training_goal_radio_button_1.grid(row=4, padx=20, sticky='w')

selected_training_goal_radio_button_2 = Radiobutton(training_center_frame, text='Minimize loss',
                                                    variable=selected_training_goal, value=2)
selected_training_goal_radio_button_2.grid(row=5, padx=20, sticky='w')

# create epochs label
epochs_label = Label(training_center_frame, text='Epochs:')
epochs_label.grid(row=3, column=1, padx=0, sticky='w')

# create epachs variable
epochs_scale_var = IntVar()
epochs_scale_var.set(5)

# create epochs scale widget
epochs_scale = Scale(training_center_frame, orient=HORIZONTAL, from_=1, to_=50, variable=epochs_scale_var)
epochs_scale.grid(row=4, column=1, padx=10, sticky='w', rowspan=2)

##################################################
# USUNĄC JAK ZNAJDZIE NORMALNY SPOSÓB NA ODSTĘP
##################################################
space = Label(training_center_frame, text='')
space.grid(row=6, sticky='w')

# create training result label
training_result_label = Label(training_center_frame, text='Trainig results:')
training_result_label.grid(row=7, column=0, padx=10, sticky='w')

# create training accuracy label
training_accuracy_label = Label(training_center_frame, text='Trainig accuracy:')
training_accuracy_label.grid(row=8, column=0, padx=20, sticky='w')

# create training loss label
training_loss_label = Label(training_center_frame, text='Trainig loss:')
training_loss_label.grid(row=9, column=0, padx=20, sticky='w')

# create test accuracy label
test_accuracy_label = Label(training_center_frame, text='Test accuracy:')
test_accuracy_label.grid(row=10, column=0, padx=20, sticky='w')

# create test loss label
test_loss_label = Label(training_center_frame, text='Test loss:')
test_loss_label.grid(row=11, column=0, padx=20, sticky='w')

# create training accuracy variable
# 3 global training_accuracy
training_accuracy = DoubleVar()
training_accuracy.set(0.00)

# create training accuracy label
training_accuracy_label = Label(training_center_frame, textvariable=training_accuracy)
training_accuracy_label.grid(row=8, column=1, padx=10, sticky='w', columnspan=2)

# create training loss variable
# 3 global training_loss
training_loss = DoubleVar()
training_loss.set(0.00)

# create training loss label
training_loss_label = Label(training_center_frame, textvariable=training_loss)
training_loss_label.grid(row=9, column=1, padx=10, sticky='w', columnspan=2)

# create test accuracy variable
# 3 global test_accuracy
test_accuracy = DoubleVar()
test_accuracy.set(0.00)

# create test accuracy label
test_accuracy_label = Label(training_center_frame, textvariable=test_accuracy)
test_accuracy_label.grid(row=10, column=1, padx=10, sticky='w', columnspan=2)

# create test loss variable
# 3 global test_loss
test_loss = DoubleVar()
test_loss.set(0.00)

# create test loss label
test_loss_label = Label(training_center_frame, textvariable=test_loss)
test_loss_label.grid(row=11, column=1, padx=10, sticky='w', columnspan=2)

# create training button
training_button = ttk.Button(training_bottom_frame,
                             text='Execute training',
                             command=lambda: run_training(selected_model_var.get(), current_data_location.get(),
                                                          selected_training_goal.get(),
                                                          image_shape, epochs_scale_var.get(), 1))
training_button.grid(row=0, column=0, padx=10, pady=10)

########################################################################################################################
# model predict frame
########################################################################################################################
# create the data frames - root.center_frame.data_frame
predict_top_frame = Frame(predict_frame, width=200, height=50)
predict_center_frame = Frame(predict_frame, width=200, height=400, pady=10)
predict_bottom_frame = Frame(predict_frame, width=200, height=50)

training_top_frame.rowconfigure(0, weight=1)

predict_top_frame.grid(row=0, column=0)
predict_center_frame.grid(row=1, column=0)
predict_bottom_frame.grid(row=2, column=0)

# set training frame title
prediction_title_label = Label(predict_top_frame, text='Prediction', font=("Arial", 15))
prediction_title_label.grid()

# create predict center frame widgets
# create loaded model label
loaded_model_label = Label(predict_center_frame, text='Model:')
loaded_model_label.grid(row=0, padx=10, sticky='w')

# create string variable for loaded model label
loaded_model_file_name_var = StringVar()
loaded_model_file_name_var.set('none')

# create label for loaded model file name
loaded_model_file_name_label = Label(predict_center_frame, textvariable=loaded_model_file_name_var)
loaded_model_file_name_label.grid(row=1, column=0, padx=20, columnspan=3, sticky='w')

# create button to change current model
load_model_button = ttk.Button(predict_center_frame,
                               text='Load model',
                               command=lambda: load_model_from_file(current_data_location.get()))
load_model_button.grid(row=0, column=2, padx=10, sticky='e')

##################################################
# USUNĄC JAK ZNAJDZIE NORMALNY SPOSÓB NA ODSTĘP
##################################################
space = Label(predict_center_frame, text='')
space.grid(row=2, sticky='w')

# create imported image label
imported_image_label = Label(predict_center_frame, height=10, width=10)
imported_image_label.grid(row=3, column=0, columnspan=2, padx=10, sticky='ew')

# create button to import image
Image_import_button = ttk.Button(predict_center_frame,
                                 text='Import image',
                                 command=lambda: import_image_from_file(current_data_location.get()))
Image_import_button.grid(row=3, column=2, padx=10, sticky='e')

# display default image
img = Image.open(current_image_file_for_prediction, mode='r')
img = img.resize((100, 100), Image.ANTIALIAS)
photoImg = ImageTk.PhotoImage(img)
imported_image_label.configure(height=100, width=100)
imported_image_label.configure(image=photoImg)
imported_image_label.image = photoImg

# create prediction result label
prediction_result_label = Label(predict_center_frame, text='Up to 4 best predictions:')
prediction_result_label.grid(row=4, column=0, columnspan=2, padx=10, sticky='w')

# create prediction result textbox
prediction_result_textbox = Text(predict_center_frame, height=4, width=5, padx=10)
prediction_result_textbox.grid(row=5, column=0, columnspan=3, sticky='ew')

##################################################
# USUNĄC JAK ZNAJDZIE NORMALNY SPOSÓB NA ODSTĘP
##################################################
# space = Label(predict_center_frame, text='')
# space.grid(row=6, sticky='w')

# create prediction button
prediction_button = ttk.Button(predict_bottom_frame,
                               text='Predict',
                               command=lambda: run_prediction())
prediction_button.grid(row=0, column=0, padx=10, pady=10)

# create exit buttom
# exit_button = ttk.Button(botom_frame,
#                         text='Exit',
#                         command=root.destroy)
# #exit_button.place(relx=0.5, rely=0.5, anchor='e')
# exit_button.grid(row=0, column=0, padx = 10, pady = 10, sticky='e')


# start_time = datetime.datetime.now()
# print("Start treningu: " + str(start_time))
#
# max_epochs = 50
# import_data(1, default_data_location, 0)
# print("Training: " + str(datetime.datetime.now()))
# run_training(0, default_data_location,1, image_shape, max_epochs, 0)
# print("Training: " + str(datetime.datetime.now()))
# run_training(0, default_data_location,2, image_shape, max_epochs, 0)
# print("Training: " + str(datetime.datetime.now()))
# run_training(1, default_data_location,1, image_shape, max_epochs, 0)
# print("Training: " + str(datetime.datetime.now()))
# run_training(1, default_data_location,2, image_shape, max_epochs, 0)
# print("Training: " + str(datetime.datetime.now()))
# run_training(2, default_data_location,1, image_shape, max_epochs, 0)
# print("Training: " + str(datetime.datetime.now()))
# run_training(2, default_data_location,2, image_shape, max_epochs, 0)


# stop_time = datetime.datetime.now()
# print("Koniec treningu: " + str(stop_time))
# print(stop_time - start_time)

root.mainloop()

# main()
