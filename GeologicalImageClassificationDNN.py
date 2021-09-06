# *******************************************************************************************************************************************
# * @License Starts
# *
# * Copyright © 2015 - present. MongoExpUser
# *
# *  License: MIT - See: https://opensource.org/licenses/MIT
# *
# * @License Ends
# *
# *******************************************************************************************************************************************
#
#  ...GeologicalImageClassificationDNN.py  implements:
#
#  GeologicalImageClassificationDNN() class for: Standard Feed-forward Deep Neural Network (Standard-FFNN) images classification.
#
#  Data Source:
#  ===========
#  Dataset for images classification is AWS' Geological Similarity dataset
#  Source of datasets: "http://aws-proserve-data-science.s3.amazonaws.com/geological_similarity.zip"
#  It was extracted into the current working directory (CWD)
#
#  Objectives for rock/reservoir images "Classification" with Standard-FFNN Model
#  ==============================================================================
#  1) Given a set of labels/categories/classes (output) for images (input)
#  2) Train the output and input (images) for classification.
#  3) For images classification, input data include: known images datasets (converted to numerical datasets).
#  4) For fitted/trained datasets, obtain a set of hyper-parameters for the DNN model (Standard-FFNN)
#  5) Evaluate and save model.
#  6) Based on saved model, then predict classifications for unseen image datasets.
#
#  Applications:
#  =============
#  Why classification: classification helps to map rock/reservoir image for type, nature, quality, fluid content, changes, disturbance, etc.
#  which can of practical use in a variety of industrial and mineral extractive applications, including:
#    i)   optimal well placement, hydraulic fracture design and production optimization in oil and gas development
#    ii)  geo-hazard identification in site locations
#    iii) level or degree of minerals site poaching
#    iv)  mapping surface rock disturbance in surface mining or other geological-related operations
#
#
# Instruction on Running the Module
# =================================
# i) Python version: 3.7.4
#
# ii) OS : Ubuntu 18.04.4 LTS or MacOS Catalina
#
# iii) Install the required packages as: sudo python3.7 -m pip install numpy matplotlib pandas scikit-image tensorflow
#      or install with Conda if working in Conda/Anaconda environment.
#
# iv) To test the module:
# 1) Download the dataset from: "http://aws-proserve-data-science.s3.amazonaws.com/geological_similarity.zip".
# 2) Unzip the downloaded file and copy the "6 folders" containing images for the labels into the current working directory (CWD).
# 3) Save this file (GeologicalImageClassificationDNN.py) in the CWD.
# 4) At the prompt, simply run: python GeologicalImageClassificationDNN.py
# *******************************************************************************************************************************************
# *******************************************************************************************************************************************

import numpy as np
from os import listdir
import tensorflow as tf
from pprint import pprint
from time import time, sleep
from pandas import DataFrame
from json import dumps, loads
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt_results
import matplotlib.pyplot as plt_compare
from os.path import dirname, isfile, join
from random import random, randint, randrange
from skimage.io import imread as skimage_imread
from skimage.color import rgb2gray as skimage_rgb2gray
from tensorflow.keras import backend, optimizers, Sequential
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

class GeologicalImageClassificationDNN():
  """
    A class for geological images classifictaion with Standard-FFNN
  """

  def __init__(self):
    print("")
    print("Initiating Geological Imaging Classification Engine - Using DNN models.")
    print("-----------------------------------------------------------------------")
    print("Using TensorFlow version", tf.__version__, "on this system. ")
    print("Using Keras version", tf.keras.__version__, "on this system.")
    print("-----------------------------------------------------------------------")
    print("")
  # End  __init__() method

  def read_and_display_image(self, input_image):
    read_image = skimage_imread(input_image)
    plt_results.figure(figsize = (25,25))
    plt_results.imshow(read_image)
    plt_results.show()
  # read_and_display_image()

  def load_and_understand_geological_similarity_dataset(self, validation_split=None):
    data_option = "geological_images"
    zipped_images_file_name = "geological_similarity.zip"
    labels = ["andesite", "rhyolite", "gneiss", "marble","schist", "quartzite"]
    print("The dataset is geological images dataset.")

    # 1. load data
    # ============
    # the dataset (image dataset) is downloaded from: "http://aws-proserve-data-science.s3.amazonaws.com/geological_similarity.zip"
    # it was been extracted into the current working directory (CWD) and are located in 6 folders within the CWD,
    # the names of the folder corrresponds to  the images label/class/category names, which are:
    # 0 - andesite  - an igneous rock
    # 1 - rhyolite  - an igneous rock
    # 2 - gneiss    - a metamorphic rock
    # 3 - marble    - a metamorphic rock
    # 4 - schist    - a metamorphic rock
    # 5 - quartzite - a metamorphic rock

    # five folders  contain 5000 images and
    # one folder (marble folder/label) contains 4998 images

    # a) read image filenames, image labels/classes/categories and image data_sets
    # declare list variables to hold for all image data_sets, label_names and data_set_filenames
    data_sets = []
    label_names = []
    data_set_filenames = []

    # declare list variables to hold  training and test image data_sets
    # images converted to grayscale
    train_data_sets = []
    test_data_sets = []
    train_labels = []
    test_labels =[]
    # original rgb scale images (to be used for viewing)
    train_data_sets_original = []
    test_data_sets_original = []

    # define list variables to hold each label image datasets
    andesite_label = []
    rhyolite_label = []
    gneiss_label = []
    marble_label = []
    schist_label = []
    quartzite_label = []

    #variable for checking, when in development
    in_development = False

    # b) split image dataset into train and test
    for label in labels:
      #note: labels are also names of sub-folders in the CWD
      file_folders = listdir(label)
      number_of_image_dataset_to_load = len([name for name in file_folders if isfile(join(label, name))])
      for index, filename in enumerate(file_folders):
        # define each image: original images are in RGB format, with shape of 28x28x3
        original_geo_image = skimage_imread("{}{}{}".format(label, "/", filename))
        # convert images to gray scale to filter out noise and to reduce size & training run time
        # the shape is also converted to 28X28 from 28*28*3
        geo_image = skimage_rgb2gray(original_geo_image)
        if in_development:
            #check basic properties of the images, when is still in development phase
            #set "in_development" to False or None when in development phase
            print("Checking Properties of Original Images")
            print("Shape: ", original_geo_image.shape)
            print("Height: ", original_geo_image.shape[0])
            print("Width: ", original_geo_image.shape[1])
            print("Dimension: ", original_geo_image.ndim)
        # load n-number of images to use in analysis
        # where n-number = number_of_image_dataset_to_load, n <= total number of files in the folder
        if index < number_of_image_dataset_to_load:
          if label == "andesite":
            data_sets.append(geo_image)
            encoded_label = self.encode_labels(label)
            label_names.append(encoded_label)
            andesite_label.append(encoded_label)
            data_set_filenames.append(filename)
            # split image datasets into train and test datasets in ratio of validation_split
            if index < round(number_of_image_dataset_to_load*validation_split):
                train_data_sets.append(geo_image)
                train_labels.append(encoded_label)
                train_data_sets_original.append(original_geo_image)
            else:
                test_data_sets.append(geo_image)
                test_labels.append(encoded_label)
                test_data_sets_original.append(original_geo_image)
          elif label == "rhyolite":
            data_sets.append(geo_image)
            encoded_label = self.encode_labels(label)
            label_names.append(encoded_label)
            rhyolite_label.append(encoded_label)
            data_set_filenames.append(filename)
            # split image datasets into train and test datasets in ratio of validation_split
            if index < round(number_of_image_dataset_to_load*validation_split ):
                train_data_sets.append(geo_image)
                train_labels.append(encoded_label)
                train_data_sets_original.append(original_geo_image)
            else:
                test_data_sets.append(geo_image)
                test_labels.append(encoded_label)
                test_data_sets_original.append(original_geo_image)
          elif label == "gneiss":
            data_sets.append(geo_image)
            encoded_label = self.encode_labels(label)
            label_names.append(encoded_label)
            gneiss_label.append(encoded_label)
            data_set_filenames.append(filename)
            # split image datasets into train and test datasets in ratio of validation_split
            if index < round(number_of_image_dataset_to_load*validation_split ):
                train_data_sets.append(geo_image)
                train_labels.append(encoded_label)
                train_data_sets_original.append(original_geo_image)
            else:
                test_data_sets.append(geo_image)
                test_labels.append(encoded_label)
                test_data_sets_original.append(original_geo_image)
          elif label == "marble":
            data_sets.append(geo_image)
            encoded_label = self.encode_labels(label)
            label_names.append(encoded_label)
            marble_label.append(encoded_label)
            data_set_filenames.append(filename)
            # split image datasets into train and test datasets in ratio of validation_split
            if index < round(number_of_image_dataset_to_load*validation_split ):
                train_data_sets.append(geo_image)
                train_labels.append(encoded_label)
                train_data_sets_original.append(original_geo_image)
            else:
                test_data_sets.append(geo_image)
                test_labels.append(encoded_label)
                test_data_sets_original.append(original_geo_image)
          elif label == "schist":
            data_sets.append(geo_image)
            encoded_label = self.encode_labels(label)
            label_names.append(encoded_label)
            schist_label.append(encoded_label)
            data_set_filenames.append(filename)
            # split image datasets into train and test datasets in ratio of validation_split
            if index < round(number_of_image_dataset_to_load*validation_split):
                train_data_sets.append(geo_image)
                train_labels.append(encoded_label)
                train_data_sets_original.append(original_geo_image)
            else:
                test_data_sets.append(geo_image)
                test_labels.append(encoded_label)
                test_data_sets_original.append(original_geo_image)
          elif label == "quartzite":
            data_sets.append(geo_image)
            encoded_label = self.encode_labels(label)
            label_names.append(encoded_label)
            quartzite_label.append(encoded_label)
            data_set_filenames.append(filename)
            # split image datasets into train and test datasets in ratio of validation_split
            if index < round(number_of_image_dataset_to_load*validation_split):
                train_data_sets.append(geo_image)
                train_labels.append(encoded_label)
                train_data_sets_original.append(original_geo_image)
            else:
                test_data_sets.append(geo_image)
                test_labels.append(encoded_label)
                test_data_sets_original.append(original_geo_image)

    # 2. understand the data
    # ========================
    # a) show number of total datasets
    #    note: if all images are loaded (i.e. number_of_image_dataset_to_load = total number of images in the label folders), then:
    #    5 folders should contain 5000 images and 1 folder (marble label) should contain 4998 images, when printed as shown below
    print("Number of images labeled as andesite - encoded 0: ", len(andesite_label))
    print("Number of images labeled as gneiss - encoded 1: ", len(gneiss_label))
    print("Number of images labeled as rhyolite - encoded 2: ", len(rhyolite_label))
    print("Number of images labeled as marble: - encoded 3", len(marble_label))
    print("Number of images labeled as schist: - encoded 4", len(schist_label))
    print("Number of images labeled as quartzite - encoded 5: ", len(quartzite_label))

    # b) define and show pandas' data frame for images filenames, classes/categories/labels and datasets
    dict_of_filename_labels_data = {'image_filenames': data_set_filenames,
                                    'image_labels': label_names,
                                    "image_data": data_sets
                                   }
    data_frame = DataFrame(dict_of_filename_labels_data)
    pprint(data_frame)

    # c) show numbers of train and test datasets (train is for training and test is for validation)
    #    the numbers should reflect tha value specified as validation_split
    print("Number of train images: ", len(train_data_sets))
    print("Number of test images:  ", len(test_data_sets))
    print("Number of train labels: ", len(train_labels))
    print("Number of test labels:  ", len(test_labels))

    # d) define and show pandas' data frame for train and test images datasets
    dict_of_train_and_test_datasets_and_labels = {'train_labels': train_labels,
                                                  'train_images': train_data_sets,
                                                  'test_labels': test_labels,
                                                  'test_images': test_data_sets,
                                                  'train_data_sets_original': train_data_sets_original,
                                                  'test_data_sets_original': test_data_sets_original
                                                  }

    # e) view data frame of the train and test datasets
    train_data_frame = DataFrame({'train_labels': train_labels, 'train_images': train_data_sets})
    test_data_frame = DataFrame({'test_labels': test_labels, 'test_images': test_data_sets})
    pprint(train_data_frame)
    pprint(test_data_frame)
    return dict_of_train_and_test_datasets_and_labels
  # End load_geological_similarity_dataset() method

  def encode_labels(self, label):
    if label == "andesite":
        return 0
    elif label == "rhyolite":
        return 1
    elif label == "gneiss":
        return 2
    elif label == "marble":
        return 3
    elif label == "schist":
        return 4
    elif label == "quartzite":
        return 5
  # End encode_labels() method

  def save_combined_images(self, train_images=None, train_labels=None, label_names=None, image_filename=None, start_from=None, label_name=None):
    plt.figure(figsize=(10,10)) # figure size is 10x10 inches
    for index in range(144):   # each data is 12*12 (144 images)
        plt.subplot(12,12, index+1, xticks=[], yticks=[])
        plt.grid(True)
        plt.imshow(train_images[index+start_from], interpolation="gaussian")
    plt.savefig("{}{}{}{}".format(image_filename, "-samples_of_", label_name, "_images.png"), dpi=300) # save figure in the CWD
  #End save_combined_images() method

  def save_some_train_images_and_train_labels_for_viewing(self, train_images=None, train_labels=None, label_names=None, image_filename=None, validation_split=None):
    # save some images, for each labels on separate templates, for viewing
    str_0 = round(0*validation_split)
    str_1 = round(str_0 + 5000*validation_split)
    str_2 = round(str_1 + 5000*validation_split)
    str_3 = round(str_2 + 4998*validation_split)
    str_4 = round(str_3 + 5000*validation_split)
    str_5 = round(str_4 + 5000*validation_split)
    # 0 - andesite - an igneous rock
    self.save_combined_images(train_images=train_images, train_labels=train_labels, label_names=label_names, image_filename=image_filename, start_from=str_0, label_name="andesite")
    # 1 - rhyolite - an igneous rock
    self.save_combined_images(train_images=train_images, train_labels=train_labels, label_names=label_names, image_filename=image_filename, start_from=str_1, label_name="rhyolite")
    # 2 - gneiss   - a metaphoric rock
    self.save_combined_images(train_images=train_images, train_labels=train_labels, label_names=label_names, image_filename=image_filename, start_from=str_2, label_name="gneiss")
    # 3 - marble   - a metaphoric rock
    self.save_combined_images(train_images=train_images, train_labels=train_labels, label_names=label_names, image_filename=image_filename, start_from=str_3, label_name="marble")
    #4 - schist   - a metaphoric rock
    self.save_combined_images(train_images=train_images, train_labels=train_labels, label_names=label_names, image_filename=image_filename, start_from=str_4, label_name="schist")
    # 5 - quartzite - a metaphoric rock
    self.save_combined_images(train_images=train_images, train_labels=train_labels, label_names=label_names, image_filename=image_filename, start_from=str_5, label_name="quartzite")
  # End save_some_train_images_and_train_labels_for_viewing() method

  def predict_and_view_with_new_or_existing_saved_model(self, option=None, model=None, input_images_to_predict=None, input_labels_expected_prediction=None, label_names=None,
                                                        image_filename=None, number_of_labels=None, original_train_images=None, original_test_images=None, validation_split=None):

    # predict labels unseen images" with coded color (green=correct, red=incorrect)
    # a. define correct, incorrect and neutral colors
    correct_color = 'green'
    incorrect_color = 'red'
    neutral_color = 'gray'

    #.b define predictions and expectations
    predictions = model.predict(input_images_to_predict)
    expectations = input_labels_expected_prediction

    # c. image plot
    def image(index=None, predictions_array=None, actual_label=None, img=None, label_name=None):
      predictions_array, actual_label, img = predictions_array, actual_label[index], img[index]
      plt.grid(False)
      plt.xticks([])
      plt.yticks([])
      label_prediction = np.argmax(predictions_array)
      answer = ""
      if option == "ffnn":
        plt.imshow(img, interpolation="gaussian")
        confirm = (label_prediction == actual_label)
      if confirm:
        color = correct_color
        answer = "Correct"
      else:
        color = incorrect_color
        answer = "Wrong"
      confidence = round(np.max(predictions_array)*100, 1)
      plt.xlabel("{}{}{}%".format(label_names[label_prediction], " @ ", confidence, " CL"), color=color)
      print("{}{}{}{}{:0.1f}%{}{}".format(label_name, " is predicted as ", label_names[label_prediction], " @ ", 100*np.max(predictions_array), " confidence level - ", answer))

    # d. value plot
    def value(index=None, predictions_list=None, actual_label=None):
      predictions_list, actual_label = predictions_list, actual_label[index]
      plt.grid(True)
      plt.xticks(range(10))
      plt.yticks(range(10))
      plt.ylabel("Confidence")
      plt.ylim([0, 1])
      plt.xlim([0, 5])
      label_prediction = np.argmax(predictions_list)
      if option == "ffnn":
        thisplot = plt.bar(range(6), predictions_list, color=neutral_color)
        thisplot[label_prediction].set_color(incorrect_color)
        thisplot[actual_label].set_color(correct_color)

    # e. final/combined plot
    def final_plot(number_of_rows=None, number_of_columns=None, start_from=None, label_name=None):
      number_of_images = number_of_rows * number_of_columns
      plt.figure(figsize=(2 * 2 * number_of_columns, 2 * number_of_rows))
      for index in range(number_of_images):
        plt.subplot(number_of_rows, 2*number_of_columns, ((2 * index) + 1))
        image(index+start_from, predictions[index + start_from], input_labels_expected_prediction, original_test_images, label_name)
        plt.subplot(number_of_rows, 2 * number_of_columns, ((2 * index) + 2))
        value(index+start_from, predictions[index+start_from], input_labels_expected_prediction)
      plt.grid(True)
      plt.tight_layout()
      plt.savefig("{}{}{}{}".format(image_filename, "-prediction_vs_expectation_of_", label_name, "_images.png"), dpi=300) # save figure in the CWD

    # f. invoke final plot
    _number_of_rows = 5
    _number_of_columns = 2
    validation_split = (1-validation_split) # for test
    str_0 = round(0*validation_split)
    str_1 = round(str_0 + 5000*validation_split)
    str_2 = round(str_1 + 5000*validation_split)
    str_3 = round(str_2 + 4998*validation_split)
    str_4 = round(str_3 + 5000*validation_split)
    str_5 = round(str_4 + 5000*validation_split)
    final_plot(number_of_rows=_number_of_rows, number_of_columns=_number_of_columns, start_from=str_0, label_name="andesite")
    final_plot(number_of_rows=_number_of_rows, number_of_columns=_number_of_columns, start_from=str_1, label_name="rhyolite")
    final_plot(number_of_rows=_number_of_rows, number_of_columns=_number_of_columns, start_from=str_2, label_name="gneiss")
    final_plot(number_of_rows=_number_of_rows, number_of_columns=_number_of_columns, start_from=str_3, label_name="marble")
    final_plot(number_of_rows=_number_of_rows, number_of_columns=_number_of_columns, start_from=str_4, label_name="schist")
    final_plot(number_of_rows=_number_of_rows, number_of_columns=_number_of_columns, start_from=str_5, label_name="quartzite")
  # End predict_and_view_with_new_or_existing_saved_model() method

  def display_results_if_predicted_results_exist(self):
    print("Now loading these saved images (.png files) from the CWD for viewing ...")
    #note: images are located in the CWD
    image_exist = "ffnn_image-prediction_vs_expectation_of_andesite_images.png"
    if image_exist:
        self.read_and_display_image("ffnn_image-prediction_vs_expectation_of_andesite_images.png")
    # rhyolite
    image_exist = "ffnn_image-prediction_vs_expectation_of_rhyolite_images.png"
    if image_exist:
        self.read_and_display_image("ffnn_image-prediction_vs_expectation_of_rhyolite_images.png")
    # gneiss
    image_exist = "ffnn_image-prediction_vs_expectation_of_gneiss_images.png"
    if image_exist:
        self.read_and_display_image("ffnn_image-prediction_vs_expectation_of_gneiss_images.png")
    # marble
    image_exist = "ffnn_image-prediction_vs_expectation_of_marble_images.png"
    if image_exist:
        self.read_and_display_image("ffnn_image-prediction_vs_expectation_of_marble_images.png")
    # schist
    image_exist = "ffnn_image-prediction_vs_expectation_of_schist_images.png"
    if image_exist:
        self.read_and_display_image("ffnn_image-prediction_vs_expectation_of_schist_images.png")
    # quartzite
    image_exist = "ffnn_image-prediction_vs_expectation_of_quartzite_images.png"
    if image_exist:
        self.read_and_display_image("ffnn_image-prediction_vs_expectation_of_quartzite_images.png")
  #End display_results_if_predicted_results_exist() method

  def evaluate_with_existing_saved_model(self, saved_model_name=None, optimizer=None, loss=None, test_images=None, test_labels=None, verbose=None):
    # 1. load existing saved model
    filename = open(saved_model_name + ".json", 'r')
    loaded_saved_model = model_from_json(filename.read())
    filename.close()
    loaded_saved_model.load_weights(saved_model_name + ".h5")
    print("Saved model is successfully loaded from the disk in the CWD")
    # 2.  evaluate loaded saved model
    loaded_saved_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print("Now evaluating loaded saved model on test data...")
    score = loaded_saved_model.evaluate(test_images, test_labels, verbose=verbose)
    #3. print loss and accuracy of evaluation
    print(loaded_saved_model.metrics_names[0], "{:0.4f}".format(score[0]))
    print(loaded_saved_model.metrics_names[1], "{:0.4f}%".format(score[1]*100))
    return loaded_saved_model
  # End evaluate_with_existing_saved_model() method

  def save_model_in_current_working_directory(self, saved_model_name=None, model=None):
    with open(saved_model_name + ".json", "w") as filename:
      filename.write(model.to_json())
    model.save_weights(saved_model_name + ".h5") # serialize weights to HDF5 (binary format)
    print("Model is successfully saved to disk in the CWD")
  # End save_model_in_current_working_directory() method

  def initialize_dataset_for_ffnn_images_classification(self, classify=None, data_option=None, validation_split=None):
    if classify:
      if data_option == "geological":
        print("----------------------------------------------------------------------")
        print("Using AWS Geological dataset to test FFNN image classification model. ")
        print("----------------------------------------------------------------------")
        print("Loading dataset ........... ")
        # geological similarity case study dataset
        geo_similarity_data_set = self.load_and_understand_geological_similarity_dataset(validation_split=validation_split)
        data_set = [ geo_similarity_data_set["train_images"], geo_similarity_data_set["train_labels"],
                     geo_similarity_data_set["test_images"], geo_similarity_data_set["test_labels"],
                     geo_similarity_data_set["train_data_sets_original"], geo_similarity_data_set["test_data_sets_original"]
                   ]
        label_names = ['andesite', 'rhyolite', 'gneiss', 'marble', 'schist', 'quartzite']
        print("The dataset is geological images dataset")

      # return dataset, hyper-parameters and other inputs
      return  { "data": data_set,
                "label_names": label_names,
                "shape_x": 28,
                "shape_y": 28,
                "input_layer_activation": 'relu',
                "hidden_layers_activation": 'relu',
                "output_layer_activation": 'softmax',
                "unit_per_input_layer": 160,
                "unit_per_hidden_layer": 40,
                "unit_per_output_layer": 6,
                "dropout": 0.01,
                "number_of_hidden_layers": 5,
                "optimizer": "Adam",
                "loss": 'sparse_categorical_crossentropy',
                "verbose": 1,
                "epochs": 300,
                "batch_size": 32,
                "existing_saved_model": False,
                "save_model": True,
                "saved_model_name": 'classification_model_ffnn',
                "image_filename": 'ffnn_image',
                "make_predictions": True,
                "input_images_to_predict": None,
                "input_labels_expected_prediction": None
              }
  # End initialize_dataset_for_ffnn_images_classification() method

  def ffnn_images_classification(self, ffnn_options=None, validation_split=None):
    """Standard Feed-forward Deep Neural Network (Standard-FFNN) for "images"  classification.
       The abstraction in this method is simplified and similar to sklearn's MLPClassifier(args),
       such that calling the method is reduced to just 1 line of statement with the properly defined
       input "image data", hyper-parameters and other inputs as arguments

       This methods implements the actual classification based on Standard-FFNN

       This methods combines all the other methods in the class to:
       1. load and undertand dataset and split them into training and test datatests with correct labeling
       2. preprocess/prepared datasets
       3. visualize and save some of the training images
       4. create, compile and train model
       5. evaluate model and print results (using existing saved or newly created trained)
       6. make predictions
       7. save predicted results (images) for visualization
    """

    if ffnn_options:
      # load/read dataset
      data_set = ffnn_options["data"]
      # a. define/pass in train's and test's images & labels
      train_images = data_set[0]
      train_labels = data_set[1]
      test_images = data_set[2]
      test_labels = data_set[3]
      # b. get copies of original images and labels: the original will be used for view/plot/display
      original_train_images =  data_set[4]
      original_train_labels = train_labels
      original_test_images = data_set[5]
      original_test_labels = test_labels

      # image data preparation/pre-processing
      # a. note: rgb scale to gray scale
      # i)  loaded images - item (a) above, have already been converted to gray scale, to filter out noise and to reduce size & training run time
      # ii) the shape has also been converted  28X28 from 28*28*3
      # b. convert image data to numpy arrays
      train_images = np.array(data_set[0])
      train_labels = np.array(data_set[1])
      test_images = np.array(data_set[2])
      test_labels = np.array(data_set[3])

      # c. re-scale pixel intensity to between 0 and 1
      train_images = train_images/255.0
      test_images =  test_images/255.0

      # define hyper-parameters and other inputs and pass theirvalues from argument dictionary (ffnn_options)
      label_names = ffnn_options.get("label_names")
      shape_x = ffnn_options.get("shape_x")
      shape_y = ffnn_options.get("shape_y")
      input_layer_activation =  ffnn_options.get("input_layer_activation")
      hidden_layers_activation =  ffnn_options.get("hidden_layers_activation")
      output_layer_activation =  ffnn_options.get("output_layer_activation")
      unit_per_input_layer = ffnn_options.get("unit_per_input_layer")
      unit_per_hidden_layer = ffnn_options.get("unit_per_hidden_layer")
      unit_per_output_layer = ffnn_options.get("unit_per_output_layer")
      dropout = ffnn_options.get("dropout")
      number_of_hidden_layers = ffnn_options.get("number_of_hidden_layers")
      optimizer = ffnn_options.get("optimizer")
      loss = ffnn_options.get("loss")
      verbose = ffnn_options.get("verbose")
      epochs = ffnn_options.get("epochs")
      batch_size = ffnn_options.get("batch_size")
      existing_saved_model = ffnn_options.get("existing_saved_model")
      save_model = ffnn_options.get("save_model")
      saved_model_name =  ffnn_options.get("saved_model_name")
      image_filename = ffnn_options.get("image_filename")
      make_predictions = ffnn_options.get("make_predictions")

      # define images and labels to predict, if prediction is desired
      if make_predictions:
        if ffnn_options.get("input_images_to_predict") == None:
          input_images_to_predict = test_images
        if ffnn_options.get("input_labels_expected_prediction") == None:
          input_labels_expected_prediction = test_labels

      #save some images with class names (to view and and verify classifications)
      self.save_some_train_images_and_train_labels_for_viewing(original_train_images, original_train_labels, label_names, image_filename, validation_split)

      # create, fit/train, evaluate and save new model
      if not existing_saved_model:
        # compose/create model with loop to generalise number of hidden layers
        model = Sequential()
        # reformat data: transforms  format of images from 2d-array (of shape_x by shape_y pixels), to a 1d-array of shape_x * shape_y pixels.
        model.add(Flatten(input_shape=(shape_x, shape_y)))
        # add dense and dropout layers for input layer
        model.add(Dense(units=unit_per_input_layer, activation=input_layer_activation))
        model.add(Dropout(dropout, noise_shape=None, seed=None))
        # add dense and dropout layers for hidden layers
        for layer_index in range(number_of_hidden_layers):
          model.add(Dense(units=unit_per_hidden_layer, activation=hidden_layers_activation))
          model.add(Dropout(dropout, noise_shape=None, seed=None))
        # add dense layers for output layer
        model.add(Dense(units=unit_per_output_layer, activation=output_layer_activation))

        # compile the model
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        # print model's topology/summary/structure
        print("Model Topology.")
        print("===============")
        model.summary()

        # fit/train the model
        history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,  verbose=verbose)

        # evaluate the model with the "test" and train datasets at every epoch
        score_test = model.evaluate(test_images, test_labels, verbose=verbose)
        score_train = model.evaluate(train_images, train_labels, verbose=verbose)

        #print loss and accuracy of evaluation (for both test and train dataset)
        print("Unseen test data point", model.metrics_names[0], "{:0.4f}%".format(score_test[0]))
        print("Unseen test data point", model.metrics_names[1], "{:0.4f}%".format(score_test[1]*100))
        print("Seen train data point", model.metrics_names[0], "{:0.4f}%".format(score_train[0]))
        print("Seen train data point", model.metrics_names[1], "{:0.4f}%".format(score_train[1]*100))

        # save model in the current working directory (CWD), if desired
        if save_model:
          self.save_model_in_current_working_directory(saved_model_name, model)

      # evaluate with existing saved model
      if existing_saved_model:
        model = self.evaluate_with_existing_saved_model(saved_model_name, optimizer, loss, test_images, test_labels, verbose)

      # if desired, finally make prediction with test_images or other images and plot predictions
      if make_predictions:
        print("Now making predictions based on unseen images, from test data or other data ...")
        number_of_labels=len(label_names)
        option="ffnn"
        self.predict_and_view_with_new_or_existing_saved_model(option, model, input_images_to_predict, input_labels_expected_prediction,
                                                              label_names, image_filename, number_of_labels, original_train_images,
                                                              original_test_images, validation_split)
        print("Predictions completed.")
        print("Check latest saved images (.png files) in the CWD for predicted results in graphical format.")
  # End ffnn_images_classification() method

  def duration_separator(self):
    print("<==============================================================================>")
  # End duration_separator method()
# End GeologicalImageClassificationDNN() class

def main(option=None):
    start_time= time()
    gic_dnn = GeologicalImageClassificationDNN()
    validation_split = 0.9

    # run the FFNN image similarity application
    if option == "ffnn":
        ffnn_options_dataset = gic_dnn.initialize_dataset_for_ffnn_images_classification(classify=True, data_option="geological", validation_split=validation_split)
        gic_dnn.ffnn_images_classification(ffnn_options=ffnn_options_dataset, validation_split=validation_split)

    # print duration of run
    duration = (time() - start_time)/60
    gic_dnn .duration_separator()
    option_to_upper = option.upper()
    print("{}{}{}".format(option_to_upper, " classification, evaluation and prediction run time (minutes):", '{0:.4f}'.format(duration)))
    gic_dnn .duration_separator()
#End main() method

# invoke app
if __name__ in ('__main__', 'app'):
    option = "ffnn"
    main(option=option)
