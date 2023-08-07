"""
    Brief Description:
        1. Based on the stock price history retrieved in 'stock_price_crawler', construct
           a DNN model
        2. Preprocess dataset before feeding it to the target DNN model
        3. Trains the target model
        4. Perform basic validation on the validation dataset
        5. Store/Load the trained model

    ToDo's:
        1.

    Date: 2023/8/2
    Ver.: 0.2a
    Author: maoyi.fan@gmail.com
    Reference:

    Revision History:
        v. 0.2a: newly created
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import sys
import math


class dnn_modeler:
    def __init__(self):
        self.data_source_file = ""
        self.data_format = "csv"
        # self.split_ratio = 0.8
        self.__split_ratio = 0.8
        self.time_list = []
        self.series = []
        self.time_train = []
        self.x_train = []
        self.time_valid = []
        self.x_valid = []
        # Parameters regarding the windowing operation
        self.__window_size = None
        self.__batch_size = None
        self.__shuffle_buffer_size = None
        self.__learning_rate = None
        # Target neural network and training results
        self.model = None
        self.initial_weights = None
        self.history = None

        return

    def __str__(self):
        obj_string = 'A class handling the life cycle to prepare/train/store/load a DNN model'
        return obj_string

    @property
    def split_ratio(self) -> float:
        return self.__split_ratio

    @split_ratio.setter
    def split_ratio(self, value):
        if value <= 0 or value >= 1:
            print(f'Split ratio set wrong: {value}')
        else:
            self.__split_ratio = value
        return

    @property
    def window_size(self) -> int:
        return self.__window_size

    @window_size.setter
    def window_size(self, value):
        if value <= 0 :
            print(f'Window size set wrong: {value}')
        else:
            self.__window_size = value
        return

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value <= 0:
            print(f'Batch size set wrong: {value}')
        else:
            self.__batch_size = value
        return

    @property
    def shuffle_buffer_size(self) -> int:
        return self.__shuffle_buffer_size

    @shuffle_buffer_size.setter
    def shuffle_buffer_size(self, value):
        if value <= 0:
            print(f'Shuffle buffer size set wrong: {value}')
        else:
            self.__shuffle_buffer_size = value
        return

    @property
    def learning_rate(self) -> float:
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        if value <= 0 or value >= 1:
            print(f'Error setting learning rate: {value}')
        else:
            self.__learning_rate = value
        return

    def time_serial_data_prep(self, data_source, data_format="csv"):
        """
        Read source file and convert data to Numpy arrays, time_list and series
        Note: This method may be very different from data source to data source!
        :param data_source:
        :param data_format:
        :return time_list, series: Numpy arrays time_list and series
        """
        if os.path.exists(data_source):
            print(f'Opening the data source file: {data_source}')
        else:
            print(f'The specified data source file: {data_source} does not exist!')
            return None, None

        time_step = []
        closing_price = []
        if data_format == "csv":
            # Open CSV file and read it row-by-row
            with open(data_source, 'r', encoding='utf-8') as csvfile:
                # Initialize a csv file reader
                reader = csv.reader(csvfile, delimiter=',')
                # Skip the first row
                next(reader)
                # Iterate the remaining rows
                no_rec = 0
                for row in reader:
                    closing_price.append(float(row[6]))
                    no_rec += 1
                    # Note: column number of time_step to be decided,
                    #       dependent on if a model accepts date-timestamp
                    time_step.append(row[1])
                # time_step = [*range(no_rec)]
        elif data_format == "json":
            pass
        #
        # Convert lists to numpy arrays so that it can be fed into a model
        # for training
        #
        self.time_list = np.array(time_step)
        self.series = np.array(closing_price)

        return self.time_list, self.series

    def split_dataset(self, split_ratio=None):
        """
        Split the source data, stored in self.series into training dataset and
        validation dataset with the split ratio specified in the split_ratio

        :param split_ratio: dataset split ratio
        :return: Ture if basic sanity check is OK, False otherwise
        """
        if split_ratio is None:
            split_ratio = self.__split_ratio
        elif split_ratio <= 0 or split_ratio >= 1:
            print(f'Setting split_ratio to an invalid value: {split_ratio}')
            return False
        else:
            split_ratio = split_ratio

        if len(self.time_list) != len(self.series):
            print(f'Different length of time_list and data series!')
            return False
        else:
            print(f'Length of time list or data series: {len(self.time_list)}')

        data_length = len(self.time_list)
        split_time = math.ceil(data_length * split_ratio)
        print(f'Split ratio: {split_ratio}, going to split the data series of length {data_length} at {split_time}')
        self.time_train = self.time_list[:split_time]
        self.x_train = self.series[:split_time]
        self.time_valid = self.time_list[split_time:]
        self.x_valid = self.series[split_time:]

        return True

    def windowed_dataset(self, series, window_size=None, batch_size=None, shuffle_buffer_size=None) -> tf.data.Dataset:
        """
        Split the input data series into slots of data series of the same window size and
        further process the windowed data to different shuffled batches
        :param series: original serial dataset for training
        :param window_size: window size for slicing the original serial dataset
        :param batch_size: batch size to group windows
        :param shuffle_buffer_size:
        :return: windowed datasets containing values, labels pair
        """
        window_size = window_size if window_size is not None else self.__window_size
        batch_size = batch_size if batch_size is not None else self.__batch_size
        shuffle_buffer_size = shuffle_buffer_size if shuffle_buffer_size is not None else self.__shuffle_buffer_size

        # Generate a TF Dataset from the series values
        dataset = tf.data.Dataset.from_tensor_slices(series)

        # Window the data and only take those with specified size
        dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)

        # Flatten the windows by putting its elements in a single batch
        dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

        # Create tuples with features and labels
        dataset = dataset.map(lambda window: (window[:-1], window[-1]))

        # Shuffle the windows
        dataset = dataset.shuffle(shuffle_buffer_size)

        # Create batches of windows
        dataset = dataset.batch(batch_size).prefetch(1)

        return dataset

    def model_configuration(self) -> tf.keras.Model:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=64,
                                   kernel_size=3,
                                   activation='relu',
                                   padding='causal',
                                   input_shape=[self.__window_size, 1]),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(30, activation="relu"),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Lambda(lambda x: x * 800)
        ])
        self.model = model
        # # print the model summary
        # model.summary()

        return self.model



