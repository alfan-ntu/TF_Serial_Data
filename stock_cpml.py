"""
    Brief Description:
        1. Test program exercising stock_crawler class in stock_quote_crawler.py

    ToDo's:
        1. Include command parser utilities to make this more formal, low priority

    Date: 2023/8/2
    Ver.: 0.2a
    Author: maoyi.fan@gmail.com
    Reference:

    Revision History:
        v. 0.2a: newly created

"""
import sys
import tensorflow as tf
from stock_price_crawler import stock_quote_crawler
from model_trainer import stk_price_modeler
from utilities import plot_functions, argv_parser


#
# main function of the stock information crawler part
#
def main_crawler(argv):
    # Exercise stock crawler functionalities
    sc = stock_quote_crawler.stock_crawler()
    # sc.symbol = "2330"
    sc.data_storage_directory=".\\data_storage\\"
    sc.crawl_stock_thru_TWSE(symbol="2330", start_date='2011-01-01')
    print(f'target symbol ID: {sc.symbol}')
    # sc.display_dataframe_info()
    sc.dataframe.set_index("日期", inplace=True)

    # print(sc.field_dictionary.get('成交金額'))
    sc.plot_stock_data(('收盤價', '成交金額'))

    # Exercise stock price prediction model related functionalities
    stp = stk_price_modeler.dnn_modeler()
    print(stp)

    plot_functions.hello()


#
# main function of the neural network modeler part
#
def main_modeler(argv):
    # Create a new class instance of dnn_modeler
    mdlr = stk_price_modeler.dnn_modeler()
    # Extract the serial data of interest from the dataset source
    time_list, x_list = mdlr.time_serial_data_prep('.\\data_storage\\2330_stock_data_2020-01-01.csv')
    if time_list is None:
        print('Specified data source not found!')
        sys.exit(2)
    else: # plot the data series
        # plot_functions.plot_series(mdlr.time_list, mdlr.series)
        pass
    #
    # Split the dataset to train and validation data subsets
    #
    # mdlr.split_ratio = 0.9
    if mdlr.split_dataset():
        print(f'Length of training data: {len(mdlr.time_train)}')
        print(f'Length of validation data: {len(mdlr.time_valid)}')

    #
    # Windowing the training data
    #
    mdlr.window_size = 20
    mdlr.batch_size = 12
    mdlr.shuffle_buffer_size = 1000
    series = mdlr.x_train
    # print(f'Window size: {mdlr.window_size}')
    # print(f'Batch size: {mdlr.batch_size}')
    # print(f'Shuffle buffer size: {mdlr.shuffle_buffer_size}')

    train_set = mdlr.windowed_dataset(series)
    # print(f'Shape of windowed_series: {windowed_series.element_spec}')
    model = mdlr.model_configuration()
    # model.summary()
    mdlr.initial_weights = model.get_weights()
    # train the model at a specified learning rate, which was obtained from previous
    # trial training using LearningRateScheduler callbacks
    mdlr.learning_rate = 1e-7
    optimizer = tf.keras.optimizers.SGD(learning_rate=mdlr.learning_rate, momentum=0.9)
    # Set the training parameters
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=['mae'])
    # Added a CSV Logger callback function to store the training history every epoch
    training_history = 'history.csv'
    history_logger = tf.keras.callbacks.CSVLogger(training_history, separator=',', append=True)
    mdlr.history = model.fit(train_set, epochs=100, callbacks=[history_logger])
    # Plot the training results
    history = mdlr.history
    mae = history.history['mae']
    loss = history.history['loss']
    # Get the number of epochs
    epochs = range(len(loss))
    plot_functions.plot_series(x=epochs,
                               y=(mae, loss),
                               title='MAE and Loss',
                               xlabel='Epochs',
                               ylabel='Loss & MAE',
                               legend=['MAE', 'Loss'],
                               block=False)

    # Plot zoomed mae and loss
    zoom_split = int(epochs[-1] * 0.2)
    epochs_zoom = epochs[zoom_split:]
    mae_zoom = mae[zoom_split:]
    loss_zoom = loss[zoom_split:]
    plot_functions.plot_series(x=epochs_zoom,
                               y=(mae_zoom, loss_zoom),
                               title='Zoomed MAE and Loss',
                               xlabel='Epochs',
                               ylabel='Loss & MAE',
                               legend=['MAE', 'Loss'])

    return


if __name__ == '__main__':
    # main_crawler(sys.argv[1:])
    main_modeler(sys.argv[1:])