"""
    Brief Description:
        1. Class supporting crawling stock information

    ToDo's:
        1. To implement data access using yfinance library

    Date: 2023/8/2
    Ver.: 0.2a
    Author: maoyi.fan@gmail.com
    Reference:
        1. Refer to  https://aronhack.com/zh/download-stock-historical-data-with-python-and-yahoo-finance-api-zh/
           to get Taiwan Stock Exchange history using yfinance API

    Revision History:
        v. 0.2a: class style implementation
        v. 0.1c: added simple command line interface and argument parser
"""
# import package
from dateutil import rrule
import urllib.request
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np
import json
import time
import ssl
import os
import sys

class stock_crawler:
    def __init__(self, symbol=None):
        self.__symbol = symbol
        self.start_date = None
        self.end_date = None
        self.store_info_per_month = False
        self.store_df_to_csv = True
        self.dataframe = None
        self.field_dictionary = {}              # field name Chinese to English translation
        self.data_storage_directory = '.\\'     # directory for storing the retrieved historical
                                                # data
        self.plot_width = 5
        self.plot_height = 3

    @property
    def symbol(self) -> str:
        if self.__symbol is None:
            print(f'Warning: Stock symbol code is not assigned yet!')

        return self.__symbol

    @symbol.setter
    def symbol(self, value):
        self.__symbol = value

    def __check_parameters(self):
        syntax_check = True
        if ((self.__symbol is None) or \
            (self.start_date is None) or \
            (self.end_date is None) or \
            (self.__symbol is None)):
            syntax_check = False

        return syntax_check

    def __save_month_response(self, response, date):
        try:
            local_file_path = self.__symbol + '_' + date.strftime('%Y%m%d') + '.json'
            with open(local_file_path, 'wb') as local_file:
                local_file.write(response)

        except Exception as e:
            print(f'Error: {e}')

        return

    def __crawl_one_month(self, date):
        url = (
                "https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date=" +
                date.strftime('%Y%m%d') +
                "&stockNo=" +
                self.__symbol
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE"}
        ssl._create_default_https_context = ssl._create_unverified_context
        req = urllib.request.Request(url, headers=headers)
        data = json.loads(urllib.request.urlopen(req).read())
        if self.store_info_per_month:
            self.__save_month_response(urllib.request.urlopen(req).read(), date)

        return pd.DataFrame(data['data'], columns=data['fields'])

    def __convert_dataframe_format(self, df):
        """
        Convert the dataframe which is usually of string type to integer or float type
        dependent on which field of the dataframe

        :param df: dataframe received from the http server
        :return: converted dataframe
        """
        df['收盤價'] = df['收盤價'].astype(float)
        df['成交金額'] = df['成交金額'].apply(lambda x: x.replace(',', '').replace('$', '')).astype(float)
        df['成交股數'] = df['成交股數'].apply(lambda x: x.replace(',', '')).astype(int)
        df['成交筆數'] = df['成交筆數'].apply(lambda x: x.replace(',', '')).astype(int)
        return df

    def __create_field_dictionary(self):
        self.field_dictionary['日期'] = 'Date'
        self.field_dictionary['收盤價'] = 'Closing Price'
        self.field_dictionary['成交金額'] = 'Transaction Amount'
        self.field_dictionary['成交股數'] = 'Trading Volume'
        self.field_dictionary['開盤價'] = 'Opening Price'
        self.field_dictionary['最高價'] = 'Highest Price'
        self.field_dictionary['最低價'] = 'Lowest Price'
        self.field_dictionary['漲跌價差'] = 'Change'
        self.field_dictionary['成交筆數'] = 'Transactions'
        return

    def __store_dataframe_to_csv(self):
        if not os.path.exists(self.data_storage_directory):
            print(f'Creating the target storage folder: {self.data_storage_directory}')
            os.mkdir(self.data_storage_directory)
        path_to_store = self.data_storage_directory + \
                        self.__symbol + \
                        "_stock_data_" + \
                        self.start_date + \
                        ".csv"
        print(f'Ultimate path to storage: {path_to_store}')
        # self.dataframe.to_csv(f'{self.__symbol}_stock_data_{self.start_date}.csv', index=True)
        self.dataframe.to_csv(path_to_store, index=True)

    def display_dataframe_info(self):
        print(f'DataFrame info: {self.dataframe.info()}')
        return

    def crawl_stock_thru_TWSE(self, symbol=None,
                              start_date=None, end_date=None,
                              store_per_month=False,
                              store_dataframe=None,
                              ) -> pd.DataFrame:

        self.__symbol = symbol if symbol is not None else self.__symbol
        self.end_date = end_date if end_date is not None else self.end_date
        self.start_date = start_date if start_date is not None else self.start_date
        store_dataframe = store_dataframe if store_dataframe is not None else self.store_df_to_csv
        if self.end_date is None:
            self.end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        if self.__check_parameters():
            print(f'Crawling stock info of {self.__symbol} from {self.start_date} to {self.end_date}')
        else:
            raise Exception("Missing mandatory field when trying to crawl stock info!")

        s_date = datetime.date(*[int(x) for x in self.start_date.split('-')])
        e_date = datetime.date(*[int(x) for x in self.end_date.split('-')])
        result = pd.DataFrame()
        for dt in rrule.rrule(rrule.MONTHLY, dtstart=s_date, until=e_date):
            print(f'requesting stock info of {dt.strftime("%Y-%m-%d")}', end='...')
            result = pd.concat([result, self.__crawl_one_month(dt)],
                               ignore_index=True)
            print(f'received stock info of {dt.strftime("%Y-%m-%d")}')
            time.sleep(2)
        result = self.__convert_dataframe_format(result)
        self.__create_field_dictionary()
        self.dataframe = result
        if store_dataframe:
            self.__store_dataframe_to_csv()

        return result

    def crawl_stock_thru_yfinance(self, symbol=None):
        if symbol is None:
            symbol = self.__symbol
        else:
            self.__symbol = symbol

    def plot_stock_data(self, info_to_plot):
        fig_w = self.plot_width
        fig_h = self.plot_height

        if type(info_to_plot) is tuple:
            number_of_subplot = len(info_to_plot)
            fig, axs = plt.subplots(number_of_subplot)
            plt.subplots_adjust(hspace=0.6)
            p = 0
            for info in info_to_plot:
                ylabel = self.field_dictionary.get(info)
                self.dataframe.loc[:][info].plot(ax=axs[p],
                                                figsize=(fig_w, fig_h*number_of_subplot),
                                                grid=True,
                                                title=ylabel+' to Date of Stock '+self.__symbol,
                                                legend=False,
                                                xlabel='Date',
                                                ylabel=ylabel,
                                                rot=20
                                                )
                p += 1
        else:
            ylabel = self.field_dictionary.get(info_to_plot)
            self.dataframe.loc[:][info_to_plot].plot(figsize=(fig_w, fig_h),
                                                grid=True,
                                                title=ylabel+' to Date of Stock '+self.__symbol,
                                                legend=False,
                                                xlabel='Date',
                                                ylabel=ylabel,
                                                rot=20
                                                )

        plt.show(block=True)
        return
