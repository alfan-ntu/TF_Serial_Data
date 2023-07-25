"""
    Brief Description:
        1. Use urllib.request to compose and send a URL request to Taiwan Stock Exchange Corporation
        2. Use json, pandas libraries: json to parse returned JSON response and store the returned
           data to Pandas dataframes
        3. Use matplotlib.pyplot to plot the selected fields to time chart
        4. A component of an AI application to exercise time series prediction techniques from
           certificated course of Coursera

    ToDo's:
        1. Further processing the received data and split it to training and validation datasets for
           model training

    Date: 2023/7/25
    Ver.: 0.1b
    Author:
    Reference: https://medium.com/renee0918/python-%E7%88%AC%E5%8F%96%E5%80%8B%E8%82%A1%E6%AD%B7%E5%B9%B4%E8%82%A1%E5%83%B9%E8%B3%87%E8%A8%8A-b6bc594c8a95B
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

stock = "2330"
begin_date = "2023-01-01"


def save_response(response, stock_number, date):
    """
    Save the response from the web-server to a local file

    :param response: response from the server, suppose the response is in json format
    :param stock_number: stock number string to query
    :param date: starting date of this query in yyyymmdd format
    :return: Pandas data record per month
    """

    try:
      local_file_path = stock_number +'_' + date.strftime('%Y%m%d') + '.json'
      with open(local_file_path, 'wb') as local_file:
        local_file.write(response)

    except Exception as e:
      print(f'Error: {e}')

    return


# Compose URL requests and pack the json response to Panda's DataFrame
def craw_one_month(stock_number, date, keep_it_local=False):
    """
    Get the stock price of specified stock symbol from the specified date

    :param stock_number: stock number to query
    :param date: starting date of this query in yyyymmdd format
    :param keep_it_local: boolean, to determine saving response locally or not
    :return: Pandas data record per month
    """
    url = (
        "https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date=" +
        date.strftime('%Y%m%d') +
        "&stockNo=" +
        stock_number
    )
    headers={"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE"}
    ssl._create_default_https_context = ssl._create_unverified_context
    req = urllib.request.Request(url, headers=headers)
    data = json.loads(urllib.request.urlopen(req).read())
    if keep_it_local:
        save_response(urllib.request.urlopen(req).read(), stock_number, date)

    return pd.DataFrame(data['data'], columns=data['fields'])


# Accept operation arguments, stock number and query start date
def craw_stock(stock_number, start_month, keep_it_local=False):
    """
    Get the stock price month by month

    :param stock_number: stock number to query
    :param start_month: start date of this query
    :param keep_it_local: bool to determine if the replied data has to be stored locally
    :return: Pandas dataframes
    """
    b_month = datetime.date(*[int(x) for x in start_month.split('-')])
    now = datetime.datetime.now().strftime("%Y-%m-%d")  # get the current date
    e_month = datetime.date(*[int(x) for x in now.split('-')])

    result = pd.DataFrame()
    for dt in rrule.rrule(rrule.MONTHLY, dtstart=b_month, until=e_month):
        result = pd.concat([result, craw_one_month(stock_number, dt, keep_it_local)], ignore_index=True)
        time.sleep(2000.0 / 1000.0)

    return result


#
# Helper function set
#
def convert_currency(val):
    """
    Helper function to convert comma separated currency string to float or int

    :param val: pandas data element in comma separated currency string format
    :return: float number with '$', ',' symbols removed
    """
    val = val.replace(',', '').replace('$', '')
    return float(val)


def dataframe_info(df):
    """
    Display dataframe structure of 'df'

    :param df: dataframe of which structure to display
    :return: None
    """
    print(f'Dataframe info: {df.info()}')


def convert_dataframe_format(df):
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


def create_field_dictionary(df):
    """
    Use a dictionary to convert field name from Chinese to English

    :param df: dataframe received from the http server
    :return: converted dataframe
    """
    field_dictionary = {}
    field_dictionary['日期'] = 'Date'
    field_dictionary['收盤價'] = 'Closing Price'
    field_dictionary['成交金額'] = 'Transaction Amount'
    field_dictionary['成交股數'] = 'Trading Volume'
    field_dictionary['開盤價'] = 'Opening Price'
    field_dictionary['最高價'] = 'Highest Price'
    field_dictionary['最低價'] = 'Lowest Price'
    field_dictionary['漲跌價差'] = 'Change'
    field_dictionary['成交筆數'] = 'Transactions'

    return field_dictionary


def plot_stock_data(dataframe, stock_info):
    """
    Plot stock information chart according to the input dataframe and selected info types

    :param dataframe: dataframe returned from the server
    :param stock_info: stock information type selected
    :return: None
    """
    fig_w = 5
    fig_h = 3
    field_dictionary = create_field_dictionary(dataframe)

    if type(stock_info) is tuple:
        number_of_subplot = len(stock_info)
        fig, axs = plt.subplots(number_of_subplot)
        plt.subplots_adjust(hspace=0.5)
        p = 0
        for info in stock_info:
            ylabel = field_dictionary.get(info)
            dataframe.loc[:][info].plot(ax=axs[p],
                                        figsize=(fig_w, fig_h*number_of_subplot),
                                        grid=True,
                                        legend=False,
                                        xlabel='Date',
                                        ylabel=ylabel,
                                        rot=20
                                        )
            p += 1
    else:
        ylabel = field_dictionary.get(stock_info)
        dataframe.loc[:][stock_info].plot(figsize=(fig_w, fig_h),
                                          grid=True,
                                          legend=False,
                                          xlabel='Date',
                                          ylabel=ylabel,
                                          rot=20)

    plt.show(block=False)

    return


def get_list_value_from_dataframe(df, field):
    """
    Get the field values to list of date

    :param df: source dataframe
    :param field: selected value list of the field
    :return: time_list, value_list
    """
    time_list = df.index.tolist()
    value_list = df[field].values.tolist()

    return time_list, value_list


# Crawling the stock information from TWSE
df = craw_stock(stock, begin_date, True)
# dataframe_info(df)
df.set_index("日期", inplace=True)

time_series, value_series = get_list_value_from_dataframe(df, '收盤價')


# Convert the raw dataformat to the format ready to plot
df = convert_dataframe_format(df)

# Plot the response of the specified field
plot_stock_data(df, '收盤價')

# Plot the response of specified fields
plot_stock_data(df, ('收盤價', '成交金額'))

plt.show()