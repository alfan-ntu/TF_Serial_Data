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
from stock_price_crawler import stock_quote_crawler
from model_trainer import stk_price_modeler
from utilities import plot_functions

# Exercise stock crawler functionalities
sc = stock_quote_crawler.stock_crawler()
# sc.symbol = "2330"
sc.data_storage_directory=".\\data_storage\\"
sc.crawl_stock_thru_TWSE(symbol="2330", start_date='2023-06-01')
print(f'target symbol ID: {sc.symbol}')
# sc.display_dataframe_info()
sc.dataframe.set_index("日期", inplace=True)

# print(sc.field_dictionary.get('成交金額'))
sc.plot_stock_data(('收盤價', '成交金額'))

# Exercise stock price prediction model related functionalities
stp = stk_price_modeler.dnn_modeler()
print(stp)

plot_functions.hello()


