"""
    Brief Description:
        1. A simple class using 'getopt', 'sys' modules to process arguments
           from the CLI to support data crawling mode, model training mode

    ToDo's:

    Date: 2023/8/2
    Ver.: 0.2a
    Author: maoyi.fan@gmail.com
    Reference:

    Revision History:
        v. 0.2a: newly created
"""
import sys
import getopt

class op_parameters:
    def __init__(self, argv):
        # Crawling mode operation parameters
        self.stock_symbol = ""
        self.__start_date = ""
        self.save_individual = False
        self.save_consolidate = True
        self.op_mode = ""
        self.parse_argv(argv)

    @property
    def start_date(self) -> str:
        return self.__start_date

    @start_date.setter
    def start_date(self, value):
        # ToDo's: check if value is of correct date format before
        #         assignment actually done
        self.__start_date = value

    def parse_argv(self, argv):
        try:
            opts, args = getopt.getopt(argv,
                                       "hic:s:d:",  # short options
                                       ["help", "individual",
                                        "consolidate=", "symbol=",
                                        "start_date="]  # long options list
                                       )
        except getopt.GetoptError:
            print("Syntax error!")
            self.syntax_info()
            sys.exit(2)
        for arg in args:
            if arg == "crawl":
                self.op_mode = "crawl"
            elif arg == "model":
                self.op_mode = "model"
            else:
                raise Exception("Operation mode set error!")
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                self.syntax_info()
                sys.exit()
            elif opt in ("-i", "--indiv"):
                self.save_individual = True
            elif opt in ("-s", "--symbol"):
                self.stock_symbol = arg
            elif opt in ("-d", "--start_date"):
                self.__start_date = arg

    def syntax_info(self):
        """
        stock_cpml.py supports both crawl(ing), model(ing) modes
        :return:
        """
        print("Syntax: stock_cpml....")