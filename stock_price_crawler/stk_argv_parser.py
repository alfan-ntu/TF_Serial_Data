"""
    Brief Description:
        1. A simple class using 'getopt', 'sys' modules to process arguments
           from the CLI

    ToDo's:

    Date: 2023/7/26
    Ver.: 0.1c
    Author: maoyi.fan@gmail.com
    Reference:

    Revision History:
        v. 0.1c: added simple command line interface and argument parser
"""
import sys
import getopt


class operation_parameters:
    def __init__(self, argv):
        self.stock_symbol = ""
        self.start_date = ""
        self.save_individual = False
        self.save_consolidate = True
        self.parse_argv(argv)

    def parse_argv(self, argv):
        try:        # parse input argument; valid arguments include -h, -s, -d
            opts, args = getopt.getopt(argv, "hic:s:d:",
                                       ["help","individual",
                                        "consolidate=", "symbol=",
                                        "start_date="])
        except getopt.GetoptError:
            print("Syntax error!")
            self.syntax_info()
            sys.exit(2)
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                self.syntax_info()
                sys.exit()
            elif opt in ("-i", "--indiv"):
                self.save_individual = True
            elif opt in ("-c", "--consolid"):
                if arg in ("True", "true"):
                    self.save_consolidate = True
                elif arg in ("False", "false"):
                    self.save_consolidate = False
            elif opt in ("-s", "--symbol"):   # stock id being queried
                self.stock_symbol = arg
            elif opt in ("-d", "--start_date"):     # query start date
                self.start_date = arg
        self.argv_validator()

    def argv_validator(self):
        argv_valid = True
        if self.stock_symbol == "":
            print("Missing Stock Symbol...")
            argv_valid = False
        if self.start_date == "":
            print("Missing Query Start Date...")
            argv_valid = False
        if not argv_valid :
            print("Syntax error!")
            self.syntax_info()
            sys.exit(2)

    def syntax_info(self):
        print("NAME\n\tstock_price_crawler - crawl stock information")
        print("SYNOPSIS\n\tstock_price_crawler [OPTIONS]...")
        print("DESCRIPTION\n\tCrawl the stock information from TWSE according to the command option.")
        print("\tCommand options include:")
        print("\t\t-h, --help\t\tprint this help information")
        print("\t\t-i,"
              "\n\t\t--indiv[=False/True]\tsave stock information of individual time_list slots")
        print("\t\t\t\t\tDefault value is False")
        print("\t\t-c,",
              "\n\t\t--consolid[=True/False]\tsave consolidate stock information received")
        print("\t\t\t\t\tDefault value is True")
        print("\t\t-s,",
              "--symbol[=ID]\tsymbol ID of the requested stock, mandatory field")
        print("\t\t-d,",
              "--date[=yyyy-mm-dd]\tstart date of this query")
