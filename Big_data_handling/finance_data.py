"""
handling the finance data 
"""

import pandas as pd

#file path 
FILE_PATH = "/home/abhi/Datasets/ihateabhi.csv"


class Data():
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None


    def make_data(self):
        """
        get the data and split in the data in X and y 
        """
        self.df = pd.read_csv(FILE_PATH, encoding='ISO-8859-1')
 

    def process_data(self):
        """
        function to processing of the data as requiured
        """
        #print(self.df.info())

        bank_name = (self.df["BankName"].unique())

        #print(type(bank_name[0]))

        #filer = self.df[self.df["BankName"].str.contains("BANK")]

        


"""
1. heading with Bankname (doesn't contain CU or FCU or BANK) - remove
2. heading terminmonths just keep 60 months
3  heding DeliveryMethod erase rhe rown with SBA_EXPRES
4. heading Guranteed% - value only between 75 to 95% inclusive

sepearte data in 2009 to 2011, 2011 to 13, 13 to 15, 15 to 17, 17 to 19

"""



if __name__ == "__main__":
    data = Data()
    data.make_data()
    data.process_data()