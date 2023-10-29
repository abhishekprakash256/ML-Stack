"""
make a dataset and filter the dataset
"""

"""
1. heading with Bankname (doesn't contain CU or FCU or BANK) - remove
2. heading terminmonths just keep 60 months
3  heding DeliveryMethod erase rhe rown with SBA_EXPRES
4. heading Guranteed% - value only between 75 to 95% inclusive

sepearte data in 2009 to 2011, 2011 to 13, 13 to 15, 15 to 17, 17 to 19

"""

import pandas as pd

#file path 
FILE_PATH = "out.csv"


df = pd.read_csv(FILE_PATH, encoding='ISO-8859-1')

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

        #print(self.df["BankName"].head(20))

        #print(self.df["BankName"].unique())

        print(self.df["BankName"].info())

        filter_bank_CU = self.df["BankName"].str.contains("CU")
        filter_bank_FCU = self.df["BankName"].str.contains("FCU")
        filter_bank_BANK = self.df["BankName"].str.contains("BANK")


        filtered_bank = self.df[filter_bank_CU | filter_bank_FCU | filter_bank_BANK]

        print(filtered_bank)

        

        


if __name__ == "__main__":

    data = Data()
    data.make_data()