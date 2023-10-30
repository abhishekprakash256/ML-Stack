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

        #Apply the first condition , keep the row have Bank and CU
        self.df = pd.read_csv(FILE_PATH, encoding='ISO-8859-1')

        filter_bank_CU = self.df["BankName"].str.contains("CU")
        filter_bank_BANK = self.df["BankName"].str.contains("Bank")


        filtered_bank = self.df[filter_bank_CU | filter_bank_BANK]

        #bank filtering done
        self.df = filtered_bank

        #Apply the second condition , keep only 60
        term_filter = self.df[(self.df["TermInMonths"] >=60) & (self.df["TermInMonths"] <= 120)]

        #print(self.df[term_filter]["TermInMonths"].head(10))

        #apply the term filter done
        self.df = term_filter
        
        #apply the delivery filter

        #delivery_filter = self.df["DeliveryMethod"].str.contains("SBA EXPRES")

        #drop the row with filter
        

        #apply the gurantee filter
        print(self.df.info())
        print(self.df["Gauranteed %"])

        #filter_gurantee_75 = self.df["Gauranteed %"].str.contains("75%")

        
        


        


if __name__ == "__main__":

    data = Data()
    data.make_data()