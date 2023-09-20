"""
The file to browse the data and download and beak it 
"""

import os
import tarfile
import pandas as pd
from six.moves import urllib



DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


class Loading_data:
	def fetch_housing_data(self,housing_url=HOUSING_URL, housing_path=HOUSING_PATH):

		if not os.path.isdir(housing_path):
			os.makedirs(housing_path)
		tgz_path = os.path.join(housing_path, "housing.tgz")
		urllib.request.urlretrieve(housing_url, tgz_path)
		housing_tgz = tarfile.open(tgz_path)
		housing_tgz.extractall(path=housing_path)
		housing_tgz.close()


	def load_housing_data(self,housing_path=HOUSING_PATH):
		csv_path = os.path.join(housing_path, "housing.csv")
		return pd.read_csv(csv_path)




if __name__ == "__main__":

	load_data = Loading_data()
	load_data.fetch_housing_data()
	data = load_data.load_housing_data()

	print(data)