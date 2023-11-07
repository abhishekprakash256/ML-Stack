"""
pandas to maipulate the file
"""
import pandas as pd 


FILE_PATH = "/home/ubuntu/s3/ihateabhi.csv"

df = pd.read_csv(FILE_PATH, encoding='ISO-8859-1')