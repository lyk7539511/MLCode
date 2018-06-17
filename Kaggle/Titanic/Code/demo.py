import re
import numpy
import pandas
import matplotlib.pyplot
import seaborn
import tensorflow

import warnings
warnings.filterwarnings('ignore')

train_data = pandas.read_csv('D:\\STUDY\\MachineLearning\\MLCode\\Kaggle\\Titanic\\Data\\train.csv')
test_data = pandas.read_csv('D:\\STUDY\\MachineLearning\\MLCode\\Kaggle\\Titanic\\Data\\test.csv')

seaborn.set_style('whitegrid')
train_data.info()