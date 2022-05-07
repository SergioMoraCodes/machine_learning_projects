import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler # normalizar los datos
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df_heart = pd.read_csv('Platzi_tutorials\Data\heart.csv')
    print(df_heart.head(3))
