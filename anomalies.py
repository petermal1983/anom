!pip install pyod



import numpy as np
import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
from pyod.utils.data import generate_data
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#contamination = 0.1  # Доля выбросов
#n_train = 500  # Количество точек в тренировочной выборке
#n_test = 500  # Количество точек в тестовой выборке
#n_features = 25 # Число признаков

#X_train, y_train, X_test, y_test = generate_data(
    #n_train=n_train, n_test=n_test,
    #n_features= n_features, 
    #contamination=contamination,random_state=1234)


from google.colab import drive
merged_data = pd.DataFrame()
drive.mount('/content/drive')
merged_data = pd.read_csv('/content/drive/MyDrive/Averaged_BearingTest_Dataset.csv')
merged_data.head()
#merged_data = pd.DataFrame()
X_train = pd.DataFrame(X_train)
print(X_train)
X_test = pd.DataFrame(X_test)
print(X_test)

X_train = StandardScaler().fit_transform(X_train)
X_train = pd.DataFrame(X_train)
X_test = StandardScaler().fit_transform(X_test)
X_test = pd.DataFrame(X_test)

pca = PCA(4)
x_pca = pca.fit_transform(X_train)
x_pca = pd.DataFrame(x_pca)
x_pca.columns=['Bearing 1','Bearing 2', 'Bearing 3', 'Bearing 4']



plt.scatter(X_train[0], X_train[1], X_train[2],  c=y_train, alpha=0.7)
plt.title('Scatter plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
x_pca.head()

clf1 = AutoEncoder(hidden_neurons =[25, 15, 10, 2, 10, 15, 25])
clf1.fit(X_train)

# Прогнозировуем оценку аномалии
y_train_scores = clf1.decision_scores_  

y_test_scores = clf1.decision_function(X_test) # функция вычисляет расстояние или оценку аномалии для каждой точки данных
y_test_scores = pd.Series(y_test_scores)

# Строим график
plt.hist(pd.Series(y_test_scores), bins='auto')  
plt.title("Histogram for Model Clf1 Anomaly Scores")
plt.show()

outlifts = y_test_scores[y_test_scores >= 4]

outlifts

