import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset from CSV file
data = pd.read_csv('iris.csv')

# Separating features and targets
X = data[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']]
y = data['Class']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model on the training data
knn.fit(X_train, y_train)

# Save the trained model to a pickle file
pickle.dump(knn, open("iris.pkl", "wb"))