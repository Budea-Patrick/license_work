import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

def load_normalized_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    
    features = np.array([item[0] for item in data])
    labels = np.array([item[1] for item in data])
    
    return features, labels

def plot_decision_boundaries(model, X, y, title):
    # Create a mesh to plot the decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='viridis', alpha=0.7)
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend1)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

def main(input_pickle='normalized_augmented_data.pkl', model_filename='svm_model.pkl', method='pca'):
    print("Loading normalized data...")
    features, labels = load_normalized_data(input_pickle)
    
    print("Loading the trained model...")
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    
    print("Scaling the data...")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    print(f"Applying {method.upper()} for dimensionality reduction...")
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("method should be either 'pca' or 'tsne'")
    
    reduced_features = reducer.fit_transform(features)
    
    print("Plotting decision boundaries and data points...")
    plot_decision_boundaries(model, reduced_features, labels, f'{method.upper()} Visualization with Decision Boundaries')

if __name__ == "__main__":
    main()
