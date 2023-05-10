# Group members:
# Ceccon Gioele - 2079425
# Nardella Gaia - 2091413
# Renna Pietro
# Rocca Valerio - 2094861




# Import
from xml.dom import pulldom
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score
import time


# Part 1: point generation (10000 points (5000 per class), 5% labeled)
# Set up parameters for the L shape

def generate_points(n_points=1000, weights = [0.25, 0.25, 0.25, 0.25]):
    # Define the means and covariance matrices for the four clusters
    cluster_params = [
        {
            'mean': np.array([-2, 0]),
            'cov': np.array([[7, 0], [0, 0.5]])
        },
        {
            'mean': np.array([2, 4]),
            'cov': np.array([[0.5, 0], [0, 4]])
        },
        {
            'mean': np.array([2, 11]),
            'cov': np.array([[7, 0], [0, 0.5]])
        },
        {
            'mean': np.array([-2, 7]),
            'cov': np.array([[0.5, 0], [0, 4]])
        }
    ]

    samples = np.empty((0, 3))

    #giro sui cluster
    for i, cluster in enumerate(cluster_params):
        # Estraggo punti
        sample = np.random.multivariate_normal(cluster['mean'], cluster['cov'], int(n_points*weights[i]))
        # Se i punti sono stati estratti dai primi due cluster, assegno ai punti label 1.
        # Se i punti sono stati estratti dagli altri due cluster, assegno ai punti label -1
        if i==0 or i==1:
            label=1
        elif i==2 or i==3:
            label =-1
        
        # np.full() is a NumPy function that creates a new array of a specified shape and fills it with a given scalar value (label)
        # The axis=1 argument tells np.concatenate() to concatenate the arrays horizontally (i.e., by adding columns), so that the label
        # column is added to the right of the sample array.
        sample = np.concatenate((sample, np.full((sample.shape[0], 1), label)), axis=1)
        #unisco tutti i punti dei 4 cluster in un unico array che per ogni elemento ha [cordinata x, cordinata y, label]
        samples = np.concatenate((samples, sample))
    
    for i in range(len(samples)):
        # dopo ogni 19esimo punto ho un punto di cui so l'etichetta, agli altri setto l'etichetta a zero (si conosce l'etichetta del 5% dei punti)
        if i % 20 != 0.0:
            samples[i][2] = 0

    # Split the samples based on their class label
    unlabeled = samples[samples[:, 2] == 0]
    class1= samples[samples[:, 2] == 1]
    class2 = samples[samples[:, 2] == -1]
    labeled = np.concatenate((class1,class2))

    return unlabeled, class1, class2, labeled  # restituisce una tupla


    # Shuffle the points
    # shuffle_indices = np.random.permutation(n_points)
    # X = X[shuffle_indices]
    # y = y[shuffle_indices]

def generate_random_labels(unlabeled):
    # Assign random label to unlabeled units

    random_labels=np.random.choice([-1, 1], size=np.shape(unlabeled)[0], p=[0.5, 0.5])
    random_labels_samples = np.concatenate((unlabeled[:,0:2], np.reshape(random_labels, (-1, 1))), axis=1)

    return random_labels_samples

def plot_points(unlabeled_samples, class1_samples, class2_samples):
    # Plot the samples of each class with a different color and marker
    plt.scatter(unlabeled_samples[:, 0], unlabeled_samples[:, 1], color='grey', label='Unlabeled', alpha=0.7)
    plt.scatter(class1_samples[:, 0], class1_samples[:, 1], color='red', label='Label 1', alpha=0.7)
    plt.scatter(class2_samples[:, 0], class2_samples[:, 1], color='blue', label='Label -1', alpha=0.7)

    # Add legend and axis labels
    plt.legend()
    plt.xlabel('X_1')
    plt.ylabel('X_2')

    # Show the plot
    plt.show()

# Normalizes a given matrix using min-max scaling method.
# It calculates the minimum and maximum values of the matrix using np.min() and np.max() functions
# Then, it scales the values in the matrix between 0 and 1 by subtracting the minimum value from each value
# and dividing by the range (max value minus min value).

# def min_max_normalize_matrix(matrix):
#     min_val = np.min(matrix)
#     max_val = np.max(matrix)
#     return (matrix - min_val) / (max_val - min_val)

def similarity_matrix(unlabeled, labeled):
    w = np.zeros((np.shape(unlabeled)[0], np.shape(labeled)[0]))
    w_bar = np.zeros((np.shape(unlabeled)[0], np.shape(unlabeled)[0]))

    # similarity matrix unlabeled-labeled
    for row in range(np.shape(w)[0]): #949 punti con label = 0
        for col in range(np.shape(w)[1]): #50 punti con label = 1 oppure = -1
            w[row, col] = np.linalg.norm(unlabeled_samples[row, :2] - labeled_samples[col, :2])
            # w[row, col] restituisce tutti float positivi

    # similarity matrix unlabeled-unlabeled
    for row in range(np.shape(w_bar)[0]):
        for col in range(np.shape(w_bar)[1]):
            w_bar[row, col] = np.linalg.norm(unlabeled_samples[row, :2] - unlabeled_samples[col, :2])
            #w_bar[row, col] restituisce tutti float positivi

    # w = min_max_normalize_matrix(w)
    # w_bar = min_max_normalize_matrix(w_bar)
    
    return w, w_bar

# Threshold selector
# def threshold_sel(y_lab_norm):
#     y_lab_out = np.copy(y_lab_norm)
#     y_lab_out[y_lab_out < 0.5] = -1
#     y_lab_out[y_lab_out >= 0.5] = 1
#     return y_lab_out


# Gradient Descent

def gradient(labeled, unlabeled, w, w_barr):

    n_unlabeled = np.shape(unlabeled_samples)[0] #949
    n_labeled = np.shape(labeled_samples)[0] #50
    grads = np.zeros(n_unlabeled)

    for j in range(n_unlabeled):
        print(w[j, :n_labeled]) #restituisce 50 valori
    return
    #     labeled_term = np.sum(w[j, :n_labeled] * (unlabeled_samples[j, 2] - labeled_samples[:, 2]))
    #     unlabeled_term = np.sum(w_bar[j, :n_unlabeled] * (unlabeled_samples[j, 2] - unlabeled_samples[:, 2]))
    #     grads[j] = 2 * (labeled_term + unlabeled_term)
        
    # return grads

# def gradient(lab_samples, unlab_samples, w, w_bar):
#     grads = []
#     for j in range(np.shape(unlab_samples)[0]):  # Da 0 a 947
#         grad = 0
#         for i in range(np.shape(lab_samples)[0]):
#             grad += w[j][i] * (unlab_samples[j][2] - lab_samples[i][2])
#         for i in range(np.shape(unlab_samples)[0]):
#             grad += w_bar[j][i] * (unlab_samples[j][2] - unlab_samples[i][2])
#         grads.append(2 * grad)  # / (np.shape(lab_samples)[0] + np.shape(unlab_samples)[0])) #GRADIENT+NORMALIZATION
#     return np.array(grads)


if __name__ == "__main__":

    # Part 1

    samples = generate_points()
    unlabeled_samples = samples[0]
    class1_samples = samples[1]
    class2_samples = samples[2]
    labeled_samples = samples[3]

    random_unlabeled_samples = generate_random_labels(unlabeled_samples)

    plot_points(unlabeled_samples, class1_samples, class2_samples)
    print(f"There are {len(labeled_samples)} labeled instances: {np.round(((len(labeled_samples)/len(samples))*100),1)}% of the total.")

    # Part 2

    weights = similarity_matrix(unlabeled_samples, labeled_samples)
    w=weights[0]
    w_bar=weights[1]
    
    # Part 3

    print(gradient(labeled_samples, unlabeled_samples, w, w_bar))

