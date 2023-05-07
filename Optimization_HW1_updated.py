from xml.dom import pulldom
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score

# Part 1: point generation (10000 points (5000 per class), 5% labeled)
# Set up parameters for the L shape
mu1 = np.array([-2, 0])
mu2 = np.array([2, 4])
mu3 = np.array([2, 11])
mu4 = np.array([-2, 7])
sigma1 = np.array([[7, 0], [0, 0.5]])
sigma2 = np.array([[0.5, 0], [0, 4]])  # Modified standard deviation for sample 2
sigma3 = np.array([[7, 0], [0, 0.5]])
sigma4 = np.array([[0.5, 0], [0, 4]])  # Modified standard deviation for sample 4
weights = [0.25, 0.25, 0.25, 0.25]

# Generate random points in the L shape using the Bivariate Gaussian Mixture Distribution
n_points = 1000  # change to 10000
samples1 = np.random.multivariate_normal(mu1, sigma1, int(n_points * weights[0]))
samples2 = np.random.multivariate_normal(mu2, sigma2, int(n_points * weights[1]))
samples3 = np.random.multivariate_normal(mu3, sigma3, int(n_points * weights[2]))
samples4 = np.random.multivariate_normal(mu4, sigma4, int(n_points * weights[3]))

# Assign class labels to the samples
# class1 = np.concatenate((samples2, samples1))
class1 = np.concatenate((samples2, samples1))
class1 = np.c_[class1, np.zeros(len(class1))]
count = 0
for i in class1:
    if count == 19:
        count = 0
        i[2] = 1
    count += 1
class2 = np.concatenate((samples3, samples4))
class2 = np.c_[class2, np.zeros(len(class2))]
count = 0
for i in class2:
    if count == 19:
        count = 0
        i[2] = -1
    count += 1

# Combine the samples of all three classes
all_samples = np.concatenate((class1, class2))

# Split the samples based on their class label
unlabeled_samples = all_samples[all_samples[:, 2] == 0]
class1_samples = all_samples[all_samples[:, 2] == 1]
class2_samples = all_samples[all_samples[:, 2] == -1]

# Labeled matrix
labeled_samples = np.concatenate((class1_samples, class2_samples))

# Assign random label to unlabeled units
random_unlabeled = np.copy(unlabeled_samples)
random_unlabeled[:, 2] = np.random.choice([-1, 1], size=np.shape(unlabeled_samples)[0], p=[0.5, 0.5])
prova = np.copy(random_unlabeled[:, 2])

# Plot the samples of each class with a different color and marker
plt.scatter(unlabeled_samples[:, 0], unlabeled_samples[:, 1], color='grey', label='Unlabeled', alpha=0.5)
plt.scatter(class1_samples[:, 0], class1_samples[:, 1], color='red', label='Label 1', alpha=0.5)
plt.scatter(class2_samples[:, 0], class2_samples[:, 1], color='blue', label='Label -1', alpha=0.5)

# Add legend and axis labels
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')

# Show the plot
# plt.show()

# ---------------------------------------------------------------

# Part 2: similarity function distance = numpy.linalg.norm(a-b)
# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# distance = np.linalg.norm(a-b)
# print(distance)

w = np.zeros((np.shape(unlabeled_samples)[0], np.shape(labeled_samples)[0]))
w_bar = np.zeros((np.shape(unlabeled_samples)[0], np.shape(unlabeled_samples)[0]))

# similarity matrix unlabeled-labeled
for row in range(np.shape(w)[0]):
    for col in range(np.shape(w)[1]):
        w[row, col] = np.linalg.norm(unlabeled_samples[row, :2] - labeled_samples[col, :2])

# similarity matrix unlabeled-unlabeled
for row in range(np.shape(w_bar)[0]):
    for col in range(np.shape(w_bar)[1]):
        w_bar[row, col] = np.linalg.norm(unlabeled_samples[row, :2] - unlabeled_samples[col, :2])

def min_max_normalize_matrix(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val)

w = min_max_normalize_matrix(w)
w_bar = min_max_normalize_matrix(w_bar)

# -----------------------------------------------------------------------
# Part 3: 

# w.shape = (948,52)
# w_bar.shape = (948,948)
# labeled_samples.shape = (52,3)
# unlabeled_samples.shape = (948,3)

# Threshold selector
def threshold_sel(y_lab_norm):
    y_lab_out = np.copy(y_lab_norm)
    y_lab_out[y_lab_out < 0.5] = -1
    y_lab_out[y_lab_out >= 0.5] = 1
    return y_lab_out


# Gradient Descent

def gradient(lab_samples, unlab_samples, w=w, w_bar=w_bar):
    grads = []
    for j in range(np.shape(unlab_samples)[0]):  # Da 0 a 947
        grad = 0
        for i in range(np.shape(lab_samples)[0]):
            grad += w[j][i] * (unlab_samples[j][2] - lab_samples[i][2])
        for i in range(np.shape(unlab_samples)[0]):
            grad += w_bar[j][i] * (unlab_samples[j][2] - unlab_samples[i][2])
        grads.append(2 * grad)  # / (np.shape(lab_samples)[0] + np.shape(unlab_samples)[0])) #GRADIENT+NORMALIZATION
    return np.array(grads)


def gradient_descent(lab_samples, unlab_samples, alpha=0.001, epochs=500):
    y_lab = np.copy(unlab_samples[:, 2])
    for i in range(epochs):
        grads = gradient(lab_samples, unlab_samples)
        y_lab -= alpha * grads
    return y_lab


y_lab_gd = gradient_descent(labeled_samples, random_unlabeled)
y_lab_gd_p = threshold_sel(y_lab_gd)
print("Accuracy for Gradient Descent")
# accuracy_score(unlabeled_samples[:, 2], y_lab_gd_p)
print("Number of 1 in Gradient Descent {}".format(np.sum(y_lab_gd_p == 1) / len(y_lab_gd_p)))
print("Number of -1 in Gradient Descent {}".format(np.sum(y_lab_gd_p == -1) / len(y_lab_gd_p)))


# Randomized BCGD

def rand_gradient(lab_samples, unlab_samples, w=w, w_bar=w_bar):
    grads = np.zeros(np.shape(unlabeled_samples)[0])
    j = np.random.randint(0, np.shape(unlab_samples)[0])
    j = int(j)
    grad = 0
    for i in range(np.shape(lab_samples)[0]):
        grad += w[j][i] * (unlab_samples[j][2] - lab_samples[i][2])
    for i in range(np.shape(unlab_samples)[0]):
        grad += w_bar[j][i] * (unlab_samples[j][2] - unlab_samples[i][2])
    grads[j] = 2 * grad  # / (np.shape(lab_samples)[0] + np.shape(unlab_samples)[0])
    return grads


def rand_bcgd(lab_samples, unlab_samples, alpha=0.001, epochs=1200):
    y = np.copy(unlab_samples)
    for _ in range(epochs):
        grads = rand_gradient(lab_samples, y)
        y[:, 2] -= alpha * grads
    y_lab = np.copy(y[:, 2])
    return y_lab


y_lab_rand_bcgd = rand_bcgd(labeled_samples, random_unlabeled)
y_lab_rand_bcgd_p = threshold_sel(y_lab_rand_bcgd)
print("Accuracy for Randomized BCGD")
# accuracy_score(unlabeled_samples[:, 2], y_lab_rand_bcgd_p)
print("Frequency of 1 in Randomized BCGD {}".format(np.sum(y_lab_rand_bcgd_p == 1) / len(y_lab_rand_bcgd_p)))
print("Frequency of -1 in Randomized BCGD {}".format(np.sum(y_lab_rand_bcgd_p == -1) / len(y_lab_rand_bcgd_p)))


# Gauss Sauthwell BCGD

def max_gradient(lab_samples, unlab_samples):
    grad = gradient(lab_samples, unlab_samples)
    j = np.argmax(np.abs(gradient(lab_samples, unlab_samples)))
    grads = np.zeros(np.shape(unlabeled_samples)[0])
    grads[j] = grad[j]
    return grads

def gs_bcgd(lab_samples, unlab_samples, alpha=0.001, epochs=1200):
    y = np.copy(unlab_samples)
    for _ in range(epochs):
        grads = max_gradient(lab_samples, y)
        y[:, 2] -= alpha * grads
    y_lab = np.copy(y[:, 2])
    return y_lab


y_lab_gs_bcgd = gs_bcgd(labeled_samples, random_unlabeled)
y_lab_gs_bcgd_p = threshold_sel(y_lab_gs_bcgd)
print("Accuracy for Gauss Sauthwell BCGD")
# accuracy_score(unlabeled_samples[:, 2], y_lab_gs_bcgd_p)

print("Frequency of 1 in Gauss Sauthwell BCGD {}".format(np.sum(y_lab_gs_bcgd_p == 1) / len(y_lab_gs_bcgd_p)))
print("Frequency of -1 in Gauss Sauthwell BCGD {}".format(np.sum(y_lab_gs_bcgd_p == -1) / len(y_lab_gs_bcgd_p)))

print(prova == y_lab_gs_bcgd_p)

