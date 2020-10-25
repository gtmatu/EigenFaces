import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics, model_selection

####################################################
####################################################
# show image from 1-D array
FIGURE_NO = 1
def show_image(pixels, label="", cmap='gray', independent=True, axis=True):
    global FIGURE_NO
    picH = 56
    picW = 46

    if independent:
        plt.figure(FIGURE_NO)
    if not axis:
        plt.axis('off')
    plt.imshow(pixels.reshape(picH, picW), cmap=cmap)
    if label:
        plt.title(label)
    FIGURE_NO += 1

####################################################
####################################################
# Return list of 2 different ints in range [0,9]
def get_random_pair():
    a = np.random.randint(10) 
    b = np.random.randint(10)
    while b == a:
        b = np.random.randint(10)
    return [a,b]

###############################################################
###############################################################
# Split dataset into train and test by picking random elements 
# of same class
def train_test_split(x_data, y_data):
    x_train = np.array([], dtype=int)
    y_train = np.array([], dtype=int)
    x_test = np.array([], dtype=int)
    y_test = np.array([], dtype=int)
    classes = 52
    for i in range(classes):
        test_indeces = get_random_pair()
        train_indeces = [t for t in range(10)]
        for a in test_indeces:
            x_test = np.append(x_test, x_data[a+10*i])
            y_test = np.append(y_test, y_data[a+10*i])
            train_indeces.remove(a)
        for b in train_indeces:
            x_train = np.append(x_train, x_data[b+10*i])
            y_train = np.append(y_train, y_data[b+10*i])   
    # reshape arrays
    shapeX = x_data.shape
    shapeY = y_data.shape
    x_train = x_train.reshape((int(shapeX[0]*0.8), shapeX[1]))
    y_train = y_train.reshape((int(shapeY[0]*0.8)))
    x_test = x_test.reshape((int(shapeX[0]*0.2), shapeX[1]))
    y_test = y_test.reshape((int(shapeY[0]*0.2)))
    return(x_train, y_train, x_test, y_test)

###############################################################
###############################################################
# Perform low-dimensional PCA
def pca(X, n_pc):
    n_samples, n_features = X.shape
    mean = np.mean(X, axis=0)
    centered_data = X-mean
    U, S, V = np.linalg.svd(centered_data)
    components = V[:n_pc]
    projected = U[:,:n_pc]*S[:n_pc]
    return projected, components, mean, centered_data

###############################################################
###############################################################
# Transform X to PCA subspace
def transform(X, components, mean):
    X = X - mean
    X_trans = np.dot(X, components.T)
    return X_trans

###############################################################
###############################################################
# Perform PCA on dataset X and return mean and std dev of
# the reconstruction error. n_pc -> num of principal components
def get_reconstruction_error(X, n_pc):
    proj, comp, mean, centered_data = pca(X, n_pc)
    num_samples = centered_data.shape[0]

    error = np.empty(num_samples, dtype=float)
    for i in range(num_samples):
        error[i] = np.linalg.norm(X[i]-np.matmul(proj[i],comp)-mean)
    
    return np.mean(error), np.std(error)

###############################################################
###############################################################
# Split X and Y data into N subsets of same size
def split_dataset(X, Y, N):
    n_samples, n_features = X.shape
    X_split = np.zeros(shape=(N, int(n_samples/N), n_features), dtype=int)
    Y_split = np.zeros(shape=(N, int(n_samples/N)), dtype=int)

    for i in range(int(n_samples/N)):
        for j in range(N):
            X_split[j][i] = X[i*N +j]
            Y_split[j][i] = Y[i*N +j]
    
    return (X_split, Y_split)

###############################################################
###############################################################
# Plot a random image from training data, it's reconstruction 
# and the difference between images. Returns the image error.
def plot_sample(X, n_pc):
    proj, comp, mean, centered_data = pca(X, n_pc)
    
    sample = np.random.randint(X.shape[0])
    
    plt.subplot(1,3,1)
    show_image(np.matmul(proj[sample],comp)+mean, label="Reconstructed", independent=False)
    plt.subplot(1,3,2)
    show_image(X[sample], label="Original", independent=False)
    plt.subplot(1,3,3)
    show_image(X[sample]-np.matmul(proj[sample],comp)-mean, label="Error", independent=False)

    return np.linalg.norm(X[sample]-np.matmul(proj[sample],comp)-mean)
    
###############################################################
###############################################################
# Get classification accuracy of NN classifier, setting cm to 
# True makes it print out confusion matrix
def get_classification_score(x_train, y_train, x_test, y_test, k=1, cm=False):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    if cm:
        cm = metrics.confusion_matrix(y_test, pred)
  
        plt.figure(figsize = (12,10))
        sn.heatmap(cm, annot=False,annot_kws={"size": 3})# font size
        plt.show()
        
    return metrics.accuracy_score(y_test, pred)
    
    
###############################################################
###############################################################
# Splits dataset into seperate classes for LDA, assumes equal
# number of images per class, and ordered by class in dataset
def split_classes(X, n_classes=52):
    n_samples, n_features = X.shape
    samples_per_class = int(n_samples/n_classes)
    X_split = np.empty((n_classes, samples_per_class, n_features))
    for i in range(n_classes):
        for j in range(samples_per_class):
            X_split[i][j] = X[i*samples_per_class+j]
            
    return X_split

###############################################################
###############################################################
# Generates between class and within class scatter matrices
# for Fisherfaces
def generate_scatter(X):
    X_split = split_classes(X)
    
    n_classes = X_split.shape[0]
    n_features = X_split.shape[2]
    global_mean = np.mean(X, axis=0)
    scatter_between = np.zeros((n_features, n_features))
    scatter_within = np.zeros((n_features, n_features))

    for c in range(n_classes):
        X_class = X_split[c]
        class_mean = np.mean(X_class, axis=0)
        
        diff = np.subtract(class_mean, global_mean)
        diff = np.reshape(diff, (diff.shape[0], 1))
        scatter_between=np.add(np.dot(diff, diff.T), scatter_between)

        n_pics = X_class.shape[0]
        for x in range(n_pics):
            diff = (X_class[x] - class_mean) * n_pics
            diff = np.reshape(diff, (diff.shape[0], 1))
            scatter_within=np.add(np.dot(diff, diff.T), scatter_within)
            
    return scatter_between, scatter_within

###############################################################
###############################################################
# Fisherfaces - combines PCA + LDA, returns transformed 
# training and testing set

def fisherface(x_train, y_train, x_test, M_pca, M_lda):
    proj, comp, mean, centered_data = pca(x_train, M_pca)
    x_test = transform(x_test, comp, mean)
    
    lda = LDA(n_components=M_lda)
    x_train = lda.fit_transform(proj, y_train)
    x_test = lda.transform(x_test)
    
    return x_train, x_test

###############################################################
###############################################################
# 
def nn_bag(x_train, y_train, n_estimators=10, samples=0.5, features=0.5):
    knn = KNeighborsClassifier(n_neighbors=1)
    
    bagging = BaggingClassifier(knn, n_estimators=n_estimators, max_samples=samples, max_features=features)
    bagging.fit(x_train, y_train)
    
    return bagging


###############################################################
###############################################################
# 
def nn_pca_classifier(X_train, Y_train, X_test, Y_test, n_pc):
    proj, comp, mean, centered_data = pca(X_train, n_pc)
    test_centered = transform(X_test, comp, mean)
    
    score = get_classification_score(proj, Y_train, test_centered, Y_test)
    
    return score
###############################################################
###############################################################
# 
def random_indeces_sampling(min_idx, max_idx, n):
    out = []
    for _ in range(n):
        num = np.random.randint(min_idx, max_idx)
        while num in out:
            num = np.random.randint(min_idx, max_idx)
        out.append(num)
    return out

###############################################################
###############################################################
# 
def get_random_subset(data, data1, min_idx, max_idx, n):
    n_components, n_features = data.shape
    out = np.empty(shape=(n, n_features))
    out1 = np.empty(shape=(n))
    indeces = random_indeces_sampling(min_idx, max_idx, n)
    for pos, i in enumerate(indeces):
        out[pos] = data[i]
        out1[pos] = data1[i]
    return out, out1

###############################################################
###############################################################
# 
def sum_fusion(predictions):
    n_models = len(predictions)
    n_samples, n_classes = predictions[0].shape
    
    out = np.empty(shape=(n_samples), dtype=int)
    
    for sample_idx in range(n_samples):
        sum_vec = np.zeros(shape=(n_classes), dtype=float)
        for model_idx in range(n_models):
            sum_vec += predictions[model_idx][sample_idx]
        predicted_class = np.argmax(sum_vec)
        out[sample_idx] = predicted_class+1
    return out
###############################################################
###############################################################
# 
def majority_fusion(predictions):
    n_models = len(predictions)
    n_samples, n_classes = predictions[0].shape
    
    out = np.empty(shape=(n_samples), dtype=int)
    
    for sample_idx in range(n_samples):
        votes_vec = np.zeros(shape=(n_classes), dtype=int)
        for model_idx in range(n_models):
            class_vote = np.argmax(predictions[model_idx][sample_idx])
            votes_vec[class_vote] += 1 
        predicted_class = np.argmax(votes_vec)
        out[sample_idx] = predicted_class+1
    return out