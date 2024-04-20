'''
HW: Satellite Image Classification
You may work on this assignment with ONE partner.

We use the SAT-4 Airborne Dataset: https://www.kaggle.com/datasets/crawford/deepsat-sat4.
Download "sat-4-full.mat" from Kaggle and place it in your working dir.

This dataset is large (500K satellite imgs of the Earth's surface).
Each img is 28x28 with 4 channels: red, green, blue, and NIR (near-infrared).
The imgs are labeled to the following 4 classes: 
barren land | trees | grassland | none

The MAT file from Kaggle contains 5 variables:
- annotations (explore this if you want to)
- train_x (400K training images), dim: (28, 28, 4, 400000)
- train_y (400k training labels), dim: (4, 400000)
- test_x (100K test images), dim: (28, 28, 4, 100000)
- test_y (100K test labels), dim: (4, 100000)

For inputs (train_x and test_x):
0th and 1st dim encode the row and column of pixel.
2nd dim describes the channel (RGB and NIR where R = 0, G = 1, B = 2, NIR = 3).
3rd dim encodes the index of the image.

Labels (train_y and test_y) are "one-hot encoded" (look this up).

Your task is to develop two classifiers, SVMs and MLPs, as accurate as you can.
'''

# TASK: Import libraries you need here

from joblib import dump, load # imported this for you, will be useful for workflow speed ups
import scipy.io
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# TASK: Load in the dataset
# Note: Use scipy.io.loadmat
# Note: Dealing with 400K and 100K images will take forever. 
# Feel free to train and test on small subsets (I did 10K and 2.5K, tune as you need).
# Just make sure your subset is rather uniformly distributed and not biased.
# Once you have your x_train, y_train, x_test, y_test variables (or however you name it),
# I suggest you save these variables using dump, then load them in subsequent runs.
# This will make things much faster as you wouldn't need to load in the full dataset each time.

# mat = scipy.io.loadmat('sat-4-full.mat')
# dump(mat,"mat")
# mat = load("mat")

"""
train_x = mat["train_x"][:,:,:,:10000]
train_y = mat["train_y"][:,:10000]
test_x = mat["test_x"][:,:,:,:2500]
test_y = mat["test_y"][:,:2500]

dump(train_x,"train_x")
dump(train_y,"train_y")
dump(test_x,"test_x")
dump(test_y,"test_y")

"""


train_x = load("train_x")
train_y = load("train_y")
test_x = load("test_x")
test_y = load("test_y")


# TASK: Pre-processing
# You need to figure out how to pass in the images as feature vectors to the models.
# You should not simply pass in the entire image as a flattened vector;
# otherwise, it's very slow and just not really effective
# Instead you should extract relevant features from the images.
# Refer to Section 4.1 of https://arxiv.org/abs/1509.03602, especially first three sentences
# and consider what features you want to extract
# And like the previous task, once you have your pre-processed feature vectors,
# you may want to dump and load because pre-processing will also take a while each time.
# MAKE SURE TO PRE-PROCESS YOUR TEST SET AS WELL!


# mean color value
"""
sum_color_vals = np.sum(np.sum(np.sum(train_x[:,:,:3,:],axis = 0),axis = 1),axis = 0)
mean_color_vals = sum_color_vals/(28*28)

dump(mean_color_vals,"mean_color_vals")

sum_color_vals_test = np.sum(np.sum(np.sum(test_x[:,:,:3,:],axis = 0),axis = 1),axis = 0)
mean_color_vals_test = sum_color_vals_test/(28*28)

dump(mean_color_vals_test,"mean_color_vals_test")
"""

mean_color_vals = load("mean_color_vals")
mean_color_vals_test = load("mean_color_vals_test")


# standard deviation
# mean color values for R, B, G of each picture
"""
sum_color_vals_rbg = np.sum(np.sum(train_x[:,:,:3,:],axis = 0),axis = 0)
mean_color_vals_rbg = sum_color_vals_rbg/(28*28)

mean_color_vals_rbg_for_all = np.tile(mean_color_vals_rbg,(28,28,1,1))

standardized_color_vals_rbg = np.subtract(train_x[:,:,:3,:],mean_color_vals_rbg_for_all)**2
sum_std_color_vals = np.sum(np.sum(np.sum(standardized_color_vals_rbg,axis = 0),axis = 1),axis = 0)
std_color_vals = (sum_std_color_vals/(28*28))**(0.5)

dump(std_color_vals,"std_color_vals")


sum_color_vals_rbg_test = np.sum(np.sum(test_x[:,:,:3,:],axis = 0),axis = 0)
mean_color_vals_rbg_test = sum_color_vals_rbg_test/(28*28)

mean_color_vals_rbg_for_all_test = np.tile(mean_color_vals_rbg_test,(28,28,1,1))

standardized_color_vals_rbg_test = np.subtract(test_x[:,:,:3,:],mean_color_vals_rbg_for_all_test)**2
sum_std_color_vals_test = np.sum(np.sum(np.sum(standardized_color_vals_rbg_test,axis = 0),axis = 1),axis = 0)
std_color_vals_test = (sum_std_color_vals_test/(28*28))**(0.5)

dump(std_color_vals_test,"std_color_vals_test")
"""



std_color_vals = load("std_color_vals")
std_color_vals_test = load("std_color_vals_test")



"""
sum_color_vals_rbg = np.sum(np.sum(train_x[:,:,:3,:],axis = 0),axis = 0)

mean_color_vals_rbg = sum_color_vals_rbg/(28*28)

mean_color_vals_rbg_for_all = np.tile(mean_color_vals_rbg,(28,28,1,1))

skew_color_vals_rbg = np.subtract(train_x[:,:,:3,:],mean_color_vals_rbg_for_all)**3
sum_skew_color_vals = np.sum(np.sum(np.sum(skew_color_vals_rbg,axis = 0),axis = 1),axis = 0)
skew_color_vals = np.sign(sum_skew_color_vals)*(np.abs(sum_skew_color_vals/(28*28)))**(1/3)

dump(skew_color_vals,"skew_color_vals")

sum_color_vals_rbg_test = np.sum(np.sum(test_x[:,:,:3,:],axis = 0),axis = 0)
mean_color_vals_rbg_test = sum_color_vals_rbg_test/(28*28)

mean_color_vals_rbg_for_all_test = np.tile(mean_color_vals_rbg_test,(28,28,1,1))

skew_color_vals_rbg_test = np.subtract(test_x[:,:,:3,:],mean_color_vals_rbg_for_all_test)**3
sum_skew_color_vals_test = np.sum(np.sum(np.sum(skew_color_vals_rbg_test,axis = 0),axis = 1),axis = 0)
skew_color_vals_test = np.sign(sum_skew_color_vals_test)*(np.abs(sum_skew_color_vals_test/(28*28)))**(1/3)

dump(skew_color_vals_test,"skew_color_vals_test")
"""

skew_color_vals = load("skew_color_vals")
skew_color_vals_test = load("skew_color_vals_test")


# convert to greyscale

train_x_rbg = train_x[:,:,:3,:]
test_x_rbg = test_x[:,:,:3,:]


train_x_rbg_reshaped = np.moveaxis(train_x_rbg, -1, 0).reshape(2, -1, 3)
# Convert RGB to grayscale manually for all images
train_x_greyscale = np.dot(train_x_rbg_reshaped[..., :3], [0.2989, 0.5870, 0.1140]).reshape(28,28,train_x_rbg.shape[3])

test_x_rbg_reshaped = np.moveaxis(test_x_rbg, -1, 0).reshape(2, -1, 3)
# Convert RGB to grayscale manually for all images
test_x_greyscale = np.dot(test_x_rbg_reshaped[..., :3], [0.2989, 0.5870, 0.1140]).reshape(28,28,test_x_rbg.shape[3])

tranpsosed_train_x_greyscale = np.transpose(train_x_greyscale,(2,0,1))
tranpsosed_test_x_greyscale = np.transpose(test_x_greyscale,(2,0,1))

# 10000x28x28
"""

train_x_entropies = np.array([ shannon_entropy(image) for image in tranpsosed_train_x_greyscale])

test_x_entropies = np.array([ shannon_entropy(image) for image in tranpsosed_test_x_greyscale])

dump(train_x_entropies,"train_x_entropies")
dump(test_x_entropies,"test_x_entropies")
"""


train_x_entropies = load("train_x_entropies")
test_x_entropies = load("test_x_entropies")

"""
# GLCM's
def get_contrast(image):
    image = (image * 255).astype(np.uint8)
    distances = [1]  # Distance between pixels
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles for texture analysis
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    return contrast


train_x_contrasts = np.array([ get_contrast(image) for image in tranpsosed_train_x_greyscale])
test_x_contrasts = np.array([ get_contrast(image) for image in tranpsosed_test_x_greyscale])

dump(train_x_contrasts,"train_x_contrasts")
dump(test_x_contrasts,"test_x_contrasts")

def get_homogeneity(image):
    image = (image * 255).astype(np.uint8)
    distances = [1]  # Distance between pixels
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles for texture analysis
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    return homogeneity


train_x_homogeneity = np.array([ get_homogeneity(image) for image in tranpsosed_train_x_greyscale])
test_x_homogeneity = np.array([ get_homogeneity(image) for image in tranpsosed_test_x_greyscale])

dump(train_x_homogeneity,"train_x_homogeneity")
dump(test_x_homogeneity,"test_x_homogeneity")
"""

"""
def get_glcm(image,prop):
    image = (image * 255).astype(np.uint8)
    distances = [1]  # Distance between pixels
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles for texture analysis
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    return graycoprops(glcm, prop).mean()

print(0)
train_x_energy = np.array([ get_glcm(image,"energy") for image in tranpsosed_train_x_greyscale])
test_x_emergy = np.array([ get_glcm(image,"energy") for image in tranpsosed_test_x_greyscale])
print(1)
train_x_correlation = np.array([ get_glcm(image,"correlation") for image in tranpsosed_train_x_greyscale])
test_x_correlation = np.array([ get_glcm(image,"correlation") for image in tranpsosed_test_x_greyscale])
print(2)
train_x_dissimilarity = np.array([ get_glcm(image,"dissimilarity") for image in tranpsosed_train_x_greyscale])
test_x_dissimilarity = np.array([ get_glcm(image,"dissimilarity") for image in tranpsosed_test_x_greyscale])
print(3)

dump(train_x_energy,"train_x_energy")
dump(test_x_emergy,"test_x_emergy")

dump(train_x_correlation,"train_x_correlation")
dump(test_x_correlation,"test_x_correlation")

dump(train_x_dissimilarity,"train_x_dissimilarity")
dump(test_x_dissimilarity,"test_x_dissimilarity")
"""

train_x_contrasts = load("train_x_contrasts")
test_x_contrasts = load("test_x_contrasts")

train_x_homogeneity = load("train_x_homogeneity")
test_x_homogeneity = load("test_x_homogeneity")

train_x_energy = load("train_x_energy")
test_x_emergy = load("test_x_emergy")

train_x_correlation = load("train_x_correlation")
test_x_correlation = load("test_x_correlation")

train_x_dissimilarity = load("train_x_dissimilarity")
test_x_dissimilarity = load("test_x_dissimilarity")


# create feature vector
X_train = np.column_stack((mean_color_vals, std_color_vals,skew_color_vals,train_x_entropies,train_x_contrasts,train_x_homogeneity,train_x_energy,train_x_correlation,train_x_dissimilarity))
# for test
X_test = np.column_stack((mean_color_vals_test, std_color_vals_test,skew_color_vals_test,test_x_entropies,test_x_contrasts,test_x_homogeneity,test_x_emergy,test_x_correlation,test_x_dissimilarity))


y_train = np.where(train_y.T==1)[1]
y_test = np.where(test_y.T==1)[1]




# TASK: TRAIN YOUR MODEL
# You have your feature vectors now, time to train.
# Again, train two models: SVM and MLP.
# Make them as accurate as possible. Tune your hyperparameters.
# Check for overfitting and other potential flaws as well.
"""
clf_svm = svm.SVC(kernel = "linear")
clf_svm.fit(X_train,y_train)

y_pred = clf_svm.predict(X_test)
print(np.unique(y_pred))

svm_accuracy = accuracy_score(y_test, y_pred, normalize = True)
print(svm_accuracy)
"""
"""
clf_mlp = MLPClassifier(hidden_layer_sizes=(150,100,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)

clf_mlp.fit(X_train, y_train)

y_pred = clf_mlp.predict(X_test)

mlp_accuracy = accuracy_score(y_test, y_pred, normalize = True)
print(mlp_accuracy)

"""
# TASK: Visualizations
# Produce two visualizations, one for SVM and one for MLP.
# These should show the justifications for choosing your hyperparameters to your classifiers,
# such as kernel type, C value, gamma value, etc. for SVM or layer sizes, depths, itersm etc. for MLPs

# mlp visualization
"""
alpha_range = np.logspace(-5, 2, 10)
train_scores, valid_scores = validation_curve(clf_mlp, X_train, y_train, param_name="alpha", param_range=alpha_range,
                                              cv=5, scoring="accuracy", n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)

# Plot validation curve
plt.figure(figsize=(8, 6))
plt.plot(alpha_range, train_scores_mean, label="Training score", color="blue")
plt.plot(alpha_range, valid_scores_mean, label="Cross-validation score", color="red")
plt.fill_between(alpha_range, train_scores_mean - np.std(train_scores, axis=1),
                 train_scores_mean + np.std(train_scores, axis=1), alpha=0.1, color="blue")
plt.fill_between(alpha_range, valid_scores_mean - np.std(valid_scores, axis=1),
                 valid_scores_mean + np.std(valid_scores, axis=1), alpha=0.1, color="red")
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.title('Validation Curve for MLP')
plt.legend(loc="best")
plt.show()
"""
# svm visualization

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
X_pca = pca.fit_transform(X_scaled)

# Train SVM model
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_pca, y_train)

X_test_scaled = scaler.fit_transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC')
# Set-up grid for plotting.
X0, X1 = X_pca[:, 0], X_pca[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, svm_model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('PCA 2')
ax.set_xlabel('PCA 1')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()

# Visualize