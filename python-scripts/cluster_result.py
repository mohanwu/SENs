# Authors: Fabian Pedregosa <fabian.pedregosa@inria.fr>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Gael Varoquaux
# License: BSD 3 clause (C) INRIA 2011

print(__doc__)
from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from autoencoder_2 import *
from sklearn import (manifold, decomposition, ensemble,
                     discriminant_analysis, random_projection)

class My_data():
    def __init__(self,data,target,s1,s2):
        self.data = data
        self.target = target
        self.images = np.reshape(self.data,(self.data.shape[0],s1,s2))

parent_dir = os.path.dirname(os.getcwd())
no_fish_imgs = crop_encoder_data.load_no_fish()
n = 400
digits = My_data(autoencoder.predict(np.concatenate((X_test[:n],no_fish_imgs[:n]),axis=0)),
                [0 for x in range(n)] + [1 for x in range(n)],32,32)

# digits2 = My_data(np.concatenate(( np.reshape(X_test[:n],(n,128*128)) , np.reshape(no_fish_imgs[:n],(n,128*128)) ),axis=0),
#                 [0 for x in range(n)] + [1 for x in range(n)],128,128)
digits2 = My_data(autoencoder.predict(np.concatenate((X_train[:n],no_fish_imgs[:n]),axis=0)),
                [0 for x in range(n)] + [1 for x in range(n)],32,32)
X = digits.data
y = digits.target
X2 = digits2.data
y2 = digits2.target
n_samples, n_features = X.shape
n_neighbors = 30



#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X,labs,title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], ('x' if labs[i] == 0 else 'o'),
                 color=('green' if labs[i] == 0 else 'red'),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i],bboxprops={'edgecolor':('green' if labs[i] == 0 else 'red')})
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

#----------------------------------------------------------------------
# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne,y,
               "t-SNE embedding of the Fish")

#----------------------------------------------------------------------
#t-SNE embedding of the digits dataset
# print("Computing t-SNE embedding")
# tsne2 = manifold.TSNE(n_components=2, init='pca', random_state=0)
# t1 = time()
# X_tsne2 = tsne2.fit_transform(X2)
#
# plot_embedding(X_tsne2,y2,
#                "t-SNE embedding of the digits (time %.2fs)" %
#                (time() - t1))
plt.show()
