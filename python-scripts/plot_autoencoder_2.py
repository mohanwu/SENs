from autoencoder_2 import *
import matplotlib.pyplot as plt
import pdb
n = 10
autoencoder.load_weights("cnn_autoencoder_weights.h5")
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(X_crop_test[i:i+1].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i+1 + n)
    result = np.reshape(autoencoder.predict(X_test[i:i+1]),(32,32))
    plt.imshow(result)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
