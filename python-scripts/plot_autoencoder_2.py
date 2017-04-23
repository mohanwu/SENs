from autoencoder_2 import *
import matplotlib.pyplot as plt
import pdb
n = 5
import os
parent_dir = os.path.dirname(os.getcwd())
no_fish_imgs = crop_encoder_data.load_no_fish()
fig1 = plt.figure(figsize=(20, 4))
fig1.canvas.set_window_title('Train')
for i in range(n):
    rn = np.random.randint(X_train.shape[0])
    # display picture
    ax = plt.subplot(3, n, i+1 )

    plt.imshow(X_train[rn:rn+1].reshape(128, 128))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display original

    ax = plt.subplot(3, n, i+1 + n)
    plt.imshow(X_crop_train[rn:rn+1].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i+1 + 2*n)
    # THIS IS WHERE the plots are being generated
    result = np.reshape(autoencoder.predict(X_train[rn:rn+1]),(32,32))
    plt.imshow(result)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)



fig2 = plt.figure(figsize=(20, 4))
fig2.canvas.set_window_title('Test')
for i in range(n):
    rn = np.random.randint(X_test.shape[0])
    # display picture
    ax = plt.subplot(3, n, i+1)

    plt.imshow(X_test[rn:rn+1].reshape(128, 128))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display original

    ax = plt.subplot(3, n, i+1 + n)
    plt.imshow(X_crop_test[rn:rn+1].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i+1 + 2*n)
    # THIS IS WHERE the plots are being generated
    result = np.reshape(autoencoder.predict(X_test[rn:rn+1]),(32,32))
    plt.imshow(result)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

fig2 = plt.figure(figsize=(20, 4))
fig2.canvas.set_window_title('NoF')
for i in range(n):
    rn = np.random.randint(no_fish_imgs.shape[0])
    # display picture
    ax = plt.subplot(3, n, i+1)

    plt.imshow(no_fish_imgs[rn:rn+1].reshape(128, 128))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display original

    # ax = plt.subplot(3, n, i+1 + n)
    # plt.imshow(X_crop_test[rn:rn+1].reshape(32, 32))
    # plt.gray()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i+1 + 2*n)
    # THIS IS WHERE the plots are being generated
    result = np.reshape(autoencoder.predict(no_fish_imgs[rn:rn+1]),(32,32))
    plt.imshow(result)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


# plt.figure(figsize=(20,4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i+1)
#     plt.imshow(X_train[i:i+1].reshape(128, 128))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i+1 + n)
#     plt.imshow(X_test[i:i+1].reshape(128, 128))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
plt.show()
