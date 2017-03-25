import numpy as np
def create_pickle():
    import os
    from PIL import Image
    import pickle
    np.random.seed(5)
    print "Creating Pickle"
    crop_folders = ['ALB', 'BET', 'DOL', 'LAG', 'SHARK', 'YFT']
    X = []
    Y = []
    img_data = {}
    for (label,folder) in enumerate(crop_folders):
        temp_file_names = os.listdir(os.getcwd()+"/train/cropped/"+folder+"/")
        temp_file_names = map(lambda x: os.getcwd()+"/train/cropped/"+folder+"/"+x, temp_file_names)
        Y += [label for x in xrange(len(temp_file_names))]
        X += temp_file_names

    #Creates Images from filepaths
    X = map(lambda f_name: np.asarray( Image.open(f_name), dtype='uint8' ), X)

    X = np.asarray(X,dtype='uint8')
    Y = np.asarray(Y,dtype='uint8')
    Y.shape = (1,Y.shape[0])

    #Shuffles data
    X_train,y_train,X_test,y_test = test_train_split(X,Y,0.7)

    img_data['X_train'] = X_train
    img_data['y_train'] = y_train
    img_data['X_test'] = X_test
    img_data['y_test'] = y_test
    with open('crop_img_data.pkl', 'wb') as my_pickle:
        pickle.dump(img_data,my_pickle)

def load_data():
    import os
    if(os.path.isfile(os.getcwd()+"/crop_img_data.pkl") == False):
        create_pickle()
    import pickle
    with open("crop_img_data.pkl","rb") as img_f:
        data = pickle.load(img_f)
    return (data['X_train'],data['y_train'],data['X_test'],data['y_test'])

def test_train_split(X_data,y_data,percentage):
    perm = np.random.permutation(np.arange(X_data.shape[0]))
    X_data = X_data[perm,:]
    y_data = y_data[:,perm]
    train_size = int(X_data.shape[0]*percentage)

    X_train = X_data[:train_size,:]
    y_train = y_data[:,:train_size]
    X_test = X_data[train_size:,:]
    y_test = y_data[:,train_size:]

    return (X_train,y_train,X_test,y_test)
