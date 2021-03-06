import numpy as np
import os
parent_dir = os.path.dirname(os.getcwd())

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
        orig_file_names = os.listdir(parent_dir+"/train/cropped/"+folder+"/")
        X += map(lambda x: parent_dir+"/train/"+folder+"/"+x, orig_file_names)
        crop_file_names = map(lambda x: parent_dir+"/train/cropped/"+folder+"/"+x, orig_file_names)
        Y += crop_file_names


    #Creates Images from filepaths
    X = map(lambda f_name: np.asarray( Image.open(f_name).convert('L').resize((128,128)), dtype='uint8' ), X)
    Y = map(lambda f_name: np.asarray( Image.open(f_name).convert('L').resize((32,32)), dtype='uint8' ), Y)
    X = np.asarray(X,dtype='uint8')
    Y = np.asarray(Y,dtype='uint8')

    #Shuffles data
    X_train,y_train,X_test,y_test = test_train_split(X,Y,0.7)

    img_data['X_train'] = X_train
    img_data['y_train'] = y_train
    img_data['X_test'] = X_test
    img_data['y_test'] = y_test
    with open(parent_dir+'/encoder_data.pkl', 'wb') as my_pickle:
        pickle.dump(img_data,my_pickle)

def load_data(perc=0.7):
    import os
    if(os.path.isfile(parent_dir+"/encoder_data.pkl") == False):
        import crop_std
        create_pickle()
    import pickle
    with open(parent_dir+"/encoder_data.pkl","rb") as img_f:
        data = pickle.load(img_f)
    return test_train_split(np.concatenate((data['X_train'],data['X_test']),axis=0),
                    np.concatenate((data['y_train'],data['y_test']),axis=0),perc)

def load_no_fish():
    import os
    import pickle
    if(os.path.isfile(parent_dir+"/no_fish_data.pkl") == False):
        from PIL import Image
        np.random.seed(5)
        print "Creating Pickle"
        crop_folders = ['NoF']
        X = []
        Y = []
        img_data = {}
        for (label,folder) in enumerate(crop_folders):
            orig_file_names = os.listdir(parent_dir+"/train/"+folder+"/")
            X += map(lambda x: parent_dir+"/train/"+folder+"/"+x, orig_file_names)
        #Creates Images from filepaths
        X = map(lambda f_name: np.asarray( Image.open(f_name).convert('L').resize((128,128)), dtype='uint8' ), X)
        X = np.reshape(np.asarray(X,dtype='uint8'), (len(X), 128 ,128, 1))
        with open(parent_dir+'/no_fish_data.pkl', 'wb') as my_pickle:
            pickle.dump(X, my_pickle)
        return X
    else:
        with open(parent_dir+"/no_fish_data.pkl","rb") as img_f:
            data = pickle.load(img_f)
    return data

def test_train_split(X_data,y_data,percentage):
    perm = np.random.permutation(np.arange(X_data.shape[0]))
    X_data = X_data[perm,:]
    y_data = y_data[perm,:]
    train_size = int(X_data.shape[0]*percentage)

    X_train = X_data[:train_size,:]
    y_train = y_data[:train_size,:]
    X_test = X_data[train_size:,:]
    y_test = y_data[train_size:,:]

    return (X_train,y_train,X_test,y_test)
