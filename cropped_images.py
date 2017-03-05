def create_pickle():
    import os
    import numpy as np
    from PIL import Image
    import pickle

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

    X = map(lambda f_name: np.asarray( Image.open(f_name), dtype='uint8' ), X)
    img_data['X'] = np.asarray(X,dtype='uint8')
    img_data['Y'] = np.asarray(Y,dtype='uint8')
    with open('test_img_data.pkl', 'wb') as my_pickle:
        pickle.dump(img_data,my_pickle)


def load_data():
    import os
    if(os.path.isfile(os.getcwd()+"/test_img_data.pkl") == False):
        create_pickle()
    import pickle
    with open("test_img_data.pkl","rb") as img_f:
        data = pickle.load(img_f)
    return (data['X'],data['Y'])
