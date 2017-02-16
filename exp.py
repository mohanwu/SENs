import scipy.misc

img_loc = 'train/'

sample_img = scipy.misc.imread(img_loc + 'ALB/img_00003.jpg')

print sample_img[0][0]
print type(sample_img)
print sample_img.shape


