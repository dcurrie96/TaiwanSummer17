import scipy.misc
import tensorflow as tf
import skimage.transform
from skimage.io import imsave, imread
#import image_slicer
#from PIL import Image
#import shutil
import os
import math
#import subprocess

width = os.popen('file einstein.jpg | grep -o "[0-9][0-9]*[0-9]x[0-9]*[0-9][0-9]" | tail -1 | cut -d "x" -f 1').read()

height = os.popen('file einstein.jpg | grep -o "[0-9][0-9]*[0-9]x[0-9]*[0-9][0-9]" | tail -1 | cut -d "x" -f 2').read()

wideness = int(width)/224
heightness = int(height)/224
wit = math.ceil(wideness)
hit = math.ceil(heightness)

tile_num = wit*hit

tile_names=[]

for i in range(tile_num):
	
	tile_names.append('tile_' + str(i) + '.jpg')


Image = "einstein"

def resize_image(path):
    img = imread(path)
    img = skimage.transform.resize(img, (224*wit, 224*hit))
    # desaturate image
    #img = (img[:,:,0] + img[:,:,1] + img[:,:,2]) / 3.0
    scipy.misc.toimage(img).save('resized.jpg')
    os.system('convert resized.jpg -crop "224x224" -type TrueColor tile_%d.jpg')

def load_image(path):
    img = imread(path)
    # resize to 224, 224
    img = skimage.transform.resize(img, (224, 224))
    # desaturate image
    return (img[:,:,0] + img[:,:,1] + img[:,:,2]) / 3.0


resize_image('einstein.jpg')


with open("colorize.tfmodel", mode='rb') as f:
    fileContent = f.read()
    

#graph_def = tf.GraphDef()
#graph_def.ParseFromString(fileContent)


for image in tile_names:
    

    image_gray = load_image(image).reshape(1, 224, 224, 1)
    #with open("colorize.tfmodel", mode='rb') as f:
	    #fileContent = f.read()
    with tf.Graph().as_default(): 
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fileContent)
        grayscale = tf.placeholder(tf.float32, [1, 224, 224, 1])
        inferred_rgb, = tf.import_graph_def(graph_def, input_map={"grayscale": grayscale },
                                            return_elements=["inferred_rgb:0"])

        with tf.Session() as sess:
        
            inferred_batch = sess.run(inferred_rgb, feed_dict={grayscale: image_gray})
            imsave(image, inferred_batch[0])
            print ("done")

maxtile = tile_num - 1
cmd_combine = "montage tile_[0-"+str(maxtile)+"].jpg -tile "+str(wit)+"x"+str(hit)+" -geometry 224x finalimagecp.jpg"
os.system(cmd_combine)
#os.system('montage tile_[0-8].jpg -tile 2x2 -geometry 224x finalimagecp.jpg')
print('BAAAAAAAAAAAAAAAAM!')
