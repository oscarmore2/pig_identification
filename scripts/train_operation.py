
import cv2
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import time
from datetime import datetime
import os
import random as rand
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plot
#from utils.data_utils import *
from utils.vis_utils import *
from utils.layer_utils import *
from utils.print_utils import *
from utils.resnet import *
import pickle
import problem_unittests as tests
import helper
from glob import glob



#display_step = 1
learning_rate = 0.001
batch_size = 40
epochs = 30
#num_epochs = 50
num_residual_blocks =20
num_residual_blocks_loc = 4
train_ema_decay = 0.95
require_improvement = 1000
keep_probability = 0.9
keep_probability_loc1 = 0.8
keep_probability_loc2 = 0.8

best_validation_accuracy = 0.0

H, W, C = 64, 120, 3
image_shape = (H, W, C)
image_flat_shape = H * W * C
num_classes = 30

x = tf.placeholder(tf.float32, [None, H, W, C], name='x')
y = tf.placeholder(tf.int32, [None, num_classes], name='y')
#keep_prob_loc1 = tf.placeholder(dtype=tf.float32, name='keep_prob_loc1')
#keep_prob_loc2 = tf.placeholder(dtype=tf.float32, name='keep_prob_loc2')
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')


def BatchingData(batch_size, X_train, Y_train, trainBatchCount):
    """
    Process batching data
    """
    for i in range(trainBatchCount):
        start_t = i*batch_size
        X_t_batch = np.cast['float'](X_train[start_t:start_t+batch_size])
        Y_t_batch = np.cast['float'](Y_train[start_t:start_t+batch_size])
        
        yield X_t_batch, Y_t_batch, i

##Ready for the Data
def _process_and_save(normalize, features, labels, filename):
    """
    Preprocess data and save it to file
    """
    features = normalize(features)
    labels = labels
    f = open(filename, 'wb')
    pickle.dump((features, labels), f)
    f.close()

def preprocess_and_data(loadFiles, normalize):
    """
    Preprocess Training and Validation Data
    """
    n_batches = 5
    valid_features = []
    valid_labels = []
    features, labels, test_feature, test_labels, val_features, val_labels = loadFiles(True, sample_rate = 0.3)
    
    batchsize = int(features.shape[0] / 5)

    _process_and_save(normalize, features, labels, 'preprocess_batch_tranning_120.p')
    _process_and_save(normalize, test_feature, test_labels, 'preprocess_batch_testing_120.p')
    _process_and_save(normalize, val_features, val_labels, 'preprocess_batch_val_120.p')
    print('FInish Preprocessed')

    

#cap2 = cv2.VideoCapture('Pig_Identification_Qualification_Train/train/1.mp4')
def center_crop(img):
    shape = img.shape
    h = shape[0]
    w = shape[1]
    c = shape[2]
    if w > h:
        lbound = int((w - h) / 2)
        rbound = w - lbound
        return img[:, lbound:rbound, :]
    else:
        tbound = int((h - w) / 2)
        bbound = h - tbound
        return img[tbound:bbound, :, :]


def LoadVideo(path):
    cap = cv2.VideoCapture(path)
    nbFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    images = []

    for i in range(nbFrames):
        ret, frame = cap.read()
        #print(ret)
        if ret:
            images.append(frame[:,:,::-1]) #convert data from BGR color space to RGB
    cap.release()
    cv2.destroyAllWindows()
    return images, nbFrames

def CreateDataSet(video, labels, rate=0.8):
    video = np.array(video)
    print('video Shape:{}'.format(video.shape))
    ids = np.arange(video.shape[0])
    rand.shuffle(ids)
    trainingCount = int(len(video) * rate)
    testCount = len(video)-trainingCount
    trainingVideo = []
    testVideo = []
    trainingLabels = []
    testLabels = []
    #print(len(video))
    #print(range(trainingCount))
    #print(range(testCount))
    #print(video[0].shape)
    for i in range(trainingCount-1) :
        trainingVideo.append(video[ids[i]])
        trainingLabels.append(labels[ids[i]])
    for j in range(testCount-1):
        testVideo.append(video[ids[trainingCount+j]])
        testLabels.append(labels[ids[trainingCount+j]])
    return np.array(trainingVideo), np.array(trainingLabels), np.array(testVideo),  np.array(testLabels)

def SplitData(data, rate=0.8):
    cuttendDataCount = int(data.shape[0] * rate)
    group1 = [] 
    group2 = []
    for i in range(len(data)):
        if i < cuttendDataCount:
            group2.append(data[i])
        else:
            group1.append(data[i])
            
    return np.array(group1), np.array(group2)

def Spatial_Transfrom (inMap, theta, out_dims=None, **args):
    sess=tf.Session()
    print("inmap{},  shape{}, shapeList{}".format(inMap, tf.shape(inMap), inMap.get_shape().as_list()))
    shape = inMap.get_shape()
    B = tf.shape(inMap)[0]
    H = shape.as_list()[1]
    W = shape.as_list()[2]
    C = shape.as_list()[3]
    
    print('b:{}'.format(B))
    #print('w:{}, shape:{}'.format(w, w.get_shape().as_list()))
    
    #construct theta matrix
    theta = tf.reshape(theta, [B, 2, 3])
    
    if out_dims:
        out_H = out_dims[0]
        out_W = out_dims[1]
        batch_grids = affine_grid_gen(out_H, out_W, theta)
    else:
        batch_grids = affine_grid_gen(H, W, theta)
    
    print('batch_grids')
    print(batch_grids.get_shape().as_list())
    xs = batch_grids[:, 0, :, :]
    ys = batch_grids[:, 1, :, :]
    
    out_samp = bilinear_sampler(inMap, xs, ys)
    print('Spatial_Transfrom')
    print(inMap.get_shape().as_list())
    print(out_samp.get_shape().as_list())
    print('\n')
    return out_samp

def affine_grid_gen(height, width, theta):
    print('affine_grid_gen, height:{}, width:{}, theta:{}'.format(height, width, theta.get_shape().as_list()))
    
    num_batch = tf.shape(theta)[0]
    
    #nornalized grid elements
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x, y)
    print('affine_grid_gen, x:{}, y:{}, x_t:{}, y_t:{}'.format(x.get_shape().as_list(), y.get_shape().as_list(), 
                                                               x_t.get_shape().as_list(), y_t.get_shape().as_list()))
    
    #flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])
    print('affine_grid_gen, x_t_flat:{}, y_t_flat:{}'.format(x_t_flat.get_shape().as_list(), y_t_flat.get_shape().as_list()))
    
    #generate grid
    ones = tf.ones_like(x_t_flat)
    print('ones')
    print(ones.get_shape().as_list())
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))
    
    #cast to float32
    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')
    
    batch_grids = tf.matmul(theta, sampling_grid)
    
    batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])
    
    return batch_grids

def bilinear_sampler(img, x, y):
    print("bilinear_sampler x:{}, y:{}".format(str(x), str(y)))
    
    B = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    C = tf.shape(img)[3]
    
    max_y = tf.cast(H -1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')
    
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    
    x = 0.5 * ((x + 1.0) * tf.cast(W, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(H, 'float32'))
    
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)
    
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)
    
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    return out

def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W, )
    - y: flattened tensor of shape (B*H*W, )
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)

def Label_OneHot(count, y_val):
    y = []
    for i in range(count):
        y.append(np.eye(30, dtype='int32')[y_val])
    return y

#imgs, frameCount = LoadVideo('Pig_Identification_Qualification_Train/convert/{}.avi'.format(i))
def LoadData(show_grid=False, sample_rate = 1):
    imgSet = []
    lablesSet = []
    for i in range(30):
        imgs, frameCount = LoadVideo('./../Pig_Identification_Qualification_Train/convert/{}.avi'.format(i))
        rand.shuffle(imgs)
        lables = Label_OneHot(len(imgs), i)
        for j in range(int(len(imgs) * sample_rate)):
            imgSet.append(imgs[j])
            lablesSet.append(lables[j])
    
    X_train, Y_train, X_test, Y_test = CreateDataSet(imgSet, lablesSet)
    
    if show_grid:
        mask = np.arange(250)
        sample = np.reshape(X_train, [-1, H, W, C])[mask]
        view_images(sample)
        
    X_train = np.reshape(X_train, [-1, H, W, C])
    X_test = np.reshape(X_test, [-1, H, W, C])
    
    X_test, X_val = SplitData(X_test)
    Y_test, Y_val = SplitData(Y_test)

    return X_train, Y_train, X_test, Y_test, X_val, Y_val

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    # TODO: Implement Function
    a = 0
    b = 255
    return (x-a)/(b-a)

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    num = len(x)
    arr = np.zeros((num, 10))
    for i, xl in enumerate(x):
        arr[i][xl] = 1
    return arr

def SelectSetFromBatchSet(Batch_size, Data1, Data2):
    ids = np.arange(Data1.shape[0])
    rand.shuffle(ids)
    rand.shuffle(ids)
    X_ = []
    Y_ = []
    for i in range(batch_size):
        X_.append(Data1[ids[i]])
        Y_.append(Data2[ids[i]])
    return X_, Y_

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # TODO: Implement Function
    loss = session.run(cost, feed_dict={x:feature_batch, y:label_batch, keep_prob:1.0, keep_prob_loc:1.0})
    valid_acc = sess.run(accuracy, feed_dict={
                x: valid_features,
                y: valid_labels,
                keep_prob: 1.,
                keep_prob_loc1:1.,
                keep_prob_loc2:1.
    })
    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                loss,
                valid_acc))
    
def print_stats2(session, feature_batch, label_batch, val_x, val_y, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # TODO: Implement Function
    miner_val_x, miner_val_y = SelectSetFromBatchSet(feature_batch.shape[0] * 10, val_x, val_y)
    loss = session.run(cost, feed_dict={x:miner_val_x, y:miner_val_y, keep_prob:1.0,  keep_prob_loc1:1., keep_prob_loc2:1.})
    valid_acc = sess.run(accuracy, feed_dict={
                    x: miner_val_x,
                    y: miner_val_y,
                    keep_prob: 1.,
                    keep_prob_loc1:1.,
                    keep_prob_loc2:1.
    })
    
    print('Validation Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                    loss,
                    valid_acc))


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    print(logits.get_shape().as_list())
    print(labels.get_shape().as_list())
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean

def train_operation(global_step, total_loss):    
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(total_loss, global_step=global_step)
    return train_op

def load_preprocess_batch(path, batch_size, dataCount):
    f = open(path, mode='rb')
    features, labels = pickle.load(f)
    f.close()
    trainBatchCount = int(features.shape[0] / batch_size)
    dataCount[0] = trainBatchCount
    
    return BatchingData(batch_size, features, labels, trainBatchCount)


def file_name(file_dir):
    L=[]
    num=[]
    for root, dirs, files in os.walk(file_dir):
        #print(files)
        for file in files:
            if os.path.splitext(file)[1] == '.JPG':
                p = os.path.join(root, file)
                n = os.path.splitext(file)[0]
                yield p, n

def loadImgsAndSave():
    _x_ = []
    paths = []
    for path, num in file_name('../Pig_Identification_Qualification_Train/test_set'):
        _x = cv2.imread(path)[:,:,::-1]
        pImg = Image.fromarray(_x, mode='RGB')
        pImg = pImg.resize((64,64))
        _x_.append(np.array(pImg))
        paths.append(num)
        #_y_.append(Label_OneHot(1, rand.randint(1, 30)))
    result = {'data':np.array(_x_), 'path':paths}
    f = open(filename, 'wb')
    pickle.dump((result), open('test_set.p', 'wb'))
    f.close()

def loadImgs():
    f = open('test_set.p', mode='rb')
    result = pickle.load(f)
    f.close()
    return result

#preprocess_and_data(LoadData, normalize)
#loadImgsAndSave()
#xx = loadImgs()
#print(xx['data'].shape)


global_step = tf.Variable(0, trainable=False)
validation_step = tf.Variable(0, trainable=False)

logit_loc2 = inference(x, num_residual_blocks_loc, dropout=0.9, reuse=False, num_class=6)
h_trans2 = Spatial_Transfrom(x, logit_loc2, out_dim=[H, H])
logits = inference(x, num_residual_blocks, dropout=keep_prob, reuse=False, num_class=num_classes)
logits = tf.identity(logits, name='logits')

regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
train_loss = loss(logits, y)

full_loss = tf.add_n([train_loss] + regu_losses)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

train_op = train_operation(global_step, full_loss)

saver = tf.train.Saver(tf.global_variables())
summary_op = tf.summary.merge_all()

X_val, Y_val = pickle.load(open('./preprocess_batch_val.p', mode='rb'))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("./notran_small2",sess.graph)  

    for i in range(epochs):
        epoch = i+1
        start_time = time.time()
        batch_i = 0
        dataCount = [0]
        for x_t, y_t, batch_i in load_preprocess_batch('./preprocess_batch_tranning.p', batch_size, dataCount):
            if batch_i == epoch:
                view_images(x_t)
                print(y_t)

            _, train_loss_value, = sess.run([train_op, full_loss], {x:x_t, y:y_t, keep_prob:keep_probability})

            #thetas = sess.run(h_trans2, feed_dict={x:x_t, keep_prob_loc1:1, keep_prob_loc2:1})

            if batch_i % int(dataCount[0] / 100) == 0:
                print('Trainning batch {}/{} in epoch {}, local loss is {:.4f}'.format(batch_i, dataCount[0], epoch+1, train_loss_value))

            if batch_i >= dataCount[0]:
                summary_str = sess.run(summary_op, feed_dict={x:x_t, y:y_t, keep_prob:keep_probability})
                writer.add_summary(summary_str, batch_i)

        print('Epoch {:>2}'.format(epoch))
        print_stats2(sess, x_t, y_t, X_val, Y_val, full_loss, accuracy)
        #view_images(thetas[0:9])

        duration = time.time() - start_time
        examples_per_sec = batch_size / duration

        format_str = ('epoch Summary: {}: loss = {:.4f} ({:.1f} examples/sec; {:.3f} ' 'sec/batch)')
        print (format_str.format(datetime.now(), train_loss_value, examples_per_sec, duration))
        print ('----------------------------')

    save_model_path = './Pigs_identity_noTran_small2'
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)

    test_total_acc = 0

    for feature, labels, batch_i in  load_preprocess_batch('./preprocess_batch_testing.p', batch_size):
        test_total_acc += sess.run(accuracy, feed_dict={x:feature, y:labels, keep_prob:1.})
    print("Testing Accuracy: {}\n".format(test_total_acc/batch_i))

    path__ = []
    ww = rand.randint(1, 25)
    for _x_, _y_, batch_i in BatchingData(batch_size, xx['data'], np.array(Label_OneHot(3000, ww)), xx['data'].shape[0]):
        __x = normalize(_x_)
        predictions  = sess.run(loaded_logits, feed_dict={x:__x, keep_prob:1.})
        random_test_predictions  = sess.run(tf.nn.top_k(tf.nn.softmax(loaded_logits), 3), feed_dict={x:__x,  keep_prob:1.})
        
        label_binarizer = LabelBinarizer()
        label_binarizer.fit(range(30))
        label_ids = label_binarizer.inverse_transform(predictions)
        
        print('label_ids shape{},test_predictions shape {}'.format(label_ids.shape, random_test_predictions.values.shape))
        
        text_=''
        for idx in range(batch_size):
            text_ += 'filename : {}, Label: {}, Prediction: {}'.format(xx['path'][idx + (batch_i * batch_size)], 
                                       label_ids[idx], random_test_predictions.values[idx])
            text_ += '\n'

        #print(text_)
        text += text_
        
        if os.path.exists('./output1.csv'):
            f2 = open('./output1.csv', 'ab')
        else:
            f2 = open('./output1.csv', 'wb')
        f2.write(text+'\n')
        path__ = []
    f2.close()
    