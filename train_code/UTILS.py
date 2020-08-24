import numpy as np
import tensorflow as tf
import math, os, random, time
from scipy.linalg import hadamard
import scipy.misc
from PIL import Image
from TRAIN import BATCH_SIZE
from TRAIN import PATCH_SIZE
import struct
import re

import time , threading
from UTILS import *

class Reader (threading.Thread):
    def __init__(self,file_name,id,input_list,gt_list):
        super(Reader,self).__init__()
        self.file_name=file_name
        self.id = id
        self.input_list=input_list
        self.gt_list=gt_list
    def run(self):
        input_image = c_getYdata(self.file_name[0])
        gt_image = c_getYdata(self.file_name[1])
        #print(input_image[0])
        in_ =[]
        gt_ =[]
        for j in range(BATCH_SIZE//8):
            input_imgY, gt_imgY = crop(input_image, gt_image, PATCH_SIZE[0], PATCH_SIZE[1], "ndarray")
            input_imgY = normalize(input_imgY)
            gt_imgY = normalize(gt_imgY)

            in_.append(input_imgY)
            gt_.append(gt_imgY)

        #print(np.shape(in_))
        self.input_list[self.id] = in_
        self.gt_list[self.id] = gt_



def normalize(x):
    x = x / 255.
    return truncate(x, 0., 1.)

def denormalize(x):
    x = x * 255.
    return truncate(x, 0., 255.)

def truncate(input, min, max):
    input = np.where(input > min, input, min)
    input = np.where(input < max, input, max)
    return input

def remap(input):
    input = 16+219/255*input
    #return tf.clip_by_value(input, 16, 235).eval()
    return truncate(input, 16.0, 235.0)

def deremap(input):
    input = (input-16)*255/219
    #return tf.clip_by_value(input, 0, 255).eval()
    return truncate(input, 0.0, 255.0)

def load_file_list(directory):
    list = []
    # for filename in [y for y in os.listdir(directory) if os.path.isfile(os.path.join(directory,y))]:
    #     print(filename)
    #     if filename.split(".")[-1]=="yuv":
    #         list.append(os.path.join(directory,filename))
    for root,dirs,files in os.walk(directory):
        for file in files:
            file_name =os.path.join(root,file)
            if file_name.split(".")[-1]=="yuv":
                list.append(file_name)
    return sorted(list)

def get_train_list(lowList, highList):
    assert len(lowList) % len(highList)==0, "low:%d, high:%d"%(len(lowList), len(highList))
    train_list = []
    for i in range(len(lowList)):
        # print(lowList[i])
        qp=lowList[i].split("\\")[-2].split("qp")[1]
        if 37<=int(qp)<=46:
            train_list.append([lowList[i], highList[i%len(highList)]])
    return train_list
def get_train_list1(lowList, highList):
    # assert len(lowList) == len(highList), "low:%d, high:%d"%(len(lowList), len(highList))
    train_list = []
    for i in range(len(lowList)):
        # print(lowList[i],type(lowList[i]))
        #idx=int(lowList[i].split("\\")[-1].split("_")[-2])-1
        # qp = int(lowList[i].split("\\")[-2].split("q")[1])
        # print(lowList[i], idx,qp)
        # if(qp==53):
        # if(lowList[i].split("_")[-3]==highList[i].split("_")[-3]):
        train_list.append([lowList[i], highList[i%len(highList)]])
        # else:
        #     print(lowList[i],highList[i])
    return train_list
#
# def prepare_nn_data(train_list):
#     batchSizeRandomList = random.sample(range(0,len(train_list)), 8)
#     #print(batchSizeRandomList)
#     input_list = []
#     gt_list = []
#
#     for i in batchSizeRandomList:
#         #print(train_list[i][0])
#         input_image = c_getYdata(train_list[i][0])
#         gt_image = c_getYdata(train_list[i][1])
#
#         for j in range(BATCH_SIZE//8):
#             input_imgY, gt_imgY = crop(input_image, gt_image, PATCH_SIZE[0], PATCH_SIZE[1], "ndarray")
#
#             input_imgY = normalize(input_imgY)
#             gt_imgY = normalize(gt_imgY)
#
#             input_list.append(input_imgY)
#             gt_list.append(gt_imgY)
#
#     input_list = np.reshape(input_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
#     gt_list = np.reshape(gt_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
#
#     return input_list, gt_list

def prepare_nn_data(train_list):
    thread_num =8
    #random_init = random.randint(0, len(train_list)-thread_num)
    batchSizeRandomList = random.sample(range(0,len(train_list)), thread_num)
    #
    #print(batchSizeRandomList)
    input_list = [0 for i in range(thread_num)]
    gt_list = [0 for i in range(thread_num)]
    t = []
    for i in range(thread_num):
        t.append(Reader(train_list[batchSizeRandomList[i]],i,input_list,gt_list))
        #t.append(Reader(train_list[random_init+i], i, input_list, gt_list))
    for i in range(thread_num):
        t[i].start()
    for i in range(thread_num):
        t[i].join()
    #print(input_list)
    input_list = np.reshape(input_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
    gt_list = np.reshape(gt_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))

    return input_list, gt_list

'''
def prepare_nn_data(train_list, idx_img=None):
    i = np.random.randint(len(train_list)) if (idx_img is None) else idx_img
    input_image  = c_getYdata(train_list[i][0])
    gt_image = c_getYdata(train_list[i][1])


    input_list = []
    gt_list = []
    inputcbcr_list = []

    for idx in range(BATCH_SIZE):

        #crop images to the disired size.
        input_imgY, gt_imgY = crop(input_image, gt_image, PATCH_SIZE[0], PATCH_SIZE[1], "ndarray")

        #input_imgY = Greying(input_imgY, PATCH_SIZE[0])
        #scipy.misc.imsave('/home/chenjs/relHM/tFromPython/%s.png'%(time.time()), input_imgY)
        #scipy.misc.imsave('/home/chenjs/relHM/tFromPython/%s.png'%(time.time()), gt_imgY)

        #normalize
        input_imgY = normalize(input_imgY)
        gt_imgY = normalize(gt_imgY)

        input_list.append(input_imgY)
        gt_list.append(gt_imgY)
        #inputcbcr_list.append(input_imgCbCr)

    input_list = np.reshape(input_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
    gt_list = np.reshape(gt_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
    #inputcbcr_list = np.resize(inputcbcr_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 2))

    return input_list, gt_list, inputcbcr_list
'''
def getLdata(path):
    img = Image.open(path)
    return  np.asarray(img, dtype='uint8')

def Greying(block, patch_size):
    newBlock = [[0 for i in range(patch_size)] for j in range(patch_size)]
    for i in range(patch_size):
        for j in range(patch_size):
            if (i >= patch_size // 2 and j >= patch_size // 2):
                newBlock[i][j] = 128
                continue
            newBlock[i][j] = block[i][j]
    return np.asarray(newBlock)

def getWH1(yuvfileName):   # Test
    w_included , h_included = os.path.splitext(os.path.basename(yuvfileName))[0].split('x')
    w = w_included.split('_')[-1]
    h = h_included.split('_')[0]
    return int(w), int(h)

def getWH(yuvfileName):   # Train
    #print(yuvfileName)
    deyuv=re.compile(r'(.+?)\.')
    deyuvFilename=deyuv.findall(yuvfileName)[0] #去yuv后缀的文件名
    # print(deyuvFilename)
    if os.path.basename(deyuvFilename).split("_")[0].isdigit():
        wxh = os.path.basename(deyuvFilename).split('_')[1]
    else:
        wxh = os.path.basename(deyuvFilename).split('_')[1]
    w, h = wxh.split('x')
   # print(deyuvFilename,w,h)
    return int(w), int(h)

def getYdata(path, size):
    w= size[0]
    h=size[1]
    #print(w,h)
    Yt = np.zeros([h, w], dtype="uint8", order='C')
    with open(path, 'rb') as fp:
        fp.seek(0, 0)
        Yt = fp.read()
        tem = Image.frombytes('L', [w, h], Yt)

        Yt = np.asarray(tem, dtype='float32')

        # for n in range(h):
        #     for m in range(w):
        #         Yt[n, m] = ord(fp.read(1))

    return Yt

def c_getYdata(path):
    #print(path)
    return getYdata(path, getWH(path))

def calcHADLoss(ori, cur):
    ori = tf.squeeze(ori)
    cur = tf.squeeze(cur)
    resi = tf.subtract(ori, cur)

    transfMatrix = hadamard(8, dtype='float32')

    assert resi.shape[0] == BATCH_SIZE
    resiList = [resi[a] for a in range(BATCH_SIZE)]

    totalSATD = 0
    for i in range(BATCH_SIZE):
        resiPatch = tf.reshape(resiList[i], [1024])

        for col in range(4):
            for row in range(4):
                start = col*8*PATCH_SIZE[0]+row*8
                qt = [[0 for i in range(8)] for j in range(8)]
                for i in range(8):
                    for j in range(8):
                        qt[i][j] = resiPatch[start + i * PATCH_SIZE[0] + j]
                transfromed = tf.matmul(transfMatrix, qt)
                transfromed = tf.matmul(transfromed, transfMatrix)

                totalSATD += tf.reduce_sum(tf.abs(transfromed))

    return totalSATD

def img2y(input_img):
    if np.asarray(input_img).shape[2] == 3:
        input_imgY = input_img.convert('YCbCr').split()[0]
        input_imgCb, input_imgCr = input_img.convert('YCbCr').split()[1:3]

        input_imgY = np.asarray(input_imgY, dtype='float32')
        input_imgCb = np.asarray(input_imgCb, dtype='float32')
        input_imgCr = np.asarray(input_imgCr, dtype='float32')


        #Concatenate Cb, Cr components for easy, they are used in pair anyway.
        input_imgCb = np.expand_dims(input_imgCb,2)
        input_imgCr = np.expand_dims(input_imgCr,2)
        input_imgCbCr = np.concatenate((input_imgCb, input_imgCr), axis=2)

    elif np.asarray(input_img).shape[2] == 1:
        print("This image has one channal only.")
        #If the num of channal is 1, remain.
        input_imgY = input_img
        input_imgCbCr = None
    else:
        print("The num of channal is neither 3 nor 1.")
        exit()
    return input_imgY, input_imgCbCr

def crop(input_image, gt_image, patch_width, patch_height, img_type):
    assert type(input_image) == type(gt_image), "types are different."
    #return a ndarray object
    if img_type == "ndarray":
        in_row_ind   = random.randint(0,input_image.shape[0]-patch_width)
        in_col_ind   = random.randint(0,input_image.shape[1]-patch_height)

        input_cropped = input_image[in_row_ind:in_row_ind+patch_width, in_col_ind:in_col_ind+patch_height]
        gt_cropped = gt_image[in_row_ind:in_row_ind+patch_width, in_col_ind:in_col_ind+patch_height]

    #return an "Image" object
    elif img_type == "Image":
        in_row_ind   = random.randint(0,input_image.size[0]-patch_width)
        in_col_ind   = random.randint(0,input_image.size[1]-patch_height)

        input_cropped = input_image.crop(box=(in_row_ind, in_col_ind, in_row_ind+patch_width, in_col_ind+patch_height))
        gt_cropped = gt_image.crop(box=(in_row_ind, in_col_ind, in_row_ind+patch_width, in_col_ind+patch_height))

    return input_cropped, gt_cropped

def save_images(inputY, inputCbCr, size, image_path):
    """Save mutiple images into one single image.

    Parameters
    -----------
    images : numpy array [batch, w, h, c]
    size : list of two int, row and column number.
        number of images should be equal or less than size[0] * size[1]
    image_path : string.

    Examples
    ---------
    >>> images = np.random.rand(64, 100, 100, 3)
    >>> tl.visualize.save_images(images, [8, 8], 'temp.png')
    """
    def merge(images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:j*h+h, i*w:i*w+w, :] = image
        return img

    inputY = inputY.astype('uint8')
    inputCbCr = inputCbCr.astype('uint8')
    output_concat = np.concatenate((inputY, inputCbCr), axis=3)

    assert len(output_concat) <= size[0] * size[1], "number of images should be equal or less than size[0] * size[1] {}".format(len(output_concat))

    new_output = merge(output_concat, size)

    new_output = new_output.astype('uint8')

    img = Image.fromarray(new_output, mode='YCbCr')
    img = img.convert('RGB')
    img.save(image_path)

def get_image_batch(train_list,offset,batch_size):
    target_list = train_list[offset:offset+batch_size]
    input_list = []
    gt_list = []
    inputcbcr_list = []
    for pair in target_list:
        input_img = Image.open(pair[0])
        gt_img = Image.open(pair[1])

        #crop images to the disired size.
        input_img, gt_img = crop(input_img, gt_img, PATCH_SIZE[0], PATCH_SIZE[1], "Image")

        #focus on Y channal only
        input_imgY, input_imgCbCr = img2y(input_img)
        gt_imgY, gt_imgCbCr = img2y(gt_img)

        #input_imgY = normalize(input_imgY)
        #gt_imgY = normalize(gt_imgY)

        input_list.append(input_imgY)
        gt_list.append(gt_imgY)
        inputcbcr_list.append(input_imgCbCr)

    input_list = np.resize(input_list, (batch_size, PATCH_SIZE[0], PATCH_SIZE[1], 1))
    gt_list = np.resize(gt_list, (batch_size, PATCH_SIZE[0], PATCH_SIZE[1], 1))

    return input_list, gt_list, inputcbcr_list

def save_test_img(inputY, inputCbCr, path):
    assert len(inputY.shape) == 4, "the tensor Y's shape is %s"%inputY.shape
    assert inputY.shape[0] == 1, "the fitst component must be 1, has not been completed otherwise.{}".format(inputY.shape)

    inputY = np.squeeze(inputY, axis=0)
    inputY = inputY.astype('uint8')

    inputCbCr = inputCbCr.astype('uint8')

    output_concat = np.concatenate((inputY, inputCbCr), axis=2)
    img = Image.fromarray(output_concat, mode='YCbCr')
    img = img.convert('RGB')
    img.save(path)

def psnr(hr_image, sr_image, max_value=255.0):
    eps = 1e-10
    if((type(hr_image)==type(np.array([]))) or (type(hr_image)==type([]))):
        hr_image_data = np.asarray(hr_image, 'float32')
        sr_image_data = np.asarray(sr_image, 'float32')

        diff = sr_image_data - hr_image_data
        mse = np.mean(diff*diff)
        mse = np.maximum(eps, mse)
        return float(10*math.log10(max_value*max_value/mse))
    else:
        assert len(hr_image.shape)==4 and len(sr_image.shape)==4
        diff = hr_image - sr_image
        mse = tf.reduce_mean(tf.square(diff))
        mse = tf.maximum(mse, eps)
        return 10*tf.log(max_value*max_value/mse)/math.log(10)


# 将filename对应的图片文件读入，并缩放到统一的大小
def _parse_function(filename, label):
  # 读取图像文件内容编码为字符串
  image_contents = tf.read_file(filename)
  # 根据图像编码后的字符串解码为uint8的tensor
  image_decoded = tf.image.decode_image(image_contents)
  # 修改图像尺寸
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label



def divide_img(img, block_size, pading_size):
    h = img.shape[1]
    w = img.shape[2]
    boh = math.ceil(h / block_size)
    bow = math.ceil(w / block_size)
    sub_imgs = []
    for i in range(boh):
        for j in range(bow):
            if i != boh - 1 and j != bow - 1:
                if i == 0 and j == 0:
                    sub_img = img[:, 0:block_size + pading_size, 0:block_size + pading_size, :]
                elif i == 0 and j != 0:
                    sub_img = img[:, 0:block_size + pading_size,
                              block_size * (j) - pading_size:block_size * (j + 1) + pading_size, :]
                elif i != 0 and j == 0:
                    sub_img = img[:, block_size * i - pading_size:block_size * (i + 1) + pading_size,
                              0:block_size + pading_size, :]
                else:
                    sub_img = img[:, block_size * i - pading_size:block_size * (i + 1) + pading_size,
                              block_size * (j) - pading_size:block_size * (j + 1) + pading_size, :]
            elif i == boh - 1 and j != bow - 1:
                if j != 0:
                    sub_img = img[:, -(block_size + pading_size):,
                              block_size * (j) - pading_size:block_size * (j + 1) + pading_size, :]
                else:
                    sub_img = img[:, -(block_size + pading_size):,
                              0:block_size * (j + 1) + pading_size, :]
            elif i != boh - 1 and j == bow - 1:
                if i != 0:
                    sub_img = img[:, block_size * i - pading_size:block_size * (i + 1) + pading_size,
                              -(block_size + pading_size):, :]
                else:
                    sub_img = img[:, 0:block_size * (i + 1) + pading_size,
                              -(block_size + pading_size):, :]
            else:
                sub_img = img[:, -(block_size + pading_size):,
                          -(block_size + pading_size):, :]

            sub_imgs.append(sub_img)

    #         plt.subplot(boh, bow, i  * bow + j+1 )
    #         #plt.title("subImg : "+ str(j)+" "+str(i))
    #         plt.axis("off")
    #         plt.imshow(sub_img[0,:,:,0],cmap='gray')
    # plt.show()
    return sub_imgs


def compose_img(img, sub_imgs, block_size, pading_size):
    h = img.shape[1]
    w = img.shape[2]
    rec = np.zeros([h, w], int)
    boh = math.ceil(h / block_size)
    bow = math.ceil(w / block_size)
    #plt.figure(figsize=(24, 7))
    for i in range(boh):
        for j in range(bow):
            if i != boh - 1 and j != bow - 1:
                if i == 0 and j == 0:
                    rec[0:block_size, 0:block_size] = np.mat(sub_imgs[ i  * bow + j])[0:block_size, 0:block_size]
                elif i == 0 and j != 0:
                    rec[ :block_size, block_size * (j):block_size * (j + 1)] = np.mat(sub_imgs[ i  * bow + j])[
                                                                                    :block_size,
                                                                                    pading_size:block_size + pading_size]
                elif i != 0 and j == 0:
                    rec[ block_size * i:block_size * (i + 1), :block_size ] = np.mat(sub_imgs[
                                                                                                     i  * bow + j])[
                                                                                                pading_size:block_size + pading_size,
                                                                                                :block_size]
                else:
                    rec[ block_size * i:block_size * (i + 1), block_size * (j):block_size * (j + 1)] = np.mat(sub_imgs[ i  * bow + j])[
                                                                                                            pading_size:block_size + pading_size,
                                                                                                            pading_size:block_size + pading_size]
            elif i == boh - 1 and j != bow - 1:
                if j != 0:
                    rec[ -block_size:, block_size * (j):block_size * (j + 1)] = np.mat(sub_imgs[ i  * bow + j])[
                                                                                     pading_size:block_size + pading_size,
                                                                                     pading_size:block_size + pading_size]
                else:
                    rec[ -block_size:, 0:block_size * (j + 1)] = np.mat(sub_imgs[ i  * bow + j])[
                                                                      pading_size:block_size + pading_size,
                                                                      0:block_size]
            elif i != boh - 1 and j == bow - 1:
                if i != 0:
                    rec[ block_size * i:block_size * (i + 1), - block_size:] = np.mat(sub_imgs[ i  * bow + j])[
                                                                                     pading_size:block_size + pading_size,
                                                                                     pading_size:block_size + pading_size]
                else:
                    rec[ 0:block_size * (i + 1) , -block_size :] = np.mat(sub_imgs[ i  * bow + j])[
                                                                                                    0:block_size,
                                                                                                    pading_size:block_size + pading_size]
            else:
                rec[ -block_size: , -block_size :] = np.mat(sub_imgs[ i  * bow + j])[
                                                                                        -block_size: , -block_size :]
    return rec


def saveImg(file,inp):

    h, w = inp[0], inp[1]
    # tem = np.asarray(inp, dtype='uint8')
    # #np.save(r"H:\KONG\cnn_2K%f" % time.time(),tem)
    # tem = Image.fromarray(tem, 'L')
    # tem.save("D:/rec/FromPython%f.jpg" % time.time())
    with open(file,'wb') as fp:
        for line in inp:
            for i in line:
                #print(i)
                fp.write(struct.pack('B',i))

    print('image saved')