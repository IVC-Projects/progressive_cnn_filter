import sys

if not hasattr(sys, 'argv'):
    sys.argv = ['']

from CNN import model as model
from UTILS import *
# from cnn_n import saveImg
from shutil import copyfile

tplt1 = "{0:^30}\t{1:^10}\t{2:^10}\t{3:^10}\t{4:^10}"  #\t{4:^10}\t{5:^10}
tplt2 = "{0:^30}\t{1:^10}\t{2:^10}"

model_set = {
    '''所用模型'''
    "CNN2_I_QP37":r"progressive_cnn_filter\models\firstCNN\HEVC\QP37\CNN2\CNN2_I_QP37.ckpt",}


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction =1
config.gpu_options.allow_growth = True

global cnn17, cnn27, cnn37, cnn47, cnn7, cnn57


def prepare_test_data(fileOrDir):
    original_ycbcr = []
    imgCbCr = []
    gt_y = []
    fileName_list = []
    # The input is a single file.
    if type(fileOrDir) is str:
        fileName_list.append(fileOrDir)

        # w, h = getWH(fileOrDir)
        # imgY = getYdata(fileOrDir, [w, h])
        imgY = c_getYdata(fileOrDir)
        imgY = normalize(imgY)

        imgY = np.resize(imgY, (1, imgY.shape[0], imgY.shape[1], 1))
        original_ycbcr.append([imgY, imgCbCr])

    ##The input is one directory of test images.
    elif len(fileOrDir) == 1:
        fileName_list = load_file_list(fileOrDir)
        for path in fileName_list:
            # w, h = getWH(path)
            # imgY = getYdata(path, [w, h])
            imgY = c_getYdata(path)
            imgY = normalize(imgY)

            imgY = np.resize(imgY, (1, imgY.shape[0], imgY.shape[1], 1))
            original_ycbcr.append([imgY, imgCbCr])

    ##The input is two directories, including ground truth.
    elif len(fileOrDir) == 2:

        fileName_list = load_file_list(fileOrDir[0])
        test_list = get_train_list(load_file_list(fileOrDir[0]), load_file_list(fileOrDir[1]))
        for pair in test_list:
            filesize = os.path.getsize(pair[0])
            picsize = getWH(pair[0])[0]*getWH(pair[0])[0] * 3 // 2
            numFrames = filesize // picsize
            # if numFrames ==1:
            or_imgY = c_getYdata(pair[0])
            gt_imgY = c_getYdata(pair[1])

            # normalize
            or_imgY = normalize(or_imgY)

            or_imgY = np.resize(or_imgY, (1, or_imgY.shape[0], or_imgY.shape[1], 1))
            gt_imgY = np.resize(gt_imgY, (1, gt_imgY.shape[0], gt_imgY.shape[1], 1))

            ## act as a placeholder
            or_imgCbCr = 0
            original_ycbcr.append([or_imgY, or_imgCbCr])
            gt_y.append(gt_imgY)
            # else:
            #     while numFrames>0:
            #         or_imgY =getOneFrameY(pair[0])
            #         gt_imgY =getOneFrameY(pair[1])
            #         # normalize
            #         or_imgY = normalize(or_imgY)
            #
            #         or_imgY = np.resize(or_imgY, (1, or_imgY.shape[0], or_imgY.shape[1], 1))
            #         gt_imgY = np.resize(gt_imgY, (1, gt_imgY.shape[0], gt_imgY.shape[1], 1))
            #
            #         ## act as a placeholder
            #         or_imgCbCr = 0
            #         original_ycbcr.append([or_imgY, or_imgCbCr])
            #         gt_y.append(gt_imgY)
    else:
        print("Invalid Inputs.")
        exit(0)

    return original_ycbcr, gt_y, fileName_list


class Predict:
    input_tensor = None
    output_tensor = None
    model = None

    def __init__(self, model, modelpath):
        self.graph = tf.Graph()  # 为每个类(实例)单独创建一个graph
        self.model = model
        with self.graph.as_default():
            self.input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
            #self.output_tensor = tf.make_template('input_scope', self.model)(self.input_tensor)
            self.output_tensor = model(self.input_tensor)
            self.output_tensor = tf.clip_by_value(self.output_tensor, 0., 1.)
            self.output_tensor = tf.multiply(self.output_tensor, 255)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph,config=config)  # 创建新的sess
        with self.sess.as_default():
            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())
                self.saver.restore(self.sess, modelpath)  # 从恢复点恢复参数
                print(modelpath)

    def predict(self, fileOrDir):
        #print("------------")
        if (isinstance(fileOrDir, str)):
            original_ycbcr, gt_y, fileName_list = prepare_test_data(fileOrDir)
            imgY = original_ycbcr[0][0]

        elif type(fileOrDir) is np.ndarray:
            imgY = fileOrDir
            imgY = normalize(np.reshape(fileOrDir, (1, len(fileOrDir), len(fileOrDir[0]), 1)))

        elif (isinstance(fileOrDir, list)):
            fileOrDir = np.asarray(fileOrDir, dtype='float32')
            # fileOrDir = fileOrDir / 255
            imgY = normalize(np.reshape(fileOrDir, (1, len(fileOrDir), len(fileOrDir[0]), 1)))


        else:
            imgY=None

        with self.sess.as_default():
            with self.sess.graph.as_default():
                out = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: imgY})
                out = np.reshape(out, (out.shape[1], out.shape[2]))
                out = np.around(out)
                out = out.astype('int')
                out = out.tolist()
                return out


def init(sliceType, QP):
    # print("init !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    global cnn17,cnn27,cnn37,cnn47,cnn7,cnn57

    cnn7 =Predict(model,model_set["CNN_I_QP7"])
    cnn17=Predict(model,model_set["CNN_I_QP17"])
    cnn27=Predict(model,model_set["CNN_I_QP27"])
    cnn37=Predict(model,model_set["CNN2_I_QP37"])
    cnn47=Predict(model,model_set["CNN_I_QP47"])
    cnn57 = Predict(model, model_set["CNN_I_QP57"])







def predict(file, QP, frame_type):
    global cnn17, cnn27, cnn37, cnn47, cnn7, cnn57
    if QP < 17:
        R = cnn7.predict(file)
    elif 17 <= QP < 27:
        R = cnn17.predict(file)
    elif 27 <= QP < 37:
        R = cnn27.predict(file)
    elif 37 <= QP < 47:
        R = cnn37.predict(file)
    elif 47 <= QP < 57:
        R = cnn47.predict(file)
    else:
        R = cnn57.predict(file)

def showImg(inp):

    h, w = inp[0], inp[1]
    tem = np.asarray(inp, dtype='uint8')
    #np.save(r"H:\KONG\cnn_2K%f" % time.time(),tem)
    tem = Image.fromarray(tem, 'L')
    tem.show()
    #tem.save("D:/rec/FromPython%f.jpg" % time.time())

def test_all_ckpt(modelPath):
    low_img = r"test_set\qp37"
    # low_img = r"H:\KONG\FRAME_!\AV!_deblock_nocdefLr\QP53"
    heigh_img = r"test_set\ori"

    NUM_CNN=3 #cnn 次数
    original_ycbcr, gt_y, fileName_list = prepare_test_data([low_img,heigh_img])
    total_imgs = len(fileName_list)

    tem = [f for f in os.listdir(modelPath) if 'data' in f]
    ckptFiles = sorted([r.split('.data')[0] for r in tem])
    max_psnr=0
    max_epoch=0
    max_ckpt_psnr = 0



    for ckpt in ckptFiles:
        cur_ckpt_psnr=0
        #epoch = int(ckpt.split('_')[3])
        # print(ckpt.split('.')[0].split('_'))
        epoch = int(ckpt.split('.')[0].split('_')[-2])
        # loss =int(ckpt.split('.')[0].split('_')[-1])
        #
        # if  epoch  <1000:
        #    continue
        #print(epoch)
        print(os.path.join(modelPath, ckpt))
        predictor = Predict(model, os.path.join(modelPath, ckpt))

        img_index = [14, 17, 4, 2, 7, 10, 12, 3, 0, 13, 16, 5, 6, 1, 15, 8, 9, 11]
        for i in img_index:
            # if i>5:
            #    continue
            imgY = original_ycbcr[i][0]
            gtY = gt_y[i] if gt_y else 0


            #showImg(rec)
            #print(np.shape(np.reshape(imgY, [np.shape(imgY)[1],np.shape(imgY)[2]])))
            #cur_psnr[cnnTime]=psnr(denormalize(np.reshape(imgY, [np.shape(imgY)[1],np.shape(imgY)[2]])),np.reshape(gtY, [np.shape(imgY)[1],np.shape(imgY)[2]]))
            cur_psnr=[]
            rec = predictor.predict(imgY)
            cur_psnr.append(psnr(rec,np.reshape(gtY, np.shape(rec))))
            for cc in range(2,NUM_CNN+1):
                rec = predictor.predict(rec)
                cur_psnr.append(psnr(rec, np.reshape(gtY, np.shape(rec))))

            # print(cur_psnr)


            #print(len(cur_psnr))
            cur_ckpt_psnr=cur_ckpt_psnr+np.mean(cur_psnr)
            # print(tplt2.format(os.path.basename(fileName_list[i]), cur_psnr,psnr(denormalize(np.reshape(imgY, np.shape(rec))),np.reshape(gtY, np.shape(rec)))))
            print("%30s"%os.path.basename(fileName_list[i]),end="")
            for cc in cur_psnr:
                print("       %.5f"%cc,end="")
            print("       %.5f" % np.mean(cur_psnr), end="")
            print("       %.5f"%psnr(denormalize(np.reshape(imgY, np.shape(rec))),np.reshape(gtY, np.shape(rec))))

        if(cur_ckpt_psnr/total_imgs>max_ckpt_psnr):
            max_ckpt_psnr=cur_ckpt_psnr/total_imgs
            max_epoch=epoch
        print("______________________________________________________________")
        print(epoch,cur_ckpt_psnr/total_imgs,max_epoch,max_ckpt_psnr)


def build_trainset():
    global mModel
    reconDir = r"train_set\hevc_div2k_train_noFilter_yuv"
    saveDir=r"train_set\hevc_div2k_train_noFilter_yuv_cnn"  #cnn1 cnn2  cnn3

    NUM_CNN = 3   #cnn次数
    for i in range(1,NUM_CNN+1):
        if os.path.exists(saveDir+str(i))==0:
            os.mkdir(saveDir+str(i))
    mModel = Predict(model,
                   r"progressive_cnn_filter\models\firstCNN\HEVC\QP37\CNN2\CNN2_I_QP37.ckpt")
    for dir in [y for y in os.listdir(reconDir) if os.path.isdir(os.path.join(reconDir, y)) ]:#
        # print(dir)
        if 36<int(dir.split("qp")[1])<47:     #改成模型对应的QP范围
            for i in range(1,NUM_CNN+1):
                sub_save_dir=os.path.join(saveDir+str(i),dir)
                # print(sub_save_dir.replace("cnn1","cnn2"))
                if os.path.exists(sub_save_dir)==0:
                    os.mkdir(sub_save_dir)
            file_list=load_file_list(os.path.join(reconDir,dir))
            # shuffle(file_list)
            for file in file_list:
                # print(file)
                # exit(0)
                # qp=int(dir[-2:])
                # print(os.path.join(sub_save_dir ,os.path.basename(file[0])),qp)
                # input_img=c_getYdata(file[0])
                # qp_img=np.ones(np.shape(input_img),dtype="int8")*qp*4
                # input_img=np.dstack((input_img,qp_img))
                rec= mModel.predict(c_getYdata(file))
                saveImg(os.path.join(saveDir+"1" ,dir, os.path.basename(file)), rec)
                for i in range(2,NUM_CNN+1):
                    rec = mModel.predict(rec)
                    saveImg(os.path.join(saveDir+str(i),dir, os.path.basename(file)), rec)

if __name__ == '__main__':
    # low_img = r"D:\konglingyi\test_result\E19091301\BasketballPass_416x240_50_l.yuv"
    # heigh_img=r"D:\konglingyi\test_result\E19091301\BasketballPass_416x240_50_h.yuv"
    # init(0,37)
    # original_ycbcr, gt_y, fileName_list=prepare_test_data(heigh_img)
    # rec1 = predict(low_img, 160,0)
    #showImg(rec1)
    #rec2 = predict(file, 212,2)
    #print(psnr(denormalize(np.reshape(original_ycbcr[0][0],np.shape(rec1))), rec1))



    # test_all_ckpt(r"progressive_cnn_filter\models\firstCNN\HEVC\QP37\CNN2")
    build_trainset()
    # for i in range(1,1):
    #     print(i)