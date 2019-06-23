import os
import gc
import string
import bottle
import cv2
import numpy
import tensorflow as tf
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data


class CodeServers:
    W1 = None
    b1 = None
    W2 = None
    b2 = None

    # 初始化
    def Init(self):
        self.W1 = numpy.load("Data/W1.npy")
        self.b1 = numpy.load("Data/b1.npy")
        self.W2 = numpy.load("Data/W2.npy")
        self.b2 = numpy.load("Data/b2.npy")

    # 去噪
    def rSalt(self, img):
        img = img[1:26, 5:55]
        return cv2.medianBlur(img, 3)

    # 二值
    def binar(self, img):
        ret, thresh1 = cv2.threshold(img, 108, 255, cv2.THRESH_BINARY)
        del ret, img
        for x in locals().keys():
            del locals()[x]
        gc.collect()
        return thresh1

    # 分割
    def cutImg(self, img, fileName):
        folder = 'newimg' + os.path.sep
        n = numpy.shape(img)[1]
        width = int(n/4)
        arr = []
        for i in range(4):
            sonImg = img[:, i * width: (i + 1) * width]
            sonFileName = folder + \
                fileName[:fileName.rfind('.')] + str(i) + ".png"
            cv2.imwrite(sonFileName, sonImg)
            arr.append(sonFileName)
            del sonImg, sonFileName
        del width, n, folder, img, fileName
        for x in locals().keys():
            del locals()[x]
        gc.collect()
        return arr

    # 转换为行向量
    def img2Vec(self, imgPath):
        img = cv2.imread(imgPath, 0)
        returnVec = numpy.zeros(numpy.shape(img), dtype=numpy.float32)
        returnVec[img == 0] = 1
        returnVec[img == 255] = 0
        del img, imgPath
        for x in locals().keys():
            del locals()[x]
        gc.collect()
        return returnVec.flatten()

    # 去噪，二值化，分割，按顺序返回子图路径
    def getAllSonImg(self, imgPath):
        im = Image.open(imgPath)
        imgPath = imgPath.replace(".jpg", "")
        if(os.path.exists(imgPath + '.png')):
            try:
                os.remove(imgPath + '.png')
            except:
                pass
        im.save(imgPath + '.png')
        try:
            del(im)
            os.remove(imgPath + '.jpg')
        except:
            pass

        imgPath = imgPath + '.png'
        img = cv2.imread(imgPath, 0)
        fileName = imgPath[(imgPath.rfind(os.path.sep) + 1):]
        img1 = self.rSalt(img)
        img2 = self.binar(img1)
        del img, img1
        for x in locals().keys():
            del locals()[x]
        gc.collect()
        return self.cutImg(img2, fileName)

    # 获取标签与序号对应的字典
    def getLabelIndexDir(self):
        dir = {}
        i = 0
        for j in range(9):
            dir[str(j)] = i
            i = i + 1
        for char in string.ascii_lowercase:
            if char == 'o' or char == 'z':
                continue
            dir[char] = i
            i = i + 1
        for x in locals().keys():
            del locals()[x]
        gc.collect()
        return dir

    # 获取分类结果
    def getRes2(self, imgPath, w1, b1, w2, b2):
        inx = numpy.mat(self.img2Vec(imgPath))
        for x in locals().keys():
            del locals()[x]
        gc.collect()
        return self.getRes1(inx, w1, b1, w2, b2)

    # 获取分类结果
    def getRes1(self, inx, w1, b1, w2, b2):
        # 构建会话
        sess = tf.InteractiveSession()
        # 创建矩阵
        inx = numpy.mat(inx)
        x = tf.convert_to_tensor(inx)
        W1 = tf.convert_to_tensor(w1)
        b1 = tf.convert_to_tensor(b1)
        W2 = tf.convert_to_tensor(w2)
        b2 = tf.convert_to_tensor(b2)
        # 激活函数，将矩阵中每行非最大的元素置0
        hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
        y = tf.nn.softmax(tf.matmul(hidden1, W2) + b2)
        tf.global_variables_initializer().run()
        index = tf.argmax(y, 1).eval()[0]
        labelIndexDir = self.getLabelIndexDir()
        i = list(labelIndexDir.keys())[
            list(labelIndexDir.values()).index(index)]
        sess.close()
        del sess, labelIndexDir, index, y, hidden1, b2, W2, b1, W1, x, inx
        for x in locals().keys():
            del locals()[x]
        gc.collect()
        return i

    # 获取识别结果
    def GetCode(self, imgPath):
        son = self.getAllSonImg(imgPath)
        res = ''
        for item in son:
            res = res + self.getRes2(item, self.W1, self.b1, self.W2, self.b2)
            self.DelImg(item)
        imgPath = imgPath.replace(".jpg", ".png")
        self.DelImg(imgPath)
        del imgPath, son
        for x in locals().keys():
            del locals()[x]
        gc.collect()
        return res

    def DelImg(self, path):
        try:
            os.remove(path)
        except:
            pass


# 初始化CodeServer对象
codeServer = CodeServers()
print("CodeServer 创建完毕")
# 初始化数据
codeServer.Init()
print("CodeServer 初始化完毕")

# 定义脚本路径
base_path = os.path.dirname(os.path.realpath(__file__))
# print("服务根目录 ： {0}".format(base_path))

# 定义上传文件存放路径
update_path = os.path.join(base_path, 'update')
print("上传文件存放目录 ： {0}" .format(update_path))


@bottle.route("/codeApi", method="POST")
def handle():
    try:
        data = bottle.request.files.get("file")
        if data.file:
            # 文件存放路径
            file_path = os.path.join(update_path, data.filename)
            data.save(file_path, overwrite=True)
            print('新的文件存放在  {0} : '.format(file_path))
            rec = codeServer.GetCode(file_path)
            print('识别结果 : {0}'.format(rec))
            return rec
    except Exception as e:
        print(e)
        pass
    finally:
        for x in locals().keys():
            del locals()[x]
        gc.collect()




bottle.run(host='0.0.0.0', port=8522)
