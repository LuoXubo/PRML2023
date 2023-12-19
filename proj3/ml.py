"""
@Description :   Machine learning methods: SVM, kernel SVM, GaussianNB
@Author      :   Xubo Luo 
@Time        :   2023/12/17 15:10:28
"""

import os
import cv2
import math
import time
import numpy as np
import tqdm
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import _pickle as pickle
import argparse
 
 
class Classifier(object):
    def __init__(self, filePath, classifier='linear_svm', is_save=False):
        self.filePath = filePath
        self.type = classifier
        self.is_save = is_save
        if classifier == 'linear_svm':
            self.classifier = LinearSVC()
        elif classifier == 'kernel_svm':
            self.classifier = SVC(kernel='rbf', C=1, gamma='scale')
        elif classifier == 'gaussian_nb':
            self.classifier = GaussianNB()

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
 
    def get_data(self):
        TrainData = []
        TestData = []
        for b in range(1,6):
            f = os.path.join(self.filePath, 'data_batch_%d' % (b, ))
            data = self.unpickle(f)
            train = np.reshape(data[b'data'], (10000, 3, 32 * 32))
            labels = np.reshape(data[b'labels'], (10000, 1))
            fileNames = np.reshape(data[b'filenames'], (10000, 1))
            datalebels = zip(train, labels, fileNames)
            TrainData.extend(datalebels)
        f = os.path.join(self.filePath,'test_batch')
        data = self.unpickle(f)
        test = np.reshape(data[b'data'], (10000, 3, 32 * 32))
        labels = np.reshape(data[b'labels'], (10000, 1))
        fileNames = np.reshape(data[b'filenames'], (10000, 1))
        TestData.extend(zip(test, labels, fileNames))

        print("Data reading finished!")
        return TrainData, TestData
 
    def get_hog_feat(self, image, stride=8, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        cx, cy = pixels_per_cell
        bx, by = cells_per_block
        sx, sy = image.shape
        n_cellsx = int(np.floor(sx // cx))  # number of cells in x
        n_cellsy = int(np.floor(sy // cy))  # number of cells in y
        n_blocksx = (n_cellsx - bx) + 1
        n_blocksy = (n_cellsy - by) + 1
        gx = np.zeros((sx, sy), dtype=np.float32)
        gy = np.zeros((sx, sy), dtype=np.float32)
        eps = 1e-5
        grad = np.zeros((sx, sy, 2), dtype=np.float32)
        for i in range(1, sx-1):
            for j in range(1, sy-1):
                gx[i, j] = image[i, j-1] - image[i, j+1]
                gy[i, j] = image[i+1, j] - image[i-1, j]
                grad[i, j, 0] = np.arctan(gy[i, j] / (gx[i, j] + eps)) * 180 / math.pi
                if gx[i, j] < 0:
                    grad[i, j, 0] += 180
                grad[i, j, 0] = (grad[i, j, 0] + 360) % 360
                grad[i, j, 1] = np.sqrt(gy[i, j] ** 2 + gx[i, j] ** 2)
        normalised_blocks = np.zeros((n_blocksy, n_blocksx, by * bx * orientations))
        for y in range(n_blocksy):
            for x in range(n_blocksx):
                block = grad[y*stride:y*stride+16, x*stride:x*stride+16]
                hist_block = np.zeros(32, dtype=np.float32)
                eps = 1e-5
                for k in range(by):
                    for m in range(bx):
                        cell = block[k*8:(k+1)*8, m*8:(m+1)*8]
                        hist_cell = np.zeros(8, dtype=np.float32)
                        for i in range(cy):
                            for j in range(cx):
                                n = int(cell[i, j, 0] / 45)
                                hist_cell[n] += cell[i, j, 1]
                        hist_block[(k * bx + m) * orientations:(k * bx + m + 1) * orientations] = hist_cell[:]
                normalised_blocks[y, x, :] = hist_block / np.sqrt(hist_block.sum() ** 2 + eps)
        return normalised_blocks.ravel()
 
    def get_feat(self, TrainData, TestData):
        train_feat = []
        test_feat = []
        for data in tqdm.tqdm(TestData):
            image = np.reshape(data[0].T, (32, 32, 3))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.
            fd = self.get_hog_feat(gray) #你可以用我写的hog提取函数，也可以用下面skimage提供的，我的速度会慢一些
            # fd = hog(gray) #你可以用我写的hog提取函数，也可以用下面skimage提供的，我的速度会慢一些
            # fd = hog(gray, 9, [8, 8], [2, 2])
            fd = np.concatenate((fd, data[1]))
            test_feat.append(fd)
        test_feat = np.array(test_feat)
        np.save("test_feat.npy", test_feat)
        print("Test features are extracted and saved.")
        for data in tqdm.tqdm(TrainData):
            image = np.reshape(data[0].T, (32, 32, 3))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.
            fd = self.get_hog_feat(gray)
            # fd = hog(gray, 9, [8, 8], [2, 2])
            fd = np.concatenate((fd, data[1]))
            train_feat.append(fd)
        train_feat = np.array(train_feat)
        np.save("train_feat.npy", train_feat)
        print("Train features are extracted and saved.")
        return train_feat, test_feat
 
    def classification(self, train_feat, test_feat):
        t0 = time.time()
        
        print("\nTraining the %s classifier."%self.type)
        tik = time.time()
        self.classifier.fit(train_feat[:, :-1], train_feat[:, -1])
        tok = time.time()
        print("Training finished.")
        print("Time cost: %f\n" % (tok - tik))

        # 保存模型
        if self.is_save:
            with open('caches/%s.pkl'%self.type, 'wb') as f:
                pickle.dump(self.classifier, f)
        
        predict_result = self.classifier.predict(test_feat[:, :-1])
        num = 0
        for i in range(len(predict_result)):
            if int(predict_result[i]) == int(test_feat[i, -1]):
                num += 1
        rate = float(num) / len(predict_result)
        t1 = time.time()
        print('The testing classification accuracy is %f' % rate)
        print('The testing time cost is :%f' % (t1 - t0))
 
        # predict_result2 = clf.predict(train_feat[:, :-1])
        # num2 = 0
        # for i in range(len(predict_result2)):
        #     if int(predict_result2[i]) == int(train_feat[i, -1]):
        #         num2 += 1
        # rate2 = float(num2) / len(predict_result2)
        # print('The Training classification accuracy is %f' % rate2)
 
    def run(self):
        if os.path.exists("train_feat.npy") and os.path.exists("test_feat.npy"):
            train_feat = np.load("train_feat.npy")
            test_feat = np.load("test_feat.npy")
        else:
            TrainData, TestData = self.get_data()
            train_feat, test_feat = self.get_feat(TrainData, TestData)
        self.classification(train_feat, test_feat)
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train', type = bool, default = True, help = 'Train the model (True - train and test, False - only test)')
    parser.add_argument('--file_path', type = str, default = '../../cifar-10-python/cifar-10-batches-py', help = 'path of cifar-10-python')
    parser.add_argument('--classifier', type = str, default = 'linear_svm', help = 'classifiers (linear_svm, kernel_svm, gaussian_nb)')
    parser.add_argument('--is_save', type = bool, default = False, help = 'save the model')

    args = parser.parse_args()
    filePath = args.file_path
    classifer = args.classifier
    is_save = args.is_save
    # is_train = args.train

    cf = Classifier(filePath, classifier=classifer, is_save=is_save)
    cf.run()