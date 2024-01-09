"""
@Description :   Machine learning methods: SVM, kernel SVM, GaussianNB
@Author      :   Xubo Luo 
@Time        :   2023/12/17 15:10:28
"""

import argparse
from models.ml_models import Classifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type = str, default = '../../cifar-10-python/cifar-10-batches-py', help = 'path of cifar-10-python')
    parser.add_argument('--classifier', type = str, default = 'linear_svm', help = 'classifiers (linear_svm, kernel_svm, gaussian_nb)')
    parser.add_argument('--hog', type = int, default = 1, help = 'use hog (1) or not (0)')
    parser.add_argument('--is_save', type = int, default = 1, help = 'save the model (1) or not (0)')

    args = parser.parse_args()
    filePath = args.file_path
    classifer = args.classifier
    is_hog = (args.hog == 1)
    is_save = (args.is_save == 1)

    cf = Classifier(filePath, classifier=classifer, is_hog = is_hog, is_save=is_save)
    cf.run()