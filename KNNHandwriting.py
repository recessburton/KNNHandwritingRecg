#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Copyright (C),2014-2015, YTC, www.bjfulinux.cn
Created on  2015-04-15 11:00

@author: ytc recessburton@gmail.com
@version: 1.0

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
'''

from numpy import *
from os import listdir
import KNN
from numpy.core import multiarray



def img2vector(filename):
    '图像文件转换成矩阵'
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):             #将32行合并成一行
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect               #一个样本最终成为一个1*1024的向量


def handwritingClassTest():
    '手写识别测试函数，调用了KNN模块的KNN分类器函数'
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = KNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "in #%d, the classifier came back with: %d, the real answer is: %d" % (i, classifierResult, classNumStr)
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
    
handwritingClassTest()
 
    
    


