#pragma once
#include <cstdio>
#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <fstream>
#include <io.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

using namespace std;

// 读入指定文件夹下的所有文件
void getFiles(string path, vector<string> &files, vector<string> &filenames)
{
    intptr_t hFile = 0; // intptr_t和uintptr_t是什么类型:typedef long int/ typedef unsigned long int
    struct _finddata_t fileinfo;
    string p;
    // assign方法可以理解为先将原字符串清空，然后赋予新的值作替换。
    if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
    {
        do
        {
            if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
            // 这句有点不明白，如果不加，识别的文件里就有.和..两个文件，哪位大神可以给解释下？感激不尽！！！
            {
                files.push_back(p.assign(path).append("\\").append(fileinfo.name));
                filenames.push_back(fileinfo.name);
            }

        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}

// 检测我们的 cuda 和  cudnn是否可用

bool isReady()
{
    // cout << "cuda是否可用：" << torch::cuda::is_available() << endl;
    // cout << "cudnn是否可用：" << torch::cuda::cudnn_is_available() << endl;
    return torch::cuda::is_available() && torch::cuda::cudnn_is_available();
}
