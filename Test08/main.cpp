// 下面为进行检测的C++ 代码

//     分别对摄像头、单张图像以及文件夹图像进行检测，摄像头和单张图像检测代码以注释掉了，然后读取文件夹中所有图像进行检测后的结果以原图像名称进行保存：

#include "util.hpp"
#include "yolo.hpp"
int main()
{
    if (!isReady())
    {
        std::cerr << "cuda 和 cudnn 没有准备好" << std::endl;
        return -1;
    }

    // set up threshold

    system("pause");
    return 0;
}
