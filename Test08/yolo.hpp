
#include <torch/script.h>
#include <memory>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>
#include "util.hpp"
#include <fstream>
#include <string>
using namespace cv;
using namespace std;

class Yolo
{
public:
    enum Det
    {
        tl_x = 0,
        tl_y = 1,
        br_x = 2,
        br_y = 3,
        score = 4,
        class_idx = 5
    };

    struct Detection
    {
        cv::Rect bbox;
        float score;
        int class_idx;
    };

public:
    Yolo(std::string &model_path, const std::string &label_path)
        : _model_path(model_path), _label_path(label_path)
    {
        torch::DeviceType device_type;
        device_type = torch::kCUDA;
        torch::Device device(device_type);
        torch::jit::script::Module module;
        module = torch::jit::load(_model_path.c_str(), device); // 加载模型
        module.eval();
        _load_names(_label_path.c_str(), &_class_names);
        float conf_thres = 0.4;
        float iou_thres = 0.5;
    }

public:
    // 1. 摄像头检测
    void CameraDetection()
    {
        // 视频检测
        VideoCapture video1(0); // 打开笔记本自带摄像头（1）为外接摄像头
        video1.set(CAP_PROP_FRAME_WIDTH, 1280);
        video1.set(CAP_PROP_FRAME_HEIGHT, 720);
        // 读取视频帧率
        double rate = video1.get(CAP_PROP_FPS);
        std::cout << "rate: " << rate << std::endl;
        // 当前视频帧
        Mat frame;
        // 每一帧之间的延时
        int delay = 1000 / rate;
        bool stop(false);
        while (!stop)
        {
            double t = (double)cv::getTickCount(); // 开始计时

            if (!video1.read(frame))
            {
                std::cout << "no video frame" << std::endl;
                break;
            }
        }
    }
    // 2. 文件夹检测
    // void Folder detection(const std::string &filePath, const string &dst_filePath)
    // {
    //     // 文件夹进行检测
    //     char windowname[10];

    //     std::vector<string> files;
    //     std::vector<string> filenames;
    //     // 得到该文件夹下所有的子文件夹
    //     getFiles(filePath, files, filenames);

    //     for (size_t i = 0; i < files.size(); i++)
    //     {
    //         // clock_t t_start = clock();
    //         string saveFilename = dst_filePath + filenames[i]; // 写入部分
    //         Mat img = imread(files[i]);
    //         clock_t t_start = clock();

    //         // cv::Mat img = imread("right_outside_005458.bmp");
    //         // inference
    //         torch::NoGradGuard no_grad;
    //         cv::Mat img_input = img.clone();
    //         std::vector<float> pad_info = LetterboxImage(img_input, img_input, cv::Size(640, 640));
    //         const float pad_w = pad_info[0];
    //         const float pad_h = pad_info[1];
    //         const float scale = pad_info[2];
    //         cv::cvtColor(img_input, img_input, cv::COLOR_BGR2RGB); // BGR -> RGB
    //         // 归一化需要是浮点类型
    //         img_input.convertTo(img_input, CV_32FC3, 1.0f / 255.0f); // normalization 1/255
    //         // 加载图像到设备
    //         auto tensor_img = torch::from_blob(img_input.data, {1, img_input.rows, img_input.cols, img_input.channels()}).to(device_type);
    //         // BHWC -> BCHW
    //         tensor_img = tensor_img.permute({0, 3, 1, 2}).contiguous(); // BHWC -> BCHW (Batch, Channel, Height, Width)
    //         std::vector<torch::jit::IValue> inputs;
    //         // 在容器尾部添加一个元素，这个元素原地构造，不需要触发拷贝构造和转移构造
    //         inputs.emplace_back(tensor_img);
    //         // start = clock();
    //         torch::jit::IValue output = module.forward(inputs);

    //         // 解析结果
    //         auto detections = output.toTuple()->elements()[0].toTensor();
    //         auto result = PostProcessing(detections, pad_w, pad_h, scale, img.size(), conf_thres, iou_thres);

    //         cv::Mat pre_img = Demo(img, result, class_names);
    //         // imwrite("结构图.jpg",pre_img);
    //         clock_t t_stop = clock();
    //         double endtime = (double)(t_stop - t_start) / CLOCKS_PER_SEC;
    //         imwrite(saveFilename, pre_img);
    //     }
    // }

    // 3. 单张图片检测
    // void Image detection()
    // {
    //     单张图像检测
    //     clock_t t_start = clock();

    //     cv::Mat img = imread("right_outside_005458.bmp");
    //     // inference
    //     torch::NoGradGuard no_grad;
    //     cv::Mat img_input = img.clone();
    //     std::vector<float> pad_info = LetterboxImage(img_input, img_input, cv::Size(640, 640));
    //     const float pad_w = pad_info[0];
    //     const float pad_h = pad_info[1];
    //     const float scale = pad_info[2];
    //     cv::cvtColor(img_input, img_input, cv::COLOR_BGR2RGB); // BGR -> RGB
    //     // 归一化需要是浮点类型
    //     img_input.convertTo(img_input, CV_32FC3, 1.0f / 255.0f); // normalization 1/255
    //     // 加载图像到设备
    //     auto tensor_img = torch::from_blob(img_input.data, {1, img_input.rows, img_input.cols, img_input.channels()}).to(device_type);
    //     // BHWC -> BCHW
    //     tensor_img = tensor_img.permute({0, 3, 1, 2}).contiguous(); // BHWC -> BCHW (Batch, Channel, Height, Width)
    //     std::vector<torch::jit::IValue> inputs;
    //     // 在容器尾部添加一个元素，这个元素原地构造，不需要触发拷贝构造和转移构造
    //     inputs.emplace_back(tensor_img);
    //     // start = clock();
    //     torch::jit::IValue output = module.forward(inputs);

    //     // 解析结果
    //     auto detections = output.toTuple()->elements()[0].toTensor();
    //     auto result = PostProcessing(detections, pad_w, pad_h, scale, img.size(), conf_thres, iou_thres);

    //     cv::Mat pre_img = Demo(img, result, class_names);
    //     imwrite("结构图.jpg", pre_img);
    //     clock_t t_stop = clock();
    //     double endtime = (double)(t_stop - t_start) / CLOCKS_PER_SEC;

    //     cout << "检测时间：" << endtime << endl;
    //     t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); // 结束计时
    //     int fps = int(1.0 / t);                                        // 转换为帧率

    //     std::cout << "FPS: " << fps << std::endl; // 输出帧率

    //     putText(pre_img, ("FPS: " + std::to_string(fps)), Point(0, 50), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 0)); // 输入到帧frame上
    //     cv::namedWindow("Result", cv::WINDOW_NORMAL);
    //     cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
    //     cv::imshow("Result", pre_img);

    //     waitKey() 函数的作用是刷新imshow() 展示的图片 if (waitKey(10) == 27) // 27是键盘摁下esc时，计算机接收到的ascii码值
    //     {
    //         break;
    //     }
    // }
    // video1.release();

private:
    void Tensor2Detection(const at::TensorAccessor<float, 2> &offset_boxes,
                          const at::TensorAccessor<float, 2> &det,
                          std::vector<cv::Rect> &offset_box_vec,
                          std::vector<float> &score_vec)
    {

        for (int i = 0; i < offset_boxes.size(0); i++)
        {
            offset_box_vec.emplace_back(
                cv::Rect(cv::Point(offset_boxes[i][Det::tl_x], offset_boxes[i][Det::tl_y]),
                         cv::Point(offset_boxes[i][Det::br_x], offset_boxes[i][Det::br_y])));
            score_vec.emplace_back(det[i][Det::score]);
        }
    }

    void ScaleCoordinates(std::vector<Detection> &data, float pad_w, float pad_h,
                          float scale, const cv::Size &img_shape)
    {
        auto clip = [](float n, float lower, float upper)
        {
            return std::max(lower, std::min(n, upper));
        };

        std::vector<Detection> detections;
        for (auto &i : data)
        {
            float x1 = (i.bbox.tl().x - pad_w) / scale; // x padding
            float y1 = (i.bbox.tl().y - pad_h) / scale; // y padding
            float x2 = (i.bbox.br().x - pad_w) / scale; // x padding
            float y2 = (i.bbox.br().y - pad_h) / scale; // y padding

            x1 = clip(x1, 0, img_shape.width);
            y1 = clip(y1, 0, img_shape.height);
            x2 = clip(x2, 0, img_shape.width);
            y2 = clip(y2, 0, img_shape.height);

            i.bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
        }
    }

    torch::Tensor xywh2xyxy(const torch::Tensor &x)
    {
        auto y = torch::zeros_like(x);
        // convert bounding box format from (center x, center y, width, height) to (x1, y1, x2, y2)
        y.select(1, Det::tl_x) = x.select(1, 0) - x.select(1, 2).div(2);
        y.select(1, Det::tl_y) = x.select(1, 1) - x.select(1, 3).div(2);
        y.select(1, Det::br_x) = x.select(1, 0) + x.select(1, 2).div(2);
        y.select(1, Det::br_y) = x.select(1, 1) + x.select(1, 3).div(2);
        return y;
    }

    std::vector<std::vector<Detection>> PostProcessing(const torch::Tensor &detections,
                                                       float pad_w, float pad_h, float scale, const cv::Size &img_shape,
                                                       float conf_thres, float iou_thres)
    {
        /***
         * 结果纬度为batch index(0), top-left x/y (1,2), bottom-right x/y (3,4), score(5), class id(6)
         * 13*13*3*(1+4)*80
         */
        constexpr int item_attr_size = 5;
        int batch_size = detections.size(0);
        // number of classes, e.g. 80 for coco dataset
        auto num_classes = detections.size(2) - item_attr_size;

        // get candidates which object confidence > threshold
        auto conf_mask = detections.select(2, 4).ge(conf_thres).unsqueeze(2);

        std::vector<std::vector<Detection>> output;
        output.reserve(batch_size);

        // iterating all images in the batch
        for (int batch_i = 0; batch_i < batch_size; batch_i++)
        {
            // apply constrains to get filtered detections for current image
            auto det = torch::masked_select(detections[batch_i], conf_mask[batch_i]).view({-1, num_classes + item_attr_size});

            // if none detections remain then skip and start to process next image
            if (0 == det.size(0))
            {
                continue;
            }

            // compute overall score = obj_conf * cls_conf, similar to x[:, 5:] *= x[:, 4:5]
            det.slice(1, item_attr_size, item_attr_size + num_classes) *= det.select(1, 4).unsqueeze(1);

            // box (center x, center y, width, height) to (x1, y1, x2, y2)
            torch::Tensor box = xywh2xyxy(det.slice(1, 0, 4));

            // [best class only] get the max classes score at each result (e.g. elements 5-84)
            std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(det.slice(1, item_attr_size, item_attr_size + num_classes), 1);

            // class score
            auto max_conf_score = std::get<0>(max_classes);
            // index
            auto max_conf_index = std::get<1>(max_classes);

            max_conf_score = max_conf_score.to(torch::kFloat).unsqueeze(1);
            max_conf_index = max_conf_index.to(torch::kFloat).unsqueeze(1);

            // shape: n * 6, top-left x/y (0,1), bottom-right x/y (2,3), score(4), class index(5)
            det = torch::cat({box.slice(1, 0, 4), max_conf_score, max_conf_index}, 1);

            // for batched NMS
            constexpr int max_wh = 4096;
            auto c = det.slice(1, item_attr_size, item_attr_size + 1) * max_wh;
            auto offset_box = det.slice(1, 0, 4) + c;

            std::vector<cv::Rect> offset_box_vec;
            std::vector<float> score_vec;

            // copy data back to cpu
            auto offset_boxes_cpu = offset_box.cpu();
            auto det_cpu = det.cpu();
            const auto &det_cpu_array = det_cpu.accessor<float, 2>();

            // use accessor to access tensor elements efficiently
            Tensor2Detection(offset_boxes_cpu.accessor<float, 2>(), det_cpu_array, offset_box_vec, score_vec);

            // run NMS
            std::vector<int> nms_indices;
            cv::dnn::NMSBoxes(offset_box_vec, score_vec, conf_thres, iou_thres, nms_indices);

            std::vector<Detection> det_vec;
            for (int index : nms_indices)
            {
                Detection t;
                const auto &b = det_cpu_array[index];
                t.bbox =
                    cv::Rect(cv::Point(b[Det::tl_x], b[Det::tl_y]),
                             cv::Point(b[Det::br_x], b[Det::br_y]));
                t.score = det_cpu_array[index][Det::score];
                t.class_idx = det_cpu_array[index][Det::class_idx];
                det_vec.emplace_back(t);
            }

            ScaleCoordinates(det_vec, pad_w, pad_h, scale, img_shape);

            // save final detection for the current image
            output.emplace_back(det_vec);
        } // end of batch iterating

        return output;
    }

    cv::Mat Demo(cv::Mat &img,
                 const std::vector<std::vector<Detection>> &detections,
                 const std::vector<std::string> &class_names,
                 bool label = true)
    {
        if (!detections.empty())
        {
            for (const auto &detection : detections[0])
            {
                const auto &box = detection.bbox;
                float score = detection.score;
                int class_idx = detection.class_idx;

                cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);

                if (label)
                {
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(2) << score;
                    std::string s = class_names[class_idx] + " " + ss.str();

                    auto font_face = cv::FONT_HERSHEY_DUPLEX;
                    auto font_scale = 1.0;
                    int thickness = 1;
                    int baseline = 0;
                    auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
                    cv::rectangle(img,
                                  cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
                                  cv::Point(box.tl().x + s_size.width, box.tl().y),
                                  cv::Scalar(0, 0, 255), -1);
                    cv::putText(img, s, cv::Point(box.tl().x, box.tl().y - 5),
                                font_face, font_scale, cv::Scalar(255, 255, 255), thickness);
                }
            }
        }
        return img;
        // cv::namedWindow("Result", cv::WINDOW_NORMAL);
        // cv::imshow("Result", img);
    }

    std::vector<float> LetterboxImage(const cv::Mat &src, cv::Mat &dst, const cv::Size &out_size)
    {
        auto in_h = static_cast<float>(src.rows);
        auto in_w = static_cast<float>(src.cols);
        float out_h = out_size.height;
        float out_w = out_size.width;

        float scale = std::min(out_w / in_w, out_h / in_h);

        int mid_h = static_cast<int>(in_h * scale);
        int mid_w = static_cast<int>(in_w * scale);

        cv::resize(src, dst, cv::Size(mid_w, mid_h));

        int top = (static_cast<int>(out_h) - mid_h) / 2;
        int down = (static_cast<int>(out_h) - mid_h + 1) / 2;
        int left = (static_cast<int>(out_w) - mid_w) / 2;
        int right = (static_cast<int>(out_w) - mid_w + 1) / 2;

        cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        std::vector<float> pad_info{static_cast<float>(left), static_cast<float>(top), scale};
        return pad_info;
    }

    // 加载标签的名字
    void _load_names(const std::string &path, std::vector<std::string> *class_names)
    {
        std::ifstream infile(path);
        if (!infile.is_open())
        {
            std::cerr << "Error loading the class names!\n";
            return -2;
        }
        std::string line;
        while (std::getline(infile, line))
        {
            class_names->emplace_back(line);
        }
        infile.close();
    }

private:
    std::string _model_path;               // 训练模型的地址
    std::string _label_path;               // 标签地址
    std::vector<std::string> _class_names; // 这里是标签的名字
};