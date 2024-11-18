#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using std::cout;
using cv::Mat;

cv::Size scale_size(const cv::Size &size, double factor) {
    return cv::Size(
        static_cast<int>(size.width * factor),
        static_cast<int>(size.height * factor)
    );
}

Mat bilinear_interpolation(const Mat &src, double factor) {
    cv::Size src_size = src.size();
    cv::Size dst_size = scale_size(src_size, factor);
    Mat dst = cv::Mat::zeros(dst_size, CV_8U);

    for(int i = 0; i < dst_size.height; ++i) {
        for(int j = 0; j < dst_size.width; ++j) {
            double src_i = i / factor;
            double src_j = j / factor;

            int top_i = std::floor(i / factor);
            int bot_i = std::ceil(i / factor);

            int left_j = std::floor(j / factor);
            int right_j = std::ceil(j / factor);
            
            double di = src_i - top_i; // Fractional part in i direction
            double dj = src_j - left_j;

            //interpolation in i direction on the left
            double first_i_interp = 
                src.at<uchar>(top_i, left_j) * (1 - di)
                + src.at<uchar>(bot_i, left_j) * (di);
            double second_i_interp = 
                src.at<uchar>(top_i, right_j) * (1 - di)
                + src.at<uchar>(bot_i, right_j) * (di);

            double intensity = first_i_interp * (1 - dj) + second_i_interp * (dj);
            dst.at<uchar>(i,j) = std::clamp(static_cast<int>(intensity), 0, 255);
        }
    }

    return dst;
}

int main(int argc, char** argv) {
    const cv::String keys = 
        "{@inputpath |  | input image path}"
        "{@outputpath |  | output image path}"
        "{@factor |  | }"
        "{show | false | show the results?}";

    cv::CommandLineParser parser(argc, argv, keys);

    auto input_path = parser.get<cv::String>(0);
    auto output_path = parser.get<cv::String>(1);
    auto factor = parser.get<double>(2);
    auto show = parser.get<bool>("show");

    Mat img = cv::imread(input_path);
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    Mat dst = bilinear_interpolation(img, factor);

    if (show) {
        cv::imshow("Original Image", img);
        cv::imshow("Interpolated Image", dst);
        while('q' != cv::waitKey(0));
    }

    cv::imwrite(output_path, dst);
    return 0;
}
