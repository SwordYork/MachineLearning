#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cv.h>


#ifndef PFAD_H
#define PFAD_H

cv::Mat total_variation(cv::Mat image);

cv::Mat cuda_total_variation(cv::Mat image);
#endif
