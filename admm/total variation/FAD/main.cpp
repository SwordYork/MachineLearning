#include <stdio.h>
#include <stdlib.h>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/photo/photo.hpp>

#include "median_filter.h"
#include "FAD.h"
#include "helper_timer.h"

using namespace cv;


int main(int argc, char** argv )
{
    
    char *img_name;
    if (argc != 2) {
        printf("please run as: ./main filename ");
        exit(-1);
    }
    else {
        img_name = argv[1];
        printf("test denoise with %s\n", img_name);
    }

    Mat origin_image, tv_image, noise_image, image, cuda_tv_image, build_in_image, median_filter_denoise;
    origin_image = imread(img_name, CV_LOAD_IMAGE_COLOR);   // Read the file
    image = origin_image.clone();

    Size size = origin_image.size();
    int imgHeight = size.height;
    int imgWidth = size.width;

    // create noise image
    image.convertTo(image, CV_32FC4);
    noise_image = image.clone();
    randn(noise_image,0,20);
    noise_image += image;

    // total variation need double type
    tv_image = noise_image.clone();

    // use openmp pFAD version to denoise
    start_timer();
    tv_image = total_variation(tv_image);
    printf("openmp tv running time:%f ms\n", elasp_time());

    // total variation need double type
    cuda_tv_image = noise_image.clone();

    // use openmp pFAD version to denoise
    start_timer();
    cuda_tv_image = cuda_total_variation(cuda_tv_image);
    printf("cuda tv running time:%f ms\n", elasp_time());

    // byte type
    noise_image.convertTo(noise_image, CV_8UC3);

    // use median_filter method to denoise
    median_filter_denoise = noise_image.clone();

    start_timer();
    median_filter(noise_image, median_filter_denoise);
    printf("median_filte running time:%f ms\n", elasp_time());


    // build in method
    build_in_image = noise_image.clone();
    start_timer();
    fastNlMeansDenoisingColored(origin_image, build_in_image, 10);
    printf("build in method running time:%f ms\n", elasp_time());

    Mat im3(2 * imgHeight, 3 * imgWidth, CV_8UC3);
    // Move right boundary to the left.
    im3.adjustROI(0, -imgHeight, 0, -2*imgWidth);
    origin_image.copyTo(im3);


    // Move the left boundary to the right, right boundary to the right.
    im3.adjustROI(0, 0, -imgWidth, imgWidth);
    noise_image.copyTo(im3);

    im3.adjustROI(0, 0, -imgWidth, imgWidth);
    build_in_image.copyTo(im3);

    im3.adjustROI(-imgHeight, imgHeight, 2*imgWidth, -2*imgWidth);
    median_filter_denoise.copyTo(im3);

    im3.adjustROI(0, 0, -imgWidth, imgWidth);
    tv_image.copyTo(im3);

    im3.adjustROI(0, 0, -imgWidth, imgWidth);
    cuda_tv_image.copyTo(im3);

    // restore original ROI.
    im3.adjustROI(imgHeight, 0, 2*imgWidth, 0);

    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", im3);                   
    imwrite("t.jpg", im3);

    waitKey(0);                                      

    return 0;
}
