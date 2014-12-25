#include<opencv2/core/core.hpp>
#include<cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    Mat image;
    image = imread("Lenna.png", CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
    
    image.convertTo(image, CV_64FC1);
    Mat gaussian_noise = image.clone();
    randn(gaussian_noise,0,10);
    image += gaussian_noise;

    cout << image.at<double>(0, 1) / 256 << endl;

    image.convertTo(image, CV_8U);
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", image);                   // Show our image inside it.
    imwrite("t.jpg", image);


    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
