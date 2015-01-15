#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
#include <omp.h>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cv.h>

#include "median_filter.h"

#define THREADSNUM 4
#define WINDOWSIZE 7
#define MAXV 256
#define MAXAB(a,b) ((a) > (b) ? (a) : (b)) 
#define MINAB(a,b) ((a) < (b) ? (a) : (b)) 



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  median filter
 *  Description:  This implement is run thread according row, which will increase cache
 *                performance, and it's easy to code.
 * =====================================================================================
 */


void median_channel(int *origin, int *result, int WIDTH, int HEIGHT)
{
    int N = WIDTH * HEIGHT;
    int n = omp_get_thread_num();
    // each threads calculate some points
    // as if there are only T-1 threads 
    // and the last thread may do few things
   
    int per_thread;
    per_thread = N / THREADSNUM;    
    int start = n * per_thread;
    int end = start + per_thread;
    int is, js, ie, je, i, j, prev;
    int total;
    int tmp[MAXV];
    int left = N - per_thread * THREADSNUM;

    while (start < end) {
        is = MAXAB(start / WIDTH - WINDOWSIZE / 2, 0);
        js = MAXAB(start % WIDTH - WINDOWSIZE / 2, 0);
        ie = MINAB(is + WINDOWSIZE, HEIGHT-1);
        je = MINAB(js + WINDOWSIZE, WIDTH-1);

        for (i=0; i<MAXV; ++i)
            tmp[i] = 0;

        for (i=is;i<ie;++i) {
            for (j=js;j<je;++j){
                tmp[origin[i*WIDTH + j]]++;
            }
        }
        
        total = 0;
        i = 0;
        prev = 0;
        while (total < (je-js)*(ie-is)/2) {
            if (tmp[i] != 0)
                prev = i;
            total += tmp[i++];
        }
        result[start] = prev;
        start++;
    }
   
    // do the left point which can't be covered.
    if (n < left){
        start =  per_thread * THREADSNUM + n;

        is = MAXAB(start / WIDTH - WINDOWSIZE / 2, 0);
        js = MAXAB(start % WIDTH - WINDOWSIZE / 2, 0);
        ie = MINAB(is + WINDOWSIZE, HEIGHT-1);
        je = MINAB(js + WINDOWSIZE, WIDTH-1);

        for (i=0; i<MAXV; ++i)
            tmp[i] = 0;

        for (i=is;i<ie;++i) {
            for (j=js;j<je;++j){
                tmp[origin[i*WIDTH+j]]++;
            }
        }
        
        total = 0;
        i = 0;
        while (total < (je-js)*(ie-is)/2) total += tmp[i++];
        result[start] = i - 1;
    }
}


void median_filter(cv::Mat input, cv::Mat output)
{
    
    cv::Mat chans[3];


    cv::Size size = input.size();
    int imgHeight = size.height;
    int imgWidth = size.width;

    int *outputImg = (int *) malloc(imgHeight * imgWidth * sizeof(int));
    int *inputImg = (int *) malloc(imgHeight * imgWidth * sizeof(int));

    cv::split(input, chans); 


    for (int c = 0; c < 3; ++c) {
        // i=0: read R data
        // i=1: read G data
        // i=2: read B data

        for (int i=0; i < imgHeight; ++i)
            for (int j=0; j < imgWidth; ++j)
                inputImg[i * imgWidth + j] = chans[c].at<uchar>(i, j);

        #pragma omp parallel num_threads(THREADSNUM)
        median_channel(inputImg, outputImg, imgWidth, imgHeight);


        for (int i=0; i < imgHeight; ++i)
            for (int j=0; j < imgWidth; ++j)
                chans[c].at<uchar>(i, j) = outputImg[i * imgWidth + j];


    }

    cv::merge(chans, 3, output);

}				/* ----------  end of function main  ---------- */
