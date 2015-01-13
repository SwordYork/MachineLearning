#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <cv.h>

#include "utils.h"
#define BLOCK_WIDTH 256


using namespace cv;


/*compute the primal error and residual*/
bool stopPri(float *X, float *sol, int m, int n, float absTol, float relTol, int display);

/*compute the dual error and resudual*/
bool stopDual(float *sol, float *sol_o, float *U1, float *U2, float *U3, int m, int n, 
				float rho, float absTol, float relTol, int display);


/*get the function value*/
__inline float get_funval(const float *sol, const float *Y, const float lam, const int imgHeight, const int imgWidth);

/*method for isotropic TV*/
int tvl2_iso(float* sol, float* Y, const int imgHeight, const int imgWidth, const float lam, const float rho, 
			  const int maxIter, const float absTol, const float relTol, const int display);

			  
__device__
float d_NewtonRoot(const float a, const float b, const float c, const float d){
	// initial x = 0.1
	double x = 0.1;
	double x2 = 0.01;
	double x3 = 0.001;
	double x4 = 0.0001;
	double y = 1e-4 + a*1e-3 + b*1e-2 + c*0.1 + d;
	double dy = 0.;
	int count = 0;

	while(y<0){
		x*=2;
		x2 = 2*x;
		x3 = 4*x;
		x4 = 8*x;
		y = x4 + a*x3 + b*x2 + c*x + d;
	}
	while(fabs(y)>1e-15){
		dy = 4*x3 + 3*a*x2 + 2*b*x + c;
		x-=y/dy;
		x2 = x*x;
		x3 = x2*x;
		x4 = x2*x2;
		y = x4 + a*x3 + b*x2 + c*x + d;
		count ++;
		if(count > 100){
			printf("Newton method fails %f\n", (float) y);
		}
	}
	
	return x;
}


__global__
void atomic_reduction_pri(float *d_X, float *d_sol, float *d_rk, float *d_ex, float *d_ez, int imgDim)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= imgDim) return;

    atomicAdd(d_rk,(float) (d_X[i]-d_sol[i])*(d_X[i]-d_sol[i]) + (d_X[imgDim + i]-d_sol[i])*(d_X[imgDim + i]-d_sol[i]) + (d_X[2*imgDim + i] - d_sol[i])*(d_X[2*imgDim + i]-d_sol[i]));
    atomicAdd(d_ex, (float) d_X[i]*d_X[i] + d_X[imgDim + i]*d_X[imgDim + i] + d_X[2*imgDim + i]*d_X[2 * imgDim + i]);
    atomicAdd(d_ez, (float) d_sol[i] * d_sol[i]);

}

__global__
void atomic_reduction_dual(float *d_sol, float *d_sol_o, float *d_U1, float *d_U2, float *d_U3, float *d_sk, float *d_ed, int imgDim)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= imgDim) return;


    atomicAdd(d_sk, (float) (d_sol[i]-d_sol_o[i])*(d_sol[i]-d_sol_o[i]));
	atomicAdd(d_ed, (float) d_U1[i]*d_U1[i] + d_U2[i]*d_U2[i] + d_U3[i]*d_U3[i]);

}

bool stopPri(float *d_X, float *d_sol, int m, int n, float absTol, float relTol, int display)
{
	const float sqrt3 = sqrt(3.0);
	float rk = 0, epri = 0, ex = 0, ez = 0;
	const int imgDim = m*n;
	int i;
	
    float *d_rk, *d_ez, *d_ex;

    checkCudaErrors(cudaMalloc((void **) &d_rk, sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_ez, sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_ex, sizeof(float)));

    checkCudaErrors(cudaMemset((void *) d_rk, 0, sizeof(float)));
    checkCudaErrors(cudaMemset((void *) d_ez, 0, sizeof(float)));
    checkCudaErrors(cudaMemset((void *) d_ex, 0, sizeof(float)));

	
    atomic_reduction_pri<<<(imgDim + BLOCK_WIDTH - 1) / BLOCK_WIDTH, BLOCK_WIDTH>>> (d_X, d_sol, d_rk, d_ex, d_ez, imgDim);
	
    checkCudaErrors(cudaMemcpy((void *) &rk, (void *) d_rk, sizeof(float),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void *) &ez, (void *) d_ez, sizeof(float),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void *) &ex, (void *) d_ex, sizeof(float),cudaMemcpyDeviceToHost));


    checkCudaErrors(cudaFree(d_rk));
    checkCudaErrors(cudaFree(d_ez));
    checkCudaErrors(cudaFree(d_ex));

	rk = sqrt(rk);
	ez = sqrt3*sqrt(ez);
	ex = sqrt(ex);
	epri = ex>ez?ex:ez;
	i = m>n?m:n;
	epri *= relTol;
	epri += sqrt3*i*absTol;

	if(display == 1){
		printf("PriError and Resideual is %f %f\n", (float) epri, (float) rk);
	}
	return (rk <= epri);
}


// dual error and dual residual: stop 2
bool stopDual(float *d_sol, float *d_sol_o, float *d_U1, float *d_U2, float *d_U3, int m, int n, 
				float rho, float absTol, float relTol, int display)
{
	float sqrt3 = sqrt(3.0);
	float sk = 0, ed = 0;
	int imgDim = m*n;
	int i;

    float *d_sk, *d_ed;

    checkCudaErrors(cudaMalloc((void **) &d_sk, sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_ed, sizeof(float)));

    checkCudaErrors(cudaMemset((void *) d_sk, 0, sizeof(float)));
    checkCudaErrors(cudaMemset((void *) d_ed, 0, sizeof(float)));

    atomic_reduction_dual<<<(imgDim + BLOCK_WIDTH - 1) / BLOCK_WIDTH, BLOCK_WIDTH>>> (d_sol, d_sol_o, d_U1, d_U2, d_U3, d_sk, d_ed, imgDim);
	
    checkCudaErrors(cudaMemcpy((void *) &sk, (void *) d_sk, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void *) &ed, (void *) d_ed, sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_sk));
    checkCudaErrors(cudaFree(d_ed));


	
	sk =rho*sqrt3*sqrt(sk);
	ed = sqrt(ed);
	ed *= relTol;
	i = m>n?m:n;
	ed += sqrt3*i*absTol;
	
	if(display == 1){
		printf("DualError and Resideual is %f %f\n", (float) ed, (float) sk);
	}
	
	return (sk <= ed);
}

float get_funval(const float *sol, const float *Y, const float lam, const int imgHeight, const int imgWidth){
	
	float obj = 0.;
	float temp = 0.;
	int i, j;
	int imgDim = imgHeight * imgWidth;
		
	for (i=0; i<imgDim;i++){
		obj += (sol[i] - Y[i])*(sol[i] - Y[i]);
	}
	obj *=0.5;

	for(i=0;i<imgHeight-1;i++){
		for(j=0;j<imgWidth-1;j++){
			temp += sqrt((sol[i+j*imgHeight]- sol[i+j*imgHeight+1])*(sol[i+j*imgHeight]- sol[i+j*imgHeight+1]) + (sol[i+j*imgHeight]- sol[i+(j+1)*imgHeight])*(sol[i+j*imgHeight]- sol[i+(j+1)*imgHeight]));
		}			
	}
				
	for(j=0;j<imgWidth-1;j++){
		temp += fabs(sol[imgHeight-1+j*imgHeight] - sol[imgHeight + imgHeight-1+j*imgHeight]);
	}

	for(i=0;i<imgHeight-1;i++){
		temp += fabs(sol[i + imgHeight*(imgWidth-1)] - sol[1+i + imgHeight*(imgWidth-1)]);
	}
	
	return obj + lam*temp;
}


__global__
void fill_BlkInd(unsigned int *d_BlkInd, const int imgHeight, const int imgDim)
{
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    threadId *= 3;
    int i = threadId % imgHeight;
    int j = threadId / imgHeight;
    int s;
    
    if (i % 3 == j % 3)
        s = 0;
    else if ((i%3) == (j+1) % 3)
        s = 1;
    else 
        s = 2;

    if (threadId < imgDim)
        d_BlkInd[threadId++] = s;

    if (threadId < imgDim)
        d_BlkInd[threadId++] = (s + 1) % 3;

    if (threadId < imgDim)
        d_BlkInd[threadId] = (s + 2) % 3;
}

__global__
void update_X(float *d_X, float *d_sol, float *d_U, const float rhoinv, const int imgDim, const int p)
{
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadId < imgDim)
		d_X[p*imgDim + threadId] = d_sol[threadId] - d_U[threadId] * rhoinv;

}

__global__
void inside_update(float *d_X, unsigned int *d_BlkInd, const int imgHeight, const int imgWidth, const float flam)
{
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    int i = threadId % imgHeight;
    int j = threadId / imgHeight;


    if (i >= imgHeight -1 || j >= imgWidth - 1)
        return;

    int k = 0;
	int ind0 = 0;
	int ind1 = 0;
	int ind2 = 0;

    int imgDim = imgWidth * imgHeight;

    float w0,w1,w2, wt0, wt1, alpha, temp, bb, cc, dd;

    float d_GGTI[4];
    float d_GTGGTI[9];


	const float flamp2 = flam*flam;
	const float aa = 8.0f*flamp2;
        
    k = d_BlkInd[i+j*imgHeight];
    ind0 = i+1+j*imgHeight;
    ind1 = i+j*imgHeight;
    ind2 = i+(j+1)*imgHeight;

    // set [w0;w1;w2];
    w0 = d_X[k * imgDim + ind0];
    w1 = d_X[k * imgDim + ind1];
    w2 = d_X[k * imgDim + ind2];

    // compute [wt0;wt1] = (GG^T)^-1Gw
    wt0 = (w1 + w2 - 2*w0) / 3;
    wt1 = (2*w2 - w1 - w0) / 3;

    // compute lam_max
    if (flamp2 >= wt1*wt1 + wt0*wt0)
    {
        // compute u = (w0 + w1 + w2)/3;
        d_X[k * imgDim + ind0] = (w0 + w1 + w2) / 3; 
        d_X[k * imgDim + ind1] = (w0 + w1 + w2) / 3; 
        d_X[k * imgDim + ind2] = (w0 + w1 + w2) / 3; 
        return;
    }

    ////////////////////////////////////////////
    // compute w\tilde, since S\Sigma is fixed and constant
    wt0 = w0 - 2*w1 + w2;
    wt1 = w2 - w0;
    wt0 *=wt0; 
    wt0 *=0.5f;
    wt1 *=wt1; 
    wt1 *=0.5f;

    // compute c0,c1,c2,c3 in the paper
    //aa = 8.0*flamp2;
	bb = flamp2*(flamp2 * 22 - wt0 - wt1);
    cc = 2.0f*flamp2 * flamp2 * (flamp2 * 12 - wt0 - 3*wt1);
	dd = flamp2 * flamp2 * flamp2 * (flamp2 * 9 - wt0 - 9*wt1);


    // compute alpha use the fourth solution in the closed form soltuions 
    alpha = d_NewtonRoot(aa,bb,cc,dd);

    /* compute inv(GG' + alpha/lamp2*I), note that the regualrziaton paramter is flam*/
    alpha *= (1 / flamp2);
    temp = alpha*alpha + 4.f*alpha + 3;
    d_GGTI[0] = d_GGTI[3] = (alpha + 2.f)/temp;
    d_GGTI[1] = d_GGTI[2] = 1.f/temp;

    /* I - G'inv(GG' + alpha/lamp2*I)G*/
    d_GTGGTI[0] = 1.0f - d_GGTI[0];
    d_GTGGTI[4] = 1.0f - d_GGTI[0] + d_GGTI[1] + d_GGTI[2] - d_GGTI[3];
    d_GTGGTI[8] = 1.0f - d_GGTI[3];
    d_GTGGTI[1] = d_GTGGTI[3] = d_GGTI[0] - d_GGTI[1];
    d_GTGGTI[2] = d_GTGGTI[6] = d_GGTI[1];
    d_GTGGTI[5] = d_GTGGTI[7] = d_GGTI[3] - d_GGTI[1];

    /* compute u*/
    d_X[k*imgDim + ind0] = (d_GTGGTI[0]*w0 + d_GTGGTI[1]*w1 + d_GTGGTI[2]*w2);
    d_X[k*imgDim + ind1] = (d_GTGGTI[3]*w0 + d_GTGGTI[4]*w1 + d_GTGGTI[5]*w2);
    d_X[k*imgDim + ind2] = (d_GTGGTI[6]*w0 + d_GTGGTI[7]*w1 + d_GTGGTI[8]*w2);

}


__global__
void width_boundary_update(float *d_X, unsigned int *d_BlkInd, const int imgHeight, const int imgWidth, const float flam)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= imgHeight -1)
        return;

    int k, ind0, ind1;
    int imgDim = imgWidth * imgHeight;
    float w0, w1;

    k = d_BlkInd[i + (imgWidth-1)*imgHeight];
    ind0 = i+(imgWidth-1)*imgHeight;
    ind1 = i+1+(imgWidth-1)*imgHeight;
    w0 = d_X[k*imgDim + ind0];
    w1 = d_X[k*imgDim + ind1];
    if(w0 > w1 + 2*flam) {
        d_X[k * imgDim + ind0] = w0 - flam; 
        d_X[k * imgDim + ind1] = w1 + flam;
    }
    else if(w1 > w0 + 2*flam) {
        d_X[k * imgDim + ind0] = w0 + flam; 
        d_X[k * imgDim + ind1] = w1 - flam;
    }
    else
    {
        d_X[k * imgDim + ind0] = d_X[k * imgDim + ind1]= (w0 + w1)*0.5;
    }
}


__global__
void height_boundary_update(float *d_X, unsigned int *d_BlkInd, const int imgHeight, const int imgWidth, const float flam)
{

    int j = threadIdx.x + blockDim.x * blockIdx.x;

    if (j >= imgWidth -1)
        return;

    int k, ind0, ind1;
    int imgDim = imgWidth * imgHeight;
    float w0, w1;


    k = d_BlkInd[imgHeight-1+j*imgHeight];
    ind0 = imgHeight-1+j*imgHeight;
    ind1 = imgHeight-1+(j+1)*imgHeight;
    w0 = d_X[k*imgDim + ind0]; 
    w1 = d_X[k*imgDim + ind1];
    if(w0 > w1 + 2*flam)
    {
        d_X[k * imgDim + ind0] = w0 - flam; 
        d_X[k * imgDim + ind1] = w1 + flam;
    }
    else if(w1 > w0 + 2*flam)
    {
        d_X[k * imgDim + ind0] = w0 + flam; 
        d_X[k * imgDim + ind1] = w1 - flam;
    }
    else
    {
        d_X[k * imgDim + ind0] = d_X[k * imgDim + ind1]= (w0 + w1)*0.5;
    }
}

__global__
void update_sol_u(float *d_X, float *d_sol, float  *d_U1, float *d_U2, float *d_U3, float  *d_Y, int imgDim, float rho)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= imgDim) return;

    d_sol[i] = (d_Y[i] + d_U1[i] + d_U2[i] + d_U3[i] + rho*(d_X[i] + d_X[imgDim + i] + d_X[2 * imgDim + i])) / (1 + 3 * rho);
    d_U1[i] += rho * (d_X[i] - d_sol[i]);
    d_U2[i] += rho * (d_X[imgDim + i] - d_sol[i]);
    d_U3[i] += rho * (d_X[2 * imgDim + i] - d_sol[i]);
}



int tvl2_iso(float* sol, float* Y, const int imgHeight, const int imgWidth, const float lam, const float rho, const int maxIter, const float absTol, const float relTol, const int display)
{
	const int imgDim = imgHeight*imgWidth;
	// set some frequently used constants
	const float flam = lam/rho;
	const float rhoinv = 1.0/rho;
	
    int iter;
	// initilization

	
    /*
     * cuda fill BlkInd
     */
    unsigned int *d_BlkInd;
    unsigned int grid_size = (imgDim + BLOCK_WIDTH * 3) / (BLOCK_WIDTH * 3);

    checkCudaErrors(cudaMalloc((void **) &d_BlkInd, imgDim * sizeof(unsigned int)));
    fill_BlkInd<<<grid_size, BLOCK_WIDTH>>>(d_BlkInd, imgHeight, imgDim);


    float *d_U1, *d_U2, *d_U3, *d_sol_o, *d_sol;
	float *d_X, *d_Y;
    checkCudaErrors(cudaMalloc((void **) &d_X, 3 * imgDim * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_sol_o, imgDim * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_sol, imgDim * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_Y, imgDim * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_U1, imgDim * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_U2, imgDim * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_U3, imgDim * sizeof(float)));

    checkCudaErrors(cudaMemset((void *) d_U1, 0, imgDim * sizeof(float)));
    checkCudaErrors(cudaMemset((void *) d_U2, 0, imgDim * sizeof(float)));
    checkCudaErrors(cudaMemset((void *) d_U3, 0, imgDim * sizeof(float)));

    checkCudaErrors(cudaMemcpy((void *)d_Y, (void *) Y, imgDim * sizeof(float),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void *)d_sol_o, (void *) Y, imgDim * sizeof(float),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void *)d_sol, (void *) Y, imgDim * sizeof(float),cudaMemcpyHostToDevice));



	for(iter = 0; iter<maxIter; iter++)
	{
     
        update_X<<<(imgDim + BLOCK_WIDTH - 1) / BLOCK_WIDTH, BLOCK_WIDTH>>>(d_X, d_sol, d_U1, rhoinv, imgDim, 0);
        update_X<<<(imgDim + BLOCK_WIDTH - 1) / BLOCK_WIDTH, BLOCK_WIDTH>>>(d_X, d_sol, d_U2, rhoinv, imgDim, 1);
        update_X<<<(imgDim + BLOCK_WIDTH - 1) / BLOCK_WIDTH, BLOCK_WIDTH>>>(d_X, d_sol, d_U3, rhoinv, imgDim, 2);

       
        checkCudaErrors(cudaDeviceSynchronize());


        inside_update<<<(imgDim + BLOCK_WIDTH - 1) / BLOCK_WIDTH, BLOCK_WIDTH>>>(d_X, d_BlkInd, imgHeight, imgWidth, flam); 
        width_boundary_update<<<(imgHeight - 1 + BLOCK_WIDTH - 1) / BLOCK_WIDTH, BLOCK_WIDTH>>>(d_X, d_BlkInd, imgHeight, imgWidth, flam); 
        height_boundary_update<<<(imgHeight - 1 + BLOCK_WIDTH - 1) / BLOCK_WIDTH, BLOCK_WIDTH>>>(d_X, d_BlkInd, imgHeight, imgWidth, flam); 


			
	/*compute sol, U1, U2, U3*/

        checkCudaErrors(cudaDeviceSynchronize());

        update_sol_u<<< (imgDim + BLOCK_WIDTH - 1) / BLOCK_WIDTH, BLOCK_WIDTH >>>(d_X, d_sol, d_U1, d_U2, d_U3, d_Y, imgDim, rho);

       /*determine if the stop conditions are satisfied*/
		int step_size = 50;
		if(iter<=50)
		{
			step_size = 50;
		}else if(iter<=100){
			step_size = 20;
		}else{
			step_size = 10;
		}

		if(iter%step_size == 0){
			if(!stopDual(d_sol, d_sol_o, d_U1, d_U2, d_U3, imgHeight, imgWidth, rho,absTol,relTol,display)){
                checkCudaErrors(cudaMemcpy(d_sol_o, d_sol, imgDim * sizeof(float), cudaMemcpyDeviceToDevice));    
				continue;
			}		
			if(stopPri(d_X,d_sol,imgHeight,imgWidth, absTol,relTol,display)){
				break;
			}
		}	
			
        checkCudaErrors(cudaMemcpy(d_sol_o, d_sol, imgDim * sizeof(float), cudaMemcpyDeviceToDevice));    
	}

    checkCudaErrors(cudaMemcpy(sol, d_sol_o, imgDim * sizeof(float), cudaMemcpyDeviceToHost));    
	
    
	checkCudaErrors(cudaFree(d_X));
	checkCudaErrors(cudaFree(d_Y));
	checkCudaErrors(cudaFree(d_U1));
	checkCudaErrors(cudaFree(d_U2));
	checkCudaErrors(cudaFree(d_U3));
	checkCudaErrors(cudaFree(d_sol));
	checkCudaErrors(cudaFree(d_sol_o));

	return iter;
}


Mat total_variation(Mat image) {
    Mat chans[3];

    Size size = image.size();
    int imgHeight = size.height;
    int imgWidth = size.width;

    float lam = 0.3;
    float gamma = 7;
    int maxIter = 1000;
    float tol[] = {1e-4, 1e-4};
    int display = 1;

    float *inputImg, *outputImg, *iter, *funVal;

    outputImg = (float *) malloc(imgHeight * imgWidth * sizeof(float));
    inputImg = (float *) malloc(imgHeight * imgWidth * sizeof(float));
    iter = (float *) malloc(sizeof(float));
    funVal = (float *) malloc(sizeof(float));

    split(image, chans);

    for (int c = 0; c < 3; ++c) {
        // B channel
        for (int i=0; i < imgHeight; ++i)
            for (int j=0; j < imgWidth; ++j) 
                inputImg[i*imgWidth + j] = chans[c].at<float>(i, j) / 256;

        memcpy(outputImg, inputImg, imgHeight * imgWidth * sizeof(float));
        //
        iter[0] = (float)tvl2_iso(outputImg, inputImg, imgHeight, imgWidth, lam, gamma, maxIter,tol[0],tol[1],display);
        funVal[0] = get_funval(outputImg,inputImg,lam,imgHeight,imgWidth);
        //

        for (int i=0; i < imgHeight; ++i)
            for (int j=0; j < imgWidth; ++j)
                chans[c].at<float>(i, j) = outputImg[i * imgWidth + j]  * 256;

    }


    merge(chans, 3, image);
    image.convertTo(image, CV_8UC3);


    free(outputImg);
    free(inputImg);
    free(iter);
    free(funVal);
    return image;
}


int main(int argc, char** argv )
{
    Mat origin_image, tv_image, noise_image, image;
    origin_image = imread("test.png", CV_LOAD_IMAGE_COLOR);   // Read the file
    image = origin_image.clone();

    Size size = image.size();

    // create noise image
    image.convertTo(image, CV_32FC4);

    noise_image = image.clone();
    randn(noise_image,0,20);
    noise_image += image;

    // total variation need float type
    tv_image = noise_image.clone();
    
    noise_image.convertTo(noise_image, CV_8UC3);


    // use openmp pFAD version to denoise
    tv_image = total_variation(tv_image);


    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", tv_image);                   

   // waitKey(0);                                      

    return 0;
}
