#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


#include <cv.h>

// #include <opencv2/photo/photo.hpp>


using namespace cv;

/*compute the primal error and residual*/
__inline bool stopPri(float **X, float *sol, int m, int n, float absTol, float relTol, int display);

/*compute the dual error and resudual*/
__inline bool stopDual(float *sol, float *sol_o, float *U1, float *U2, float *U3, int m, int n, 
				float rho, float absTol, float relTol, int display);

/*compute the root of the quartic function*/
__inline float NewtonRoot(const float a, const float b, const float c, const float d);

/*get the function value*/
__inline float get_funval(const float *sol, const float *Y, const float lam, const int imgHeight, const int imgWidth);

/*method for isotropic TV*/
int tvl2_iso(float* sol, float* Y, const int imgHeight, const int imgWidth, const float lam, const float rho, 
			  const int maxIter, const float absTol, const float relTol, const int display);

			  
__inline float NewtonRoot(const float a, const float b, const float c, const float d){
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

bool stopPri(float **X, float *sol, int m, int n, float absTol, float relTol, int display)
{
	const float sqrt3 = sqrt(3.0);
	float rk = 0, epri = 0, ex = 0, ez = 0;
	const int imgDim = m*n;
	int i;
	int max_threads = omp_get_max_threads();
	
	omp_set_num_threads(max_threads);
	#pragma omp parallel shared (X,sol) private(i) reduction(+: rk, ex, ez) 
	{
		#pragma omp for
		for(i = 0; i<imgDim; i++)
		{
			rk += (X[0][i]-sol[i])*(X[0][i]-sol[i]) + (X[1][i]-sol[i])*(X[1][i]-sol[i]) + (X[2][i]-sol[i])*(X[2][i]-sol[i]);
			ex += X[0][i]*X[0][i] + X[1][i]*X[1][i] + X[2][i]*X[2][i];
			ez += sol[i]*sol[i];
		}
	}
	
	
	
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
bool stopDual(float *sol, float *sol_o, float *U1, float *U2, float *U3, int m, int n, 
				float rho, float absTol, float relTol, int display)
{
	float sqrt3 = sqrt(3.0);
	float sk = 0, ed = 0;
	int imgDim = m*n;
	int i;
	int max_threads = omp_get_max_threads();
	
	omp_set_num_threads(max_threads);
	#pragma omp parallel shared(sol,sol_o,U1,U2,U3,imgDim) private(i) reduction(+: sk,ed)
	{	
		#pragma omp for
		for (i=0;i<imgDim;i++)
		{
			sk += (sol[i]-sol_o[i])*(sol[i]-sol_o[i]);
			ed += U1[i]*U1[i] + U2[i]*U2[i] + U3[i]*U3[i];
		}
	}
	
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

int tvl2_iso(float* sol, float* Y, const int imgHeight, const int imgWidth, const float lam, const float rho, 
			  const int maxIter, const float absTol, const float relTol, const int display)
{
	float *U1, *U2, *U3, *sol_o, *T, td, w0,w1,w2, temp1, temp2;
	float **X;
	int iter, i, j;
	unsigned int *BlkInd;
	const int imgDim = imgHeight*imgWidth;
	float wt0,wt1;
	// set some frequently used constants
	const float flam = lam/rho;
	const float rhoinv = 1.0/rho;
	const float flamp2 = flam*flam;
	const float flamp2_12 = 12*flamp2;
	const float flamp2_22 = 22*flamp2;
	const float flamp2_9 = 9*flamp2;
	const float invflamp2 = 1/flamp2;
	const float flamp4 = flamp2*flamp2;
	const float flamp6 = flamp4*flamp2;
	const float sqrt2 = sqrt(2.0);
	const float sqrt3 = sqrt(3.0);
	const float aa = 8.0*flamp2;
	const float inv3 = 1/3.;
	float alpha, temp,obj, ox1,ox2,ox3,oz;

	// intilize some used matrix
	float GGTI[4] = {0, 0,
					  0, 0};
	float GTGGTI[9] = {-1.0, 1.0, 0, 
						  0, -1.0, 1.0,
						  0, -1.0 , 1.0};
	
	float bb,cc,dd, aa2, bb2,aabb;
	float t1,t2;
	
	X = (float **) malloc(3*sizeof(float *));
	X[0] = (float *)malloc(imgDim*sizeof(float));
	X[1] = (float *)malloc(imgDim*sizeof(float));
	X[2] = (float *)malloc(imgDim*sizeof(float));
	BlkInd = (unsigned int *) malloc(imgDim*sizeof(unsigned int));
	sol_o  =  (float *) malloc(imgDim*sizeof(float));
	U1  =  (float *) calloc(imgDim, sizeof(float));
	U2  =  (float *) calloc(imgDim, sizeof(float));
	U3  =  (float *) calloc(imgDim, sizeof(float));

	int max_threads = omp_get_max_threads();
	
	// initilization
	td = 1.0/(1 + 3* rho);

	
	memcpy(sol,Y,imgDim*sizeof(float));
	memcpy(sol_o,Y,imgDim*sizeof(float));
		
	omp_set_num_threads(max_threads);
	#pragma omp parallel shared(BlkInd) private(i,j)
	{
		#pragma omp for //schedule(dynamic, NUM_CHUNK)
		for(j = 0; j<imgWidth; j++)
		{
			for(i = j%3; i<imgHeight; i+=3)
			{
				BlkInd[i+j*imgHeight] = 0;
			}

			for(i = (j+1)%3; i<imgHeight; i+=3)
			{
				BlkInd[i+j*imgHeight] = 1;
			}

			for(i = (j+2)%3; i<imgHeight; i+=3)
			{
				BlkInd[i+j*imgHeight] = 2;
			}
		}
	}
	
	int k = 0;
	int ind0 = 0;
	int ind1 = 0;
	int ind2 = 0;
	for(iter = 0; iter<maxIter; iter++)
	{
		omp_set_num_threads(max_threads);
		#pragma omp parallel shared(X,sol,U1,U2,U3) private(i) 
		{
			#pragma omp for //schedule(dynamic)
			for(i=0;i<imgDim;i++)
			{
				X[0][i] = sol[i] - U1[i] * rhoinv;
				X[1][i] = sol[i] - U2[i] * rhoinv;
				X[2][i] = sol[i] - U3[i] * rhoinv;
			}
		}		
			
		omp_set_num_threads(max_threads);
		#pragma omp parallel shared(X,BlkInd) private(i,j,k,ind0,ind1,ind2,w0,w1,w2,wt0,wt1,bb,cc,dd,alpha,temp,GGTI,GTGGTI) //schedule(dynamic, NUM_CHUNK)
		{
			#pragma omp for
			for(j = 0; j<imgWidth-1; j++)
			{
				for(i = 0; i<imgHeight-1; i++)
				{
				
					k = BlkInd[i+j*imgHeight];
					ind0 = i+1+j*imgHeight;
					ind1 = i+j*imgHeight;
					ind2 = i+(j+1)*imgHeight;

					// set [w0;w1;w2];
					w0 = X[k][ind0];w1 = X[k][ind1];w2=X[k][ind2];

					// compute [wt0;wt1] = (GG^T)^-1Gw
					wt0 = (w1 + w2 - 2*w0)*inv3;
					wt1 = (2*w2 - w1 - w0)*inv3;
					
					// compute lam_max
					if (flamp2 >= wt1*wt1 + wt0*wt0)
					{
						// compute u = (w0 + w1 + w2)/3;
						X[k][ind0] = X[k][ind1] = X[k][ind2] = (w0 + w1 + w2)*inv3; 
						continue;
					}

					////////////////////////////////////////////
					// compute w\tilde, since S\Sigma is fixed and constant
					wt0 = w0 - 2*w1 + w2;
					wt1 = w2 - w0;
					wt0 *=wt0; wt0 *=0.5;
					wt1 *=wt1; wt1 *=0.5;

					// compute c0,c1,c2,c3 in the paper
					//aa = 8.0*flamp2;
					bb = flamp2*(flamp2_22 - wt0 - wt1);
					cc = 2.0*flamp4 * (flamp2_12 - wt0 - 3*wt1);
					dd = flamp6 * (flamp2_9 - wt0 - 9*wt1);
					

					// compute alpha use the fourth solution in the closed form soltuions 
					alpha = NewtonRoot(aa,bb,cc,dd);
					
					/* compute inv(GG' + alpha/lamp2*I), note that the regualrziaton paramter is flam*/
					alpha *= invflamp2;
					temp = alpha*alpha + 4.*alpha + 3;
					GGTI[0] = GGTI[3] = (alpha + 2.)/temp;
					GGTI[1] = GGTI[2] = 1./temp;

					/* I - G'inv(GG' + alpha/lamp2*I)G*/
					GTGGTI[0] = 1.0 - GGTI[0];
					GTGGTI[4] = 1.0 - GGTI[0] + GGTI[1] + GGTI[2] - GGTI[3];
					GTGGTI[8] = 1.0 - GGTI[3];
					GTGGTI[1] = GTGGTI[3] = GGTI[0] - GGTI[1];
					GTGGTI[2] = GTGGTI[6] = GGTI[1];
					GTGGTI[5] = GTGGTI[7] = GGTI[3] - GGTI[1];

					/* compute u*/
					X[k][ind0]   = (GTGGTI[0]*w0 + GTGGTI[1]*w1 + GTGGTI[2]*w2);
					X[k][ind1]     = (GTGGTI[3]*w0 + GTGGTI[4]*w1 + GTGGTI[5]*w2);
					X[k][ind2] = (GTGGTI[6]*w0 + GTGGTI[7]*w1 + GTGGTI[8]*w2);

				}
			}

			#pragma omp for
			for(i = 0; i < imgHeight-1; i++)
			{
				k = BlkInd[i + (imgWidth-1)*imgHeight];
				ind0 = i+(imgWidth-1)*imgHeight;
				ind1 = i+1+(imgWidth-1)*imgHeight;
				w0 = X[k][ind0]; w1 = X[k][ind1];
				if(w0 > w1 + 2*flam){
					X[k][ind0] = w0 - flam; X[k][ind1] = w1 + flam;}
				else if(w1 > w0 + 2*flam){
					X[k][ind0] = w0 + flam; X[k][ind1] = w1 - flam;}
				else
				{
					X[k][ind0] = X[k][ind1]= (w0 + w1)*0.5;
				}
			}

			#pragma omp for
			for(j = 0; j<imgWidth-1; j++)
			{
				k = BlkInd[imgHeight-1+j*imgHeight];
				ind0 = imgHeight-1+j*imgHeight;
				ind1 = imgHeight-1+(j+1)*imgHeight;
				w0 = X[k][ind0]; w1 = X[k][ind1];
				if(w0 > w1 + 2*flam)
				{
					X[k][ind0] = w0 - flam; X[k][ind1] = w1 + flam;
				}
				else if(w1 > w0 + 2*flam)
				{
					X[k][ind0] = w0 + flam; X[k][ind1] = w1 - flam;
				}
				else
				{
					X[k][ind0] = X[k][ind1]= (w0 + w1)*0.5;
				}
			}
		}
			
	/*compute sol, U1, U2, U3*/
		td = 1./(1+3*rho);			
		omp_set_num_threads(max_threads);
		#pragma omp parallel shared(sol, Y, U1, U2, U3, X,td) private(i)
		{
			#pragma omp for
			for(i=0; i<imgDim;i++)
			{
				sol[i] = (Y[i] + U1[i] + U2[i] + U3[i] 
							+ rho*(X[0][i] + X[1][i] + X[2][i]))*td;
				U1[i] += rho * (X[0][i] - sol[i]);
				U2[i] += rho * (X[1][i] - sol[i]);
				U3[i] += rho * (X[2][i] - sol[i]);
			}
		}

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
			if(!stopDual(sol,sol_o,U1,U2,U3,imgHeight,imgWidth, rho,absTol,relTol,display)){
				memcpy(sol_o, sol, imgDim*sizeof(float));
				continue;
			}		
			if(stopPri(X,sol,imgHeight,imgWidth, absTol,relTol,display)){
				break;
			}
		}	
			
		memcpy(sol_o, sol, imgDim*sizeof(float));
	}

	
	free(X[0]);
	free(X[1]);
	free(X[2]);
	free(X);
	free(BlkInd);
	free(U1);
	free(U2);
	free(U3);
	free(sol_o);
	
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
    int display = 0;

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

