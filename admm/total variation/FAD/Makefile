opencvinclude = -I/usr/include/opencv 
opencvlib = -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab -ltbb /lib64/libXext.so /lib64/libX11.so /lib64/libICE.so /lib64/libSM.so /lib64/libGL.so /lib64/libGLU.so -lrt -lpthread -lm -ldl

all:	main

main: main.o  median_filter.o pFAD_OMP.o FAD_CUDA.o
	nvcc -Xcompiler -fopenmp main.o pFAD_CUDA.o median_filter.o pFAD_OMP.o $(opencvlib) -o main


main.o: main.cpp
	g++ -c main.cpp $(opencvinclude) -o main.o

median_filter.o: median_filter.cpp
	g++ -fopenmp -c median_filter.cpp $(opencvinclude) -o median_filter.o

pFAD_OMP.o: pFAD_OMP.cpp
	g++ -fopenmp -c pFAD_OMP.cpp $(opencvinclude) -o pFAD_OMP.o

FAD_CUDA.o: FAD_CUDA.cu
	nvcc -c FAD_CUDA.cu  $(opencvinclude)  


clean:
	rm -f *.o main
