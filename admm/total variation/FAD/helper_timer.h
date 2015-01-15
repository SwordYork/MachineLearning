#include <stdlib.h>
#include <time.h>



struct timespec now, tmstart;

void start_timer() {
    clock_gettime(CLOCK_REALTIME, &tmstart);
    return;
}


void stop_timer() {
    clock_gettime(CLOCK_REALTIME, &now);
    return;
}

double elasp_time() {
    stop_timer();
//    printf("s: %ld, %ld\n", tmstart.tv_sec, tmstart.tv_nsec);
//    printf("n: %ld, %ld\n", now.tv_sec, now.tv_nsec);
    return (now.tv_sec * 1000 + now.tv_nsec * 1e-6 - tmstart.tv_sec * 1000 - tmstart.tv_nsec*1e-6);
}
