#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include "computepi.h"

double compute_pi_baseline(size_t N)
{
    double pi = 0.0;               
    for (size_t i = 1; i < N*4; i+=4) {
        double x = 1.0/(double) i - 1.0/((double) i+2);      /*calculat 1/1-1/3+1/5-1/7....*/ 
        pi += x;       
    }
    return pi * 4.0;
}

double compute_pi_openmp(size_t N, int threads)
{
    double pi = 0.0;
    double x =0.0;
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for private(x) reduction(+:pi)
        for (size_t i = 1; i < N*4; i+=4) {
        x = 1.0/(double) i - 1.0/((double) i+2);      /*calculat 1/1-1/3+1/5-1/7....*/ 
        pi += x;       
    }
    }
    return pi * 4.0;
}

double compute_pi_avx(size_t N)
{
    double pi = 0.0;
    
    register __m256d ymm1, ymm2, ymm3;
    
    ymm3 = _mm256_setzero_pd();             // sum of pi

    for (int i = 1; i <= N*4; i += 16) {
        ymm2 = _mm256_set_pd(1.0/(i),1.0/(i+4.0),1.0/(i+8.0),1.0/(i+12.0));      // 1/i postive
        ymm1 = _mm256_set_pd(1.0/(i+2.0),1.0/(i+6.0),1.0/(i+10.0),1.0/(i+14.0));   // 1/(i+2) subtract
        ymm3 = _mm256_add_pd(ymm3, ymm2);   // pi+1/(i)+1/(i+4).....
        ymm3 = _mm256_sub_pd(ymm3, ymm1);   // pi-1/(i+2)+1/(i+6).....
        
    }
    double tmp[4] __attribute__((aligned(32)));
    _mm256_store_pd(tmp, ymm3);             // move packed float64 values to  256-bit aligned memory location
    pi += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    return pi * 4.0;
}

double compute_pi_avx_unroll(size_t N)
{
    double pi = 0.0;
    register __m256d ymm1, ymm2, ymm3, ymm4,
             ymm5, ymm6, ymm7, ymm8, ymm9,
             ymm10,ymm11, ymm12;
    

    ymm9 = _mm256_setzero_pd();             // first sum of pi
    ymm10 = _mm256_setzero_pd();             // second sum of pi
    ymm11 = _mm256_setzero_pd();             // third sum of pi
    ymm12 = _mm256_setzero_pd();             // fourth sum of pi

    for (int i = 1; i <= N*4; i += 64) {
        ymm1 = _mm256_set_pd(1.0/((double) i),1.0/(i+4.0),1.0/(i+8.0),1.0/(i+12.0));      // 1/i postive
        ymm2 = _mm256_set_pd(1.0/(i+2.0),1.0/(i+6.0),1.0/(i+10.0),1.0/(i+14.0));   // 1/(i+2) subtract
        ymm3 = _mm256_set_pd(1.0/(i+16.0),1.0/(i+20.0),1.0/(i+24.0),1.0/(i+28.0));      // 1/i postive
        ymm4 = _mm256_set_pd(1.0/(i+18.0),1.0/(i+22.0),1.0/(i+26.0),1.0/(i+30.0));   // 1/(i+2) subtract
        ymm5 = _mm256_set_pd(1.0/(i+32.0),1.0/(i+36.0),1.0/(i+40.0),1.0/(i+44.0));      // 1/i postive
        ymm6 = _mm256_set_pd(1.0/(i+34.0),1.0/(i+38.0),1.0/(i+42.0),1.0/(i+46.0));   // 1/(i+2) subtract
        ymm7 = _mm256_set_pd(1.0/(i+48.0),1.0/(i+52.0),1.0/(i+56.0),1.0/(i+60.0));      // 1/i postive
        ymm8 = _mm256_set_pd(1.0/(i+50.0),1.0/(i+54.0),1.0/(i+58.0),1.0/(i+62.0));   // 1/(i+2) subtract

        

        
        ymm1 = _mm256_sub_pd(ymm1, ymm2);
        ymm3 = _mm256_sub_pd(ymm3, ymm4);
        ymm5 = _mm256_sub_pd(ymm5, ymm6);
        ymm7 = _mm256_sub_pd(ymm7, ymm8);
        

        ymm9  =  _mm256_add_pd(ymm9, ymm1);
        ymm10 =  _mm256_add_pd(ymm10, ymm3);
        ymm11 =  _mm256_add_pd(ymm11, ymm5);
        ymm12 = _mm256_add_pd(ymm12, ymm7);
    }

    double tmp1[4] __attribute__((aligned(32)));
    double tmp2[4] __attribute__((aligned(32)));
    double tmp3[4] __attribute__((aligned(32)));
    double tmp4[4] __attribute__((aligned(32)));

    _mm256_store_pd(tmp1, ymm9);
    _mm256_store_pd(tmp2, ymm10);
    _mm256_store_pd(tmp3, ymm11);
    _mm256_store_pd(tmp4, ymm12);

    pi += tmp1[0] + tmp1[1] + tmp1[2] + tmp1[3] +
          tmp2[0] + tmp2[1] + tmp2[2] + tmp2[3] +
          tmp3[0] + tmp3[1] + tmp3[2] + tmp3[3] +
          tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];
    return pi * 4.0;
}
