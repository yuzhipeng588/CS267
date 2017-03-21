const char* dgemm_desc = "Simple blocked dgemm.";
#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <stdlib.h>

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#define STEP_SIZE 32
#define DEBUG 0
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */



  static void new_avx_block (int lda,double* A, double* B, double* C)
  {
  	  __m256d out0,out1,out2,out3,out4,out5,out6,out7;
  __m256d out_sum0,out_sum1,out_sum2,out_sum3;
  __m128d sum_high,result,output;
 
 double* out = (double*)memalign(32,2*sizeof(double));
     for (int j= 0; j < BLOCK_SIZE; ++j) {
      double* B_pos = B+j*BLOCK_SIZE;
      double* C_pos = C+j*lda;
      for (int i=0;i<BLOCK_SIZE;i++){
      	double* A_pos = A+i*BLOCK_SIZE;
        for (int k = 0; k < BLOCK_SIZE;k+=32){

            out0 = _mm256_mul_pd(_mm256_load_pd(A_pos+k), _mm256_load_pd(B_pos+k));
            out1 = _mm256_mul_pd(_mm256_load_pd(A_pos+k+4), _mm256_load_pd(B_pos+k+4));
            out2 = _mm256_mul_pd(_mm256_load_pd(A_pos+k+8), _mm256_load_pd(B_pos+k+8));
            out3 = _mm256_mul_pd(_mm256_load_pd(A_pos+k+12), _mm256_load_pd(B_pos+k+12));
            out4 = _mm256_mul_pd(_mm256_load_pd(A_pos+k+16), _mm256_load_pd(B_pos+k+16));
            out5 = _mm256_mul_pd(_mm256_load_pd(A_pos+k+20), _mm256_load_pd(B_pos+k+20));
            out6 = _mm256_mul_pd(_mm256_load_pd(A_pos+k+24), _mm256_load_pd(B_pos+k+24));
            out7 = _mm256_mul_pd(_mm256_load_pd(A_pos+k+28), _mm256_load_pd(B_pos+k+28));

        out_sum0 = _mm256_add_pd(out0,out1);
        out_sum1 = _mm256_add_pd(out2,out3);
        out_sum2 = _mm256_add_pd(out4,out5);
        out_sum3 = _mm256_add_pd(out6,out7);
        out_sum0 = _mm256_add_pd(out_sum0,out_sum1);
        out_sum2 = _mm256_add_pd(out_sum2,out_sum3);
        out_sum0 = _mm256_add_pd(out_sum0,out_sum2);
      
        sum_high = _mm256_extractf128_pd(out_sum0,1);
        result = _mm_add_pd(sum_high, _mm256_castpd256_pd128(out_sum0));
        output = _mm_hadd_pd(result,_mm_setzero_pd());
        _mm_store1_pd(out,output);
         *(C_pos + i) = *(C_pos + i) + out[0];
        }
      }
    }
  };

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
 void square_dgemm (int lda, double* A, double* B, double* C)
 {

  //Copy optimization

  int PAD_NUM = BLOCK_SIZE;
  int orig_lda = lda;
  if (lda % PAD_NUM != 0)
    lda = lda + (PAD_NUM - lda % PAD_NUM);

  double* new_A = (double*)memalign(32,lda*lda*sizeof(double));
  double* new_B = (double*)memalign(32,lda*lda*sizeof(double));
  double* restrict new_C = (double*)memalign(32,lda*lda*sizeof(double));

  //Assign the array
  int curr_idx = 0;
  int row_idx,col_idx;
  /* For each block-row */
  for (int i = 0; i < lda; i+= BLOCK_SIZE){
  /* For each block-column */
    for (int j = 0 ; j < lda; j+= BLOCK_SIZE){
      //Block content
      for (int a =0 ; a < BLOCK_SIZE; a++){
        row_idx = i + a;
        if (row_idx >= orig_lda){
          new_A[curr_idx] = 0.0;
          new_B[curr_idx] = 0.0;
          curr_idx = curr_idx +BLOCK_SIZE;
        } else{
        for (int b =0 ; b < BLOCK_SIZE; b++)
        {
          col_idx = j + b;
          if ((col_idx >= orig_lda)){
            new_A[curr_idx] = 0.0;
            new_B[curr_idx] = 0.0;
          }
          else {
            new_A[curr_idx] = A[row_idx + orig_lda * col_idx];
            //At any time, the B array current index is the mirror coordinate of A. (i,j) --> (j,i)
            new_B[curr_idx] = B[col_idx + orig_lda * row_idx];
          }
          curr_idx = curr_idx + 1;
        }
        }
      
      }
    }
  }

  /* For each block-row of A */
  for (int j = 0; j < lda; j += BLOCK_SIZE){
    /* For each block-column of B */
    for (int k = 0; k < lda; k += BLOCK_SIZE){
      for (int i = 0; i < lda; i += BLOCK_SIZE){

        new_avx_block(lda, 
          new_A + i*lda + k*BLOCK_SIZE, 
          new_B + j*lda + k*BLOCK_SIZE, 
          new_C + i + j*lda);

      }
    }
  }

  for (int i = 0; i < orig_lda; ++i)
    memcpy(C+i*orig_lda,new_C+i*lda,orig_lda*sizeof(double));

}
