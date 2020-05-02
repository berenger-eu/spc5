#ifndef SPC5_H
#define SPC5_H

#ifdef __cplusplus
extern "C" {
#endif

void* spc5_init_float(const int rows, const int cols, const int nnz, const int *row_indx, const int * col_indx, const float *values);
void* spc5_init_double(const int rows, const int cols, const int nnz, const int *row_indx, const int * col_indx, const double *values);
void spc5_free_float(void*);
void spc5_free_double(void*);
void spc5_spmv_float(void* ptr, const float* x, float* y);
void spc5_spmv_double(void* ptr, const double* x, double* y);
#ifdef _OPENMP
void spc5_spmv_float_omp(void* ptr, const float* x, float* y);
void spc5_spmv_double_omp(void* ptr, const double* x, double* y);
#endif

#ifdef __cplusplus
}
#endif


#endif
