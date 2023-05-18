#ifndef SPC5_H
#define SPC5_H

#ifdef __cplusplus
extern "C" {
#endif

enum SPC5_MATRIX_TYPE_C{
    SPC5_FORMAT_1rVc_WT,
    SPC5_FORMAT_2rV2c_WT,
    SPC5_FORMAT_2rV2c,
    SPC5_FORMAT_2rVc,
    SPC5_FORMAT_4rV2c,
    SPC5_FORMAT_4rVc,
    SPC5_FORMAT_8rV2c,
    SPC5_FORMAT_DEFAULT
};

void* spc5_init_float(const enum SPC5_MATRIX_TYPE_C format, const int rows, const int cols, const int nnz, const int *row_indx, const int * col_indx, const float *values);
void* spc5_init_double(const enum SPC5_MATRIX_TYPE_C format, const int rows, const int cols, const int nnz, const int *row_indx, const int * col_indx, const double *values);
void spc5_free_float(void*);
void spc5_free_double(void*);
void spc5_spmv_float(void* ptr, const float* x, float* y);
void spc5_spmv_double(void* ptr, const double* x, double* y);
int spc5_blockcount_float(void* ptr);
int spc5_blockcount_double(void* ptr);
#ifdef _OPENMP
void spc5_spmv_float_omp(void* ptr, const float* x, float* y);
void spc5_spmv_double_omp(void* ptr, const double* x, double* y);
#endif

#ifdef __cplusplus
}
#endif


#endif
