#include "spc5_interface.h"

#include "spc5.hpp"

#include <memory>
#include <vector>

template <class ValueType>
struct MatrixContainer{
    SPC5_MATRIX_TYPE_C format;
    SPC5Mat<ValueType> csr;
    #ifdef _OPENMP
    std::vector<ThreadInterval<ValueType>> intervals;
    #endif
};

template <class ValueType>
void* spc5_init_t(const SPC5_MATRIX_TYPE_C inFormat, const int rows, const int cols, const int nnz, const int *row_indx, const int * col_indx, const ValueType *values){
    std::unique_ptr<Ijv<ValueType>[]> values_ijv(new Ijv<ValueType>[nnz]);

    for(int idxVal = 0 ; idxVal < nnz ; ++idxVal){
        assert(0 <= row_indx[idxVal] && row_indx[idxVal] < rows);
        assert(0 <= col_indx[idxVal] && col_indx[idxVal] < cols);
        values_ijv[idxVal].i = row_indx[idxVal];
        values_ijv[idxVal].j = col_indx[idxVal];
        values_ijv[idxVal].v = values[idxVal];
    }

    MatrixContainer<ValueType>* container = new MatrixContainer<ValueType>;
    if(inFormat == SPC5_FORMAT_DEFAULT){
        if(getenv("SPC5_FORMAT")){
            const char* strFormat = getenv("SPC5_FORMAT");
            if(strcmp(strFormat,"1WT") == 0){
                container->format = SPC5_FORMAT_1rVc_WT;
            }
            else if(strcmp(strFormat,"2") == 0){
                container->format = SPC5_FORMAT_2rVc;
            }
            else if(strcmp(strFormat,"4") == 0){
                container->format = SPC5_FORMAT_4rVc;
            }
            else{
                printf("[SPC5] Invalid SPC5_FORMAT = %s\n", strFormat);
                printf("[SPC5] Should be: 1WT, 2, 4\n");
            }
        }
    }
    else{
        container->format = inFormat;
    }
    switch(container->format){
    case SPC5_FORMAT_1rVc_WT:
        container->csr = (COO_to_SPC5_1rVc<ValueType>(rows, cols, values_ijv.get(), nnz));
        break;
    case SPC5_FORMAT_2rVc:
        container->csr = (COO_to_SPC5_2rVc<ValueType>(rows, cols, values_ijv.get(), nnz));
        break;
    case SPC5_FORMAT_4rVc:
        container->csr = (COO_to_SPC5_4rVc<ValueType>(rows, cols, values_ijv.get(), nnz));
        break;
    default:
        printf("[SPC5] Invalid SPC5 format %d (line %d file %s)\n", container->format, __LINE__, __FILE__);
        exit(-1);
    }

#ifdef _OPENMP
    container->intervals = SPC5_1rVc_split_omp<ValueType>(container->csr);
#endif

    return container;
}


void* spc5_init_float(const enum SPC5_MATRIX_TYPE_C format, const int rows, const int cols, const int nnz, const int *row_indx, const int * col_indx, const float *values){
    return spc5_init_t<float>(format, rows, cols, nnz, row_indx, col_indx, values);
}

void* spc5_init_double(const enum SPC5_MATRIX_TYPE_C format, const int rows, const int cols, const int nnz, const int *row_indx, const int * col_indx, const double *values){
    return spc5_init_t<double>(format, rows, cols, nnz, row_indx, col_indx, values);
}

void spc5_free_float(void* ptr){
    delete ((MatrixContainer<float>*)ptr);
}

void spc5_free_double(void* ptr){
    delete ((MatrixContainer<double>*)ptr);
}

template <class ValueType>
void spc5_spmv_core(void* ptr, const ValueType* x, ValueType* y){
    switch(((MatrixContainer<ValueType>*)ptr)->format){
    case SPC5_FORMAT_1rVc_WT:
        // core_SPC5_rVc_Spmv_scalar<ValueType,1>(((MatrixContainer<ValueType>*)ptr)->csr, x, y);
        SPC5_1rVc_Spmv<ValueType>(((MatrixContainer<ValueType>*)ptr)->csr, x, y);
        break;
    case SPC5_FORMAT_2rVc:
        SPC5_2rVc_Spmv<ValueType>(((MatrixContainer<ValueType>*)ptr)->csr, x, y);
        break;
    case SPC5_FORMAT_4rVc:
        SPC5_4rVc_Spmv<ValueType>(((MatrixContainer<ValueType>*)ptr)->csr, x, y);
        break;
    default:
        printf("[SPC5] Invalid SPC5 format %d (line %d file %s)\n", ((MatrixContainer<ValueType>*)ptr)->format, __LINE__, __FILE__);
        exit(-1);
    }
}

void spc5_spmv_float(void* ptr, const float* x, float* y){
    spc5_spmv_core<float>(ptr, x, y);
}

void spc5_spmv_double(void* ptr, const double* x, double* y){
    spc5_spmv_core<double>(ptr, x, y);
}

int spc5_blockcount_float(void* ptr){
    return (((MatrixContainer<float>*)ptr)->csr.numberOfBlocks);
}

int spc5_blockcount_double(void* ptr){
    return (((MatrixContainer<double>*)ptr)->csr.numberOfBlocks);
}

#ifdef _OPENMP

template <class ValueType>
void spc5_spmv_omp_core(void* ptr, const ValueType* x, ValueType* y){
    switch(((MatrixContainer<ValueType>*)ptr)->format){
    case SPC5_FORMAT_1rVc_WT:
        SPC5_1rVc_Spmv_omp<ValueType>(((MatrixContainer<ValueType>*)ptr)->csr, x, y, ((MatrixContainer<ValueType>*)ptr)->intervals);
        break;
    case SPC5_FORMAT_2rVc:
        SPC5_2rVc_Spmv_omp<ValueType>(((MatrixContainer<ValueType>*)ptr)->csr, x, y, ((MatrixContainer<ValueType>*)ptr)->intervals);
        break;
    case SPC5_FORMAT_4rVc:
        SPC5_4rVc_Spmv_omp<ValueType>(((MatrixContainer<ValueType>*)ptr)->csr, x, y, ((MatrixContainer<ValueType>*)ptr)->intervals);
        break;
    default:
        printf("[SPC5] Invalid SPC5 format %d (line %d file %s)\n", ((MatrixContainer<ValueType>*)ptr)->format, __LINE__, __FILE__);
        exit(-1);
    }
}

void spc5_spmv_float_omp(void* ptr, const float* x, float* y){
    spc5_spmv_omp_core<float>(ptr, x, y);
}

void spc5_spmv_double_omp(void* ptr, const double* x, double* y){
    spc5_spmv_omp_core<double>(ptr, x, y);
}
#endif

