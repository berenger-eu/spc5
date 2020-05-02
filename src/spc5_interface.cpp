#include "spc5_interface.h"

#include "spc5.hpp"

#include <memory>
#include <vector>

template <class ValueType>
struct MatrixContainer{
    SPC5Mat<ValueType> csr;
    #ifdef _OPENMP
    std::vector<ThreadInterval<ValueType>> intervals;
    #endif
};

template <class ValueType>
void* spc5_init_t(const int rows, const int cols, const int nnz, const int *row_indx, const int * col_indx, const ValueType *values){
    std::unique_ptr<Ijv<ValueType>[]> values_ijv(new Ijv<ValueType>[nnz]);

    for(int idxVal = 0 ; idxVal < nnz ; ++idxVal){
        values_ijv[idxVal].i = row_indx[idxVal];
        values_ijv[idxVal].j = col_indx[idxVal];
        values_ijv[idxVal].v = values[idxVal];
    }

    MatrixContainer<ValueType>* container = new MatrixContainer<ValueType>;
    container->csr = (COO_to_SPC5_1rVc<ValueType>(rows, cols, values_ijv.get(), nnz));

#ifdef _OPENMP
    container->intervals = SPC5_1rVc_split_omp<ValueType>(container->csr);
#endif

    return container;
}


void* spc5_init_float(const int rows, const int cols, const int nnz, const int *row_indx, const int * col_indx, const float *values){
    return spc5_init_t<float>(rows, cols, nnz, row_indx, col_indx, values);
}

void* spc5_init_double(const int rows, const int cols, const int nnz, const int *row_indx, const int * col_indx, const double *values){
    return spc5_init_t<double>(rows, cols, nnz, row_indx, col_indx, values);
}

void spc5_free_float(void* ptr){
    delete ((MatrixContainer<float>*)ptr);
}

void spc5_free_double(void* ptr){
    delete ((MatrixContainer<double>*)ptr);
}

void spc5_spmv_float(void* ptr, const float* x, float* y){
    SPC5_1rVc_Spmv<float>(((MatrixContainer<float>*)ptr)->csr, x, y);
}

void spc5_spmv_double(void* ptr, const double* x, double* y){
    SPC5_1rVc_Spmv<double>(((MatrixContainer<double>*)ptr)->csr, x, y);
}

#ifdef _OPENMP
void spc5_spmv_float_omp(void* ptr, const float* x, float* y){
    SPC5_1rVc_Spmv_omp<float>(((MatrixContainer<float>*)ptr)->csr, x, y, ((MatrixContainer<float>*)ptr)->intervals);
}

void spc5_spmv_double_omp(void* ptr, const double* x, double* y){
    SPC5_1rVc_Spmv_omp<double>(((MatrixContainer<double>*)ptr)->csr, x, y, ((MatrixContainer<double>*)ptr)->intervals);
}
#endif

//////////////////////////////////////////////////////////////////////////////
/// Redefine New operators to have aligned memory
//////////////////////////////////////////////////////////////////////////////

#include <cstdlib>
#include <new>

// Regular scalar new
void* operator new(std::size_t n);
void* operator new[]( std::size_t n );
void* operator new  ( std::size_t n, const std::nothrow_t& tag);
void* operator new[] ( std::size_t n, const std::nothrow_t& tag);

// Regular scalar delete
void operator delete(void* p) noexcept;
void operator delete[](void* p) noexcept;
void operator delete  ( void* p, const std::nothrow_t& /*tag*/);
void operator delete[]( void* p, const std::nothrow_t& /*tag*/);

// Default alignement for the complete application by redirecting the new operator
static const int DefaultMemAlignement = 64;

namespace aligned_malloc {

template <std::size_t AlignementValue>
inline void* malloc(const std::size_t inSize){
    if(inSize == 0){
        return nullptr;
    }

    // Ensure it is a power of 2
    static_assert(AlignementValue != 0 && ((AlignementValue-1)&AlignementValue) == 0, "Alignement must be a power of 2");
    // We will need to store the adress of the real blocks
    const std::size_t sizeForAddress = (AlignementValue < sizeof(unsigned char*)? sizeof(unsigned char*) : AlignementValue);

    unsigned char* allocatedMemory      = reinterpret_cast<unsigned char*>(std::malloc(inSize + AlignementValue-1 + sizeForAddress));
    unsigned char* alignedMemoryAddress = reinterpret_cast<unsigned char*>((reinterpret_cast<std::size_t>(allocatedMemory) + AlignementValue-1 + sizeForAddress) & ~static_cast<std::size_t>(AlignementValue-1));
    unsigned char* ptrForAddress        = (alignedMemoryAddress - sizeof(unsigned char*));

    // Save allocated adress
    *reinterpret_cast<unsigned char**>(ptrForAddress) = allocatedMemory;
    // Return aligned address
    return reinterpret_cast<void*>(alignedMemoryAddress);
}

inline void free(void* ptrToFree){
    if( ptrToFree ){
        unsigned char** storeRealAddress = reinterpret_cast<unsigned char**>(reinterpret_cast<unsigned char*>(ptrToFree) - sizeof(unsigned char*));
        std::free(*storeRealAddress);
    }
}
}


// Regular scalar new
void* operator new(std::size_t n) {
    void* const allocated = aligned_malloc::malloc<DefaultMemAlignement>(n);
    if(allocated){
        return allocated;
    }
    throw std::bad_alloc();
    return allocated;
}

void* operator new[]( std::size_t n ) {
    void* const allocated = aligned_malloc::malloc<DefaultMemAlignement>(n);
    if(allocated){
        return allocated;
    }
    throw std::bad_alloc();
    return allocated;
}

void* operator new  ( std::size_t n, const std::nothrow_t& /*tag*/){
    void* const allocated = aligned_malloc::malloc<DefaultMemAlignement>(n);
    return allocated;
}

void* operator new[] ( std::size_t n, const std::nothrow_t& /*tag*/){
    void* const allocated = aligned_malloc::malloc<DefaultMemAlignement>(n);
    return allocated;
}

// Regular scalar delete
void operator delete(void* p)  noexcept {
    aligned_malloc::free(p);
}

void operator delete[](void* p) noexcept {
    aligned_malloc::free(p);
}

void operator delete  ( void* p, const std::nothrow_t& /*tag*/) {
    aligned_malloc::free(p);
}

void operator delete[]( void* p, const std::nothrow_t& /*tag*/) {
    aligned_malloc::free(p);
}

