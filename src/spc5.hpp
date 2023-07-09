#ifndef SPC5_HPP
#define SPC5_HPP

#include <memory>       // For unique_ptr
#include <algorithm>    // For sort
#include <cassert>      // For assert function
#include <limits>       // For <int>::max
#include <vector>       // For vector class
#include <stdexcept>    // For invalid argument
#include <cstring>


//////////////////////////////////////////////////////////////////////////////
/// Matrix structure
//////////////////////////////////////////////////////////////////////////////

enum SPC5_MATRIX_TYPE{
    UNDEFINED_FORMAT,
    FORMAT_CSR,
    FORMAT_1rVc_WT,
    FORMAT_2rVc,
    FORMAT_4rVc,
    FORMAT_8rVc
};

enum SPC5_VEC_PADDING {
    SPC5_VEC_PADDING_X = 15,
    SPC5_VEC_PADDING_Y = 7
};

template <class ValueType>
struct SPC5Mat_Mask;

template <>
struct SPC5Mat_Mask<double>{
    using type = unsigned char;
};

template <>
struct SPC5Mat_Mask<float>{
    using type = unsigned short;
};

template <class ValueType>
struct SPC5Mat {
    SPC5_MATRIX_TYPE format;

    // For the CSR format
    int numberOfRows;  //< the number of rows of the matrix
    int numberOfColumns;  //< the number of columns of the matrix
    int numberOfNNZ;    //< the number of numberOfNNZ (== rowsSize[numberOfRows])
    std::unique_ptr<ValueType[]> values;  //< the values (of size numberOfNNZ)
    std::unique_ptr<int[]> rowsSize;//< the usual "rowsSize/rowptr" (of size numberOfRows+1)
    std::unique_ptr<int[]> valuesColumnIndexes;//< the colidx of each NNZ (of size numberOfNNZ)

    // For the SPC5 format
    std::unique_ptr<unsigned char[]> blocksColumnIndexesWithMasks;// Specific for each storage
    int numberOfBlocks;

    // Default constructor
    SPC5Mat() :format(UNDEFINED_FORMAT), numberOfRows(0), numberOfColumns(0), numberOfNNZ(0), numberOfBlocks(0){}
};

//////////////////////////////////////////////////////////////////////////////
/// CSR
//////////////////////////////////////////////////////////////////////////////

// Useful to convert from COO/IJV to CSR
template <class ValueType>
struct Ijv{
    int i;
    int j;
    ValueType v;
};


template <class ValueType>
inline SPC5Mat<ValueType> COO_sorted_to_CSR(const int nbRows, const int nbCols,
                                      const Ijv<ValueType> values[], int nbValues){
    SPC5Mat<ValueType> csr;

    csr.format = SPC5_MATRIX_TYPE::FORMAT_CSR;

    csr.numberOfRows = nbRows;
    csr.numberOfColumns = nbCols;
    csr.numberOfNNZ = nbValues;
    csr.values.reset(new ValueType[nbValues]());
    csr.rowsSize.reset(new int[csr.numberOfRows+1]());
    csr.valuesColumnIndexes.reset(new int[csr.numberOfNNZ]());

    for(int idxElement = 0 ; idxElement < nbValues ; ++idxElement){
        csr.values[idxElement] = values[idxElement].v;
        csr.valuesColumnIndexes[idxElement] = values[idxElement].j;
        csr.rowsSize[values[idxElement].i+1] += 1;
    }

    for(int idxRow = 2 ; idxRow <= csr.numberOfRows ; ++idxRow){
        csr.rowsSize[idxRow] += csr.rowsSize[idxRow-1];
    }
    assert(csr.rowsSize[0] == 0);
    assert(csr.rowsSize[csr.numberOfRows] == csr.numberOfNNZ);

    return csr;
}

// values cannot be const as it will be sorted in row major first
template <class ValueType>
inline SPC5Mat<ValueType> COO_unsorted_to_CSR(const int nbRows, const int nbCols,
                                        Ijv<ValueType> values[], int nbValues){
    std::sort(&values[0], &values[nbValues], [](const Ijv<ValueType>& v1, const Ijv<ValueType>& v2){
        return v1.i < v2.i || (v1.i == v2.i && v1.j < v2.j);
    });
    return COO_sorted_to_CSR(nbRows, nbCols, values, nbValues);
}

template <class ValueType>
inline void CSR_Spmv_scalar(const SPC5Mat<ValueType>& csr, const ValueType x[], ValueType y[]){
    assert(csr.format == SPC5_MATRIX_TYPE::FORMAT_CSR);

    for(int idxRow = 0 ; idxRow < csr.numberOfRows ; ++idxRow){
        ValueType sum = 0;
        for(int idxVal = csr.rowsSize[idxRow] ; idxVal < csr.rowsSize[idxRow+1] ; ++idxVal){
            sum += x[csr.valuesColumnIndexes[idxVal]] * csr.values[idxVal];
        }
        y[idxRow] += sum;
    }
}


template <class ValueType, class FuncType>
inline void CSR_iterate(SPC5Mat<ValueType>& csr, const FuncType&& func){
    assert(csr->format == SPC5_MATRIX_TYPE::FORMAT_CSR);

    for(int idxRow = 0 ; idxRow < csr.numberOfRows ; ++idxRow){
        for(int idxVal = csr.rowsSize[idxRow] ; idxVal < csr.rowsSize[idxRow+1] ; ++idxVal){
            func(idxRow, csr.valuesColumnIndexes[idxVal], csr.values[idxVal]);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
/// Utils
//////////////////////////////////////////////////////////////////////////////

template <class ValueType>
struct ValPerVec{
    static const int size = int(512/(8*sizeof(ValueType)));
};


template <class ValueType>
std::unique_ptr<ValueType[]> ToUniquePtr(const std::vector<ValueType>& values){
    std::unique_ptr<ValueType[]> ptr(new ValueType[values.size()]);
    std::copy(values.begin(), values.end(), ptr.get());
    return ptr;
}



//////////////////////////////////////////////////////////////////////////////
/// Convertion functions
//////////////////////////////////////////////////////////////////////////////


template <class ValueType, int nbRowsPerBlock>
inline void core_CSR_to_SPC5_rVc(SPC5Mat<ValueType>* csr){
    assert(csr->format == SPC5_MATRIX_TYPE::FORMAT_CSR);
    std::vector<unsigned char> blocks;
    blocks.reserve((sizeof(int)+sizeof(short))*csr->numberOfNNZ/ValPerVec<ValueType>::size);

    std::unique_ptr<ValueType[]> newValues(new ValueType[csr->numberOfNNZ]);
    int globalIdxValues = 0;

    int previousNbBlocks = 0;
    for(int idxRow = 0 ; idxRow < csr->numberOfRows ; idxRow += nbRowsPerBlock){
        int currentNbBlocks = 0;
        int idxVal[nbRowsPerBlock] = {0};
        for(int idxSubRow = 0 ; idxSubRow < nbRowsPerBlock ; ++idxSubRow){
            if(idxRow + idxSubRow < csr->numberOfRows){
                idxVal[idxSubRow] = csr->rowsSize[idxRow+idxSubRow];
            }
        }

        int idxCptVal = 0;

        while(true){
            bool hasWork = false;
            int idxCol = std::numeric_limits<int>::max();
            for(int idxSubRow = 0 ; idxSubRow < nbRowsPerBlock ; ++idxSubRow){
                if(idxRow + idxSubRow < csr->numberOfRows
                        && idxVal[idxSubRow] < csr->rowsSize[idxRow+idxSubRow+1]){
                    hasWork = true;
                    idxCol = std::min(idxCol, csr->valuesColumnIndexes[idxVal[idxSubRow]]);
                }
            }
            if(hasWork == false){
                break;
            }


            typename SPC5Mat_Mask<ValueType>::type valMask[nbRowsPerBlock] = {0u};
            for(int idxSubRow = 0 ; idxSubRow < nbRowsPerBlock ; ++idxSubRow){
                if(idxRow + idxSubRow < csr->numberOfRows){
                    while(idxVal[idxSubRow] < csr->rowsSize[idxRow+idxSubRow+1]
                          && csr->valuesColumnIndexes[idxVal[idxSubRow]] < idxCol+ValPerVec<ValueType>::size){
                        assert(globalIdxValues < csr->numberOfNNZ);
                        newValues[globalIdxValues++] = csr->values[idxVal[idxSubRow]];
                        valMask[idxSubRow] |= typename SPC5Mat_Mask<ValueType>::type(1u << (csr->valuesColumnIndexes[idxVal[idxSubRow]]-idxCol));
                        idxVal[idxSubRow] += 1;
                        idxCptVal += 1;
                    }
                }
            }

            blocks.insert(blocks.end(), (unsigned char*)&idxCol, (unsigned char*)(&idxCol+1));
            blocks.insert(blocks.end(), (unsigned char*)&valMask[0], (unsigned char*)(&valMask[nbRowsPerBlock]));
            currentNbBlocks += 1;
        }

        csr->rowsSize[idxRow/nbRowsPerBlock] = previousNbBlocks;
        previousNbBlocks += currentNbBlocks;
    }

    csr->numberOfBlocks = previousNbBlocks;
    csr->values = std::move(newValues);
    csr->rowsSize[((csr->numberOfRows-1)/nbRowsPerBlock)+1] = previousNbBlocks;
    csr->blocksColumnIndexesWithMasks = (ToUniquePtr(blocks));
}


template <class ValueType, int nbRowsPerBlock>
inline void core_SPC5_rVc_Spmv_scalar(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[]){
    int idxVal = 0;
    for(int idxRow = 0 ; idxRow < mat.numberOfRows ; idxRow += nbRowsPerBlock){
        ValueType sum[nbRowsPerBlock] = {0};
        for(int idxBlock = mat.rowsSize[idxRow]; idxBlock < mat.rowsSize[idxRow+1] ; ++idxBlock){
            const int idxCol = *(int*)&mat.blocksColumnIndexesWithMasks[idxBlock*(sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*nbRowsPerBlock)];
            for(int idxRowBlock = 0 ; idxRowBlock < nbRowsPerBlock ; idxRowBlock += 1){
                const typename SPC5Mat_Mask<ValueType>::type valMask = *(typename SPC5Mat_Mask<ValueType>::type*)&mat.blocksColumnIndexesWithMasks[idxBlock*(sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*nbRowsPerBlock) + sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*(idxRowBlock)];
                for(int idxvv = 0 ; idxvv < ValPerVec<ValueType>::size ; ++idxvv){
                    if((1 << idxvv) & valMask){
                        sum[idxRowBlock] += x[idxCol+idxvv] * mat.values[idxVal];
                        idxVal += 1;
                    }
                }
            }
        }
        for(int idxRowBlock = 0 ; idxRowBlock < nbRowsPerBlock ; ++idxRowBlock){
            y[idxRow+idxRowBlock] += sum[idxRowBlock];
        }
    }
}


template <class ValueType, int nbRowsPerBlock, class FuncType>
inline void core_SPC5_rVc_iterate(SPC5Mat<ValueType>& mat, const FuncType&& func){
    int idxVal = 0;
    for(int idxRow = 0 ; idxRow < mat.numberOfRows ; idxRow += nbRowsPerBlock){
        const int idxRowBlock = idxRow/nbRowsPerBlock;
        for(int idxBlock = mat.rowsSize[idxRowBlock]; idxBlock < mat.rowsSize[idxRowBlock+1] ; ++idxBlock){
            const int idxCol = *(int*)&mat.blocksColumnIndexesWithMasks[idxBlock*(sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*nbRowsPerBlock)];
            for(int idxRowBlockp = 0 ; idxRowBlockp < nbRowsPerBlock ; idxRowBlockp += 1){
                const typename SPC5Mat_Mask<ValueType>::type valMask = *(typename SPC5Mat_Mask<ValueType>::type*)&mat.blocksColumnIndexesWithMasks[idxBlock*(sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*nbRowsPerBlock) + sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*(idxRowBlock)];
                for(int idxvv = 0 ; idxvv < ValPerVec<ValueType>::size ; ++idxvv){
                    if((1 << idxvv) & valMask){
                        func(idxRow+idxRowBlockp, idxCol+idxvv, mat.values[idxVal]);
                        idxVal += 1;
                    }
                }
            }
        }
    }
}


template <class ValueType, int nbRowsInBlock, int nbColsInBlock>
int core_SPC5_block_count(const SPC5Mat<ValueType>& csr){
    assert(csr.format == SPC5_MATRIX_TYPE::FORMAT_CSR);
    int nbBlocks = 0;

    for(int idxRow = 0 ; idxRow < csr.numberOfRows ; idxRow += nbRowsInBlock){
        int idxVal[nbRowsInBlock] = {0};
        for(int idxSubRow = 0 ; idxSubRow < nbRowsInBlock ; ++idxSubRow){
            if(idxRow + idxSubRow < csr.numberOfRows){
                idxVal[idxSubRow] = csr.rowsSize[idxRow+idxSubRow];
            }
        }

        while(true){
            bool hasWork = false;
            int idxCol = std::numeric_limits<int>::max();
            for(int idxSubRow = 0 ; idxSubRow < nbRowsInBlock ; ++idxSubRow){
                if(idxRow + idxSubRow < csr.numberOfRows
                        && idxVal[idxSubRow] < csr.rowsSize[idxRow+idxSubRow+1]){
                    hasWork = true;
                    idxCol = std::min(idxCol, csr.valuesColumnIndexes[idxVal[idxSubRow]]);
                }
            }
            if(hasWork == false){
                break;
            }

            nbBlocks += 1;

            for(int idxSubRow = 0 ; idxSubRow < nbRowsInBlock ; ++idxSubRow){
                if(idxRow + idxSubRow < csr.numberOfRows){
                    while(idxVal[idxSubRow] < csr.rowsSize[idxRow+idxSubRow+1]
                          && csr.valuesColumnIndexes[idxVal[idxSubRow]] < idxCol+nbColsInBlock){
                        idxVal[idxSubRow] += 1;
                    }
                }
            }
        }
    }

    return nbBlocks;
}



//////////////////////////////////////////////////////////////////////////////
/// OpenMP workload division
//////////////////////////////////////////////////////////////////////////////

#ifdef _OPENMP

#include <omp.h>
#include <cstring>
#include <cmath>

template <class ValueType>
struct ThreadInterval{
    int startingRow;
    int numberOfRows;
    std::unique_ptr<ValueType[]> threadY;

    std::unique_ptr<ValueType[]> threadValues;
    std::unique_ptr<int[]> threadRowsSize;
    std::unique_ptr<int[]> threadValuesColumnIndexes;
    std::unique_ptr<unsigned char[]> threadBlocksColumnIndexesWithMasks;

    int valuesOffset;
};


template <class ValueType, int nbRowsPerBlock>
inline std::vector<ThreadInterval<ValueType>> core_SPC5_rVc_threadsplit(const SPC5Mat<ValueType>& mat, const int nbThreads){
    std::vector<ThreadInterval<ValueType>> intervals(nbThreads);
    int idxCurrentThread = 0;
    const double blocksPerThreads = double(mat.numberOfBlocks)/double(nbThreads);
    intervals[0].startingRow = 0;
    intervals[0].valuesOffset = 0;

    int idxVal = 0;
    for(int idxRow = 0 ; idxRow < mat.numberOfRows ; idxRow += nbRowsPerBlock){
        if(idxCurrentThread != nbThreads-1 && idxRow // not the last thread
                && std::abs((double(idxCurrentThread+1)*blocksPerThreads)-mat.rowsSize[idxRow/nbRowsPerBlock])
                    < std::abs((double(idxCurrentThread+1)*blocksPerThreads)-mat.rowsSize[idxRow/nbRowsPerBlock+1])){
            intervals[idxCurrentThread].numberOfRows = idxRow - intervals[idxCurrentThread].startingRow;

            idxCurrentThread += 1;
            intervals[idxCurrentThread].startingRow = idxRow;
            intervals[idxCurrentThread].valuesOffset = idxVal;
        }

        for(int idxBlock = mat.rowsSize[idxRow/nbRowsPerBlock]; idxBlock < mat.rowsSize[idxRow/nbRowsPerBlock+1] ; ++idxBlock){
            //const int idxCol = *(int*)&mat.blocksColumnIndexesWithMasks[idxBlock*(sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*nbRowsPerBlock)];
            for(int idxRowBlock = 0 ; idxRowBlock < nbRowsPerBlock ; idxRowBlock += 1){
                const typename SPC5Mat_Mask<ValueType>::type valMask = *(typename SPC5Mat_Mask<ValueType>::type*)&mat.blocksColumnIndexesWithMasks[idxBlock*(sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*nbRowsPerBlock) + sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*(idxRowBlock)];
                for(int idxvv = 0 ; idxvv < ValPerVec<ValueType>::size ; ++idxvv){
                    if((1 << idxvv) & valMask){
                        idxVal += 1;
                    }
                }
            }
        }
    }


    intervals[idxCurrentThread].numberOfRows = mat.numberOfRows - intervals[idxCurrentThread].startingRow;
    idxCurrentThread += 1;
    while(idxCurrentThread != nbThreads){
        intervals[idxCurrentThread].startingRow = mat.numberOfRows;
        intervals[idxCurrentThread].valuesOffset = idxVal;
        intervals[idxCurrentThread].numberOfRows = 0;
        idxCurrentThread += 1;
    }

#pragma omp parallel num_threads(nbThreads)
    {
        const int idxThread = omp_get_thread_num();
        intervals[idxThread].threadY.reset(new ValueType[intervals[idxThread].numberOfRows + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y]());

        intervals[idxThread].threadRowsSize.reset(new int[intervals[idxThread].numberOfRows + 1]());
        memcpy(intervals[idxThread].threadRowsSize.get(),
                mat.rowsSize.get()+intervals[idxThread].startingRow/nbRowsPerBlock,
                sizeof(int)*(intervals[idxThread].numberOfRows/nbRowsPerBlock + 1));

        const int nbBlocks = mat.rowsSize[intervals[idxThread].startingRow/nbRowsPerBlock + intervals[idxThread].numberOfRows]
                           - mat.rowsSize[intervals[idxThread].startingRow/nbRowsPerBlock];
        const int maskSize = (sizeof(ValueType)==4?2:1);

        intervals[idxThread].threadBlocksColumnIndexesWithMasks.reset(new unsigned char[nbBlocks*(4+maskSize*nbRowsPerBlock)]());
        memcpy(intervals[idxThread].threadBlocksColumnIndexesWithMasks.get(),
                mat.blocksColumnIndexesWithMasks.get()+mat.rowsSize[intervals[idxThread/nbRowsPerBlock].startingRow]*(4+maskSize*nbRowsPerBlock),
                sizeof(unsigned char)*(nbBlocks*(4+maskSize*nbRowsPerBlock)));

        const int nbValues = (idxThread+1 != nbThreads ? intervals[idxThread+1].valuesOffset : idxVal)
                - intervals[idxThread].valuesOffset;

        intervals[idxThread].threadValues.reset(new ValueType[nbValues]());
        memcpy(intervals[idxThread].threadValues.get(),
                mat.values.get()+intervals[idxThread].valuesOffset,
                sizeof(ValueType)*nbValues);

    }

    return intervals;
}


template <class ValueType>
void SPC5_opti_merge(ValueType dest[], const ValueType src[], const int nbValues);


extern "C" void SPC5_opti_merge_double(double dest[], const double src[], const int nbValues);
extern "C" void SPC5_opti_merge_float(float dest[], const float src[], const int nbValues);

template <>
inline void SPC5_opti_merge<double>(double dest[], const double src[], const int nbValues){
    SPC5_opti_merge_double(dest, src, nbValues);
}

template <>
inline void SPC5_opti_merge<float>(float dest[], const float src[], const int nbValues){
    SPC5_opti_merge_float(dest, src, nbValues);
}


#endif

//////////////////////////////////////////////////////////////////////////////
/// SPC5_1rVc
//////////////////////////////////////////////////////////////////////////////


template <class ValueType>
inline void CSR_to_SPC5_1rVc(SPC5Mat<ValueType>* csr){
    assert(csr->format == SPC5_MATRIX_TYPE::FORMAT_CSR);
    csr->format = SPC5_MATRIX_TYPE::FORMAT_1rVc_WT;

    std::vector<unsigned char> blocks;
    blocks.reserve((sizeof(int)+sizeof(short))*csr->numberOfNNZ/ValPerVec<ValueType>::size);

    int previousNbBlocks = 0;
    for(int idxRow = 0 ; idxRow < csr->numberOfRows ; ++idxRow){
        int currentNbBlocks = 0;
        int idxVal = csr->rowsSize[idxRow];
        while(idxVal < csr->rowsSize[idxRow+1]){
            int idxCol = csr->valuesColumnIndexes[idxVal];
            typename SPC5Mat_Mask<ValueType>::type valMask = 1u;
            idxVal += 1;
            while(idxVal < csr->rowsSize[idxRow+1] && csr->valuesColumnIndexes[idxVal] < idxCol+ValPerVec<ValueType>::size){
                valMask |= typename SPC5Mat_Mask<ValueType>::type(static_cast<typename SPC5Mat_Mask<ValueType>::type>(1u) << (csr->valuesColumnIndexes[idxVal]-idxCol));
                idxVal += 1;
            }

            blocks.insert(blocks.end(), (unsigned char*)&idxCol, (unsigned char*)(&idxCol+1));
            blocks.insert(blocks.end(), (unsigned char*)&valMask, (unsigned char*)(&valMask+1));
            currentNbBlocks += 1;
        }

        csr->rowsSize[idxRow] = previousNbBlocks;
        previousNbBlocks += currentNbBlocks;
    }

    csr->numberOfBlocks = previousNbBlocks;
    csr->rowsSize[csr->numberOfRows] = previousNbBlocks;
    csr->blocksColumnIndexesWithMasks = (ToUniquePtr(blocks));
}

template <class ValueType>
inline SPC5Mat<ValueType> COO_to_SPC5_1rVc(const int nbRows, const int nbCols,
                                    Ijv<ValueType> values[], int nbValues){
    SPC5Mat<ValueType> mat = COO_unsorted_to_CSR(nbRows, nbCols, values, nbValues);
    CSR_to_SPC5_1rVc(&mat);
    return mat;
}

template <class ValueType>
inline void SPC5_1rVc_Spmv_scalar(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[]){
    assert(mat->format == SPC5_MATRIX_TYPE::FORMAT_1rVc_WT);
    core_SPC5_rVc_Spmv_scalar<ValueType,1>(mat, x, y);
}

template <class ValueType, class FuncType>
inline void SPC5_1rVc_iterate(const SPC5Mat<ValueType>& mat, FuncType&& func){
    assert(mat->format == SPC5_MATRIX_TYPE::FORMAT_1rVc_WT);
    core_SPC5_rVc_iterate<ValueType,1>(mat, std::forward<FuncType>(func));
}

template <class ValueType>
inline int SPC5_1rVc_block_count(const SPC5Mat<ValueType>& csr){
    return core_SPC5_block_count<ValueType,1,ValPerVec<ValueType>::size>(csr);
}


extern "C" void core_SPC5_1rVc_Spmv_double(const long int nbRows, const int* rowsSizes,
                                   const unsigned char* blocksColumnIndexesWithMasks,
                                                   const double* values,
                                                   const double* x, double* y);
extern "C" void core_SPC5_1rVc_Spmv_float(const long int nbRows, const int* rowsSizes,
                                   const unsigned char* blocksColumnIndexesWithMasks,
                                                   const float* values,
                                                   const float* x, float* y);

template <class ValueType>
inline void SPC5_1rVc_Spmv(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[]);

template <>
inline void SPC5_1rVc_Spmv<double>(const SPC5Mat<double>& mat, const double x[], double y[]){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_1rVc_WT);
    core_SPC5_1rVc_Spmv_double(mat.numberOfRows, mat.rowsSize.get(),
                                   mat.blocksColumnIndexesWithMasks.get(), mat.values.get(),
                                   x, y);
}

template <>
inline void SPC5_1rVc_Spmv<float>(const SPC5Mat<float>& mat, const float x[], float y[]){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_1rVc_WT);
    core_SPC5_1rVc_Spmv_float(mat.numberOfRows, mat.rowsSize.get(),
                                   mat.blocksColumnIndexesWithMasks.get(), mat.values.get(),
                                   x, y);
}

#ifdef _OPENMP

template <class ValueType>
inline std::vector<ThreadInterval<ValueType>> SPC5_1rVc_split_omp(const SPC5Mat<ValueType>& mat, const int numThreads){
    return core_SPC5_rVc_threadsplit<ValueType,1>(mat, numThreads);
}

template <class ValueType>
inline std::vector<ThreadInterval<ValueType>> SPC5_1rVc_split_omp(const SPC5Mat<ValueType>& mat){
    return core_SPC5_rVc_threadsplit<ValueType,1>(mat, omp_get_max_threads());
}


template <class ValueType>
inline void SPC5_1rVc_Spmv_omp(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[],
                               const std::vector<ThreadInterval<ValueType>>& threadsVecs);

template <>
inline void SPC5_1rVc_Spmv_omp<double>(const SPC5Mat<double>& mat, const double x[], double y[],
                                   const std::vector<ThreadInterval<double>>& threadsVecs){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_1rVc_WT);
    const int numThreads = int(threadsVecs.size());
#pragma omp parallel num_threads(numThreads)
    {
        memset(threadsVecs[omp_get_thread_num()].threadY.get(), 0, sizeof(double)*threadsVecs[omp_get_thread_num()].numberOfRows);

        core_SPC5_1rVc_Spmv_double(threadsVecs[omp_get_thread_num()].numberOfRows,
                                       threadsVecs[omp_get_thread_num()].threadRowsSize.get(),
                                       threadsVecs[omp_get_thread_num()].threadBlocksColumnIndexesWithMasks.get(),
                                       threadsVecs[omp_get_thread_num()].threadValues.get(),
                                       x, threadsVecs[omp_get_thread_num()].threadY.get());

        SPC5_opti_merge(&y[threadsVecs[omp_get_thread_num()].startingRow], threadsVecs[omp_get_thread_num()].threadY.get(),
                threadsVecs[omp_get_thread_num()].numberOfRows);
    }
}

template <>
inline void SPC5_1rVc_Spmv_omp<float>(const SPC5Mat<float>& mat, const float x[], float y[],
                                  const std::vector<ThreadInterval<float>>& threadsVecs){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_1rVc_WT);
    const int numThreads = int(threadsVecs.size());
#pragma omp parallel num_threads(numThreads)
    {
        memset(threadsVecs[omp_get_thread_num()].threadY.get(), 0, sizeof(float)*threadsVecs[omp_get_thread_num()].numberOfRows);

        core_SPC5_1rVc_Spmv_float(threadsVecs[omp_get_thread_num()].numberOfRows,
                                       threadsVecs[omp_get_thread_num()].threadRowsSize.get(),
                                       threadsVecs[omp_get_thread_num()].threadBlocksColumnIndexesWithMasks.get(),
                                       threadsVecs[omp_get_thread_num()].threadValues.get(),
                                       x, threadsVecs[omp_get_thread_num()].threadY.get());

        SPC5_opti_merge(&y[threadsVecs[omp_get_thread_num()].startingRow], threadsVecs[omp_get_thread_num()].threadY.get(),
                threadsVecs[omp_get_thread_num()].numberOfRows);
    }
}


#endif


//////////////////////////////////////////////////////////////////////////////
/// 2rVc => 2 rows, VEC/2 columns
//////////////////////////////////////////////////////////////////////////////

template <class ValueType>
inline void CSR_to_SPC5_2rVc(SPC5Mat<ValueType>* csr){
    core_CSR_to_SPC5_rVc<ValueType, 2>(csr);
    csr->format = SPC5_MATRIX_TYPE::FORMAT_2rVc;
}

template <class ValueType>
inline SPC5Mat<ValueType> COO_to_SPC5_2rVc(const int nbRows, const int nbCols,
                                    Ijv<ValueType> values[], int nbValues){
    SPC5Mat<ValueType> mat = COO_unsorted_to_CSR(nbRows, nbCols, values, nbValues);
    CSR_to_SPC5_2rVc(&mat);
    return mat;
}

template <class ValueType>
inline void SPC5_2rVc_Spmv_scalar(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[]){
    assert(mat->format == SPC5_MATRIX_TYPE::FORMAT_2rVc);
    core_SPC5_rVc_Spmv_scalar<ValueType,2>(mat, x, y);
}

template <class ValueType, class FuncType>
inline void SPC5_2rVc_iterate(const SPC5Mat<ValueType>& mat, FuncType&& func){
    assert(mat->format == SPC5_MATRIX_TYPE::FORMAT_2rVc);
    core_SPC5_rVc_iterate<ValueType,2>(mat, std::forward<FuncType>(func));
}

template <class ValueType>
inline int SPC5_2rVc_block_count(const SPC5Mat<ValueType>& csr){
    return core_SPC5_block_count<ValueType,2,ValPerVec<ValueType>::size>(csr);
}


extern "C" void core_SPC5_2rVc_Spmv_double(const long int nbRows, const int* rowsSizes,
                                   const unsigned char* blocksColumnIndexesWithMasks,
                                                   const double* values,
                                                   const double* x, double* y);
extern "C" void core_SPC5_2rVc_Spmv_float(const long int nbRows, const int* rowsSizes,
                                   const unsigned char* blocksColumnIndexesWithMasks,
                                                   const float* values,
                                                   const float* x, float* y);

template <class ValueType>
inline void SPC5_2rVc_Spmv(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[]);

template <>
inline void SPC5_2rVc_Spmv<double>(const SPC5Mat<double>& mat, const double x[], double y[]){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_2rVc);
    core_SPC5_2rVc_Spmv_double(mat.numberOfRows, mat.rowsSize.get(),
                                   mat.blocksColumnIndexesWithMasks.get(), mat.values.get(),
                                   x, y);
}

template <>
inline void SPC5_2rVc_Spmv<float>(const SPC5Mat<float>& mat, const float x[], float y[]){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_2rVc);
    core_SPC5_2rVc_Spmv_float(mat.numberOfRows, mat.rowsSize.get(),
                                   mat.blocksColumnIndexesWithMasks.get(), mat.values.get(),
                                   x, y);
}



#ifdef _OPENMP

template <class ValueType>
inline std::vector<ThreadInterval<ValueType>> SPC5_2rVc_split_omp(const SPC5Mat<ValueType>& mat, const int numThreads){
    return core_SPC5_rVc_threadsplit<ValueType,2>(mat, numThreads);
}

template <class ValueType>
inline std::vector<ThreadInterval<ValueType>> SPC5_2rVc_split_omp(const SPC5Mat<ValueType>& mat){
    return core_SPC5_rVc_threadsplit<ValueType,2>(mat, omp_get_max_threads());
}


template <class ValueType>
inline void SPC5_2rVc_Spmv_omp(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[],
                               const std::vector<ThreadInterval<ValueType>>& threadsVecs);

template <>
inline void SPC5_2rVc_Spmv_omp<double>(const SPC5Mat<double>& mat, const double x[], double y[],
                                   const std::vector<ThreadInterval<double>>& threadsVecs){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_2rVc);
    const int numThreads = int(threadsVecs.size());
#pragma omp parallel num_threads(numThreads)
    {
        memset(threadsVecs[omp_get_thread_num()].threadY.get(), 0, sizeof(double)*threadsVecs[omp_get_thread_num()].numberOfRows);

        core_SPC5_2rVc_Spmv_double(threadsVecs[omp_get_thread_num()].numberOfRows,
                                       threadsVecs[omp_get_thread_num()].threadRowsSize.get(),
                                       threadsVecs[omp_get_thread_num()].threadBlocksColumnIndexesWithMasks.get(),
                                       threadsVecs[omp_get_thread_num()].threadValues.get(),
                                       x, threadsVecs[omp_get_thread_num()].threadY.get());

        SPC5_opti_merge(&y[threadsVecs[omp_get_thread_num()].startingRow], threadsVecs[omp_get_thread_num()].threadY.get(),
                threadsVecs[omp_get_thread_num()].numberOfRows);
    }
}

template <>
inline void SPC5_2rVc_Spmv_omp<float>(const SPC5Mat<float>& mat, const float x[], float y[],
                                  const std::vector<ThreadInterval<float>>& threadsVecs){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_2rVc);
    const int numThreads = int(threadsVecs.size());
#pragma omp parallel num_threads(numThreads)
    {
        memset(threadsVecs[omp_get_thread_num()].threadY.get(), 0, sizeof(float)*threadsVecs[omp_get_thread_num()].numberOfRows);

        core_SPC5_2rVc_Spmv_float(threadsVecs[omp_get_thread_num()].numberOfRows,
                                       threadsVecs[omp_get_thread_num()].threadRowsSize.get(),
                                       threadsVecs[omp_get_thread_num()].threadBlocksColumnIndexesWithMasks.get(),
                                       threadsVecs[omp_get_thread_num()].threadValues.get(),
                                       x, threadsVecs[omp_get_thread_num()].threadY.get());

        SPC5_opti_merge(&y[threadsVecs[omp_get_thread_num()].startingRow], threadsVecs[omp_get_thread_num()].threadY.get(),
                threadsVecs[omp_get_thread_num()].numberOfRows);
    }
}


#endif


//////////////////////////////////////////////////////////////////////////////
/// 4rVc => 4 rows, VEC columns
//////////////////////////////////////////////////////////////////////////////

template <class ValueType>
inline void CSR_to_SPC5_4rVc(SPC5Mat<ValueType>* csr){
    core_CSR_to_SPC5_rVc<ValueType, 4>(csr);
    csr->format = SPC5_MATRIX_TYPE::FORMAT_4rVc;
}

template <class ValueType>
inline SPC5Mat<ValueType> COO_to_SPC5_4rVc(const int nbRows, const int nbCols,
                                    Ijv<ValueType> values[], int nbValues){
    SPC5Mat<ValueType> mat = COO_unsorted_to_CSR(nbRows, nbCols, values, nbValues);
    CSR_to_SPC5_4rVc(&mat);
    return mat;
}

template <class ValueType>
inline void SPC5_4rVc_Spmv_scalar(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[]){
    assert(mat->format == SPC5_MATRIX_TYPE::FORMAT_4rVc);
    core_SPC5_rVc_Spmv_scalar<ValueType,4>(mat, x, y);
}

template <class ValueType, class FuncType>
inline void SPC5_4rVc_iterate(const SPC5Mat<ValueType>& mat, FuncType&& func){
    assert(mat->format == SPC5_MATRIX_TYPE::FORMAT_4rVc);
    core_SPC5_rVc_iterate<ValueType,4>(mat, std::forward<FuncType>(func));
}

template <class ValueType>
inline int SPC5_4rVc_block_count(const SPC5Mat<ValueType>& csr){
    return core_SPC5_block_count<ValueType,4,ValPerVec<ValueType>::size>(csr);
}

extern "C" void core_SPC5_4rVc_Spmv_double(const long int nbRows, const int* rowsSizes,
                                   const unsigned char* blocksColumnIndexesWithMasks,
                                                   const double* values,
                                                   const double* x, double* y);
extern "C" void core_SPC5_4rVc_Spmv_float(const long int nbRows, const int* rowsSizes,
                                   const unsigned char* blocksColumnIndexesWithMasks,
                                                   const float* values,
                                                   const float* x, float* y);

template <class ValueType>
inline void SPC5_4rVc_Spmv(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[]);

template <>
inline void SPC5_4rVc_Spmv<double>(const SPC5Mat<double>& mat, const double x[], double y[]){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_4rVc);
    core_SPC5_4rVc_Spmv_double(mat.numberOfRows, mat.rowsSize.get(),
                                   mat.blocksColumnIndexesWithMasks.get(), mat.values.get(),
                                   x, y);
}

template <>
inline void SPC5_4rVc_Spmv<float>(const SPC5Mat<float>& mat, const float x[], float y[]){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_4rVc);
    core_SPC5_4rVc_Spmv_float(mat.numberOfRows, mat.rowsSize.get(),
                                   mat.blocksColumnIndexesWithMasks.get(), mat.values.get(),
                                   x, y);
}



#ifdef _OPENMP

template <class ValueType>
inline std::vector<ThreadInterval<ValueType>> SPC5_4rVc_split_omp(const SPC5Mat<ValueType>& mat, const int numThreads){
    return core_SPC5_rVc_threadsplit<ValueType,4>(mat, numThreads);
}

template <class ValueType>
inline std::vector<ThreadInterval<ValueType>> SPC5_4rVc_split_omp(const SPC5Mat<ValueType>& mat){
    return core_SPC5_rVc_threadsplit<ValueType,4>(mat, omp_get_max_threads());
}


template <class ValueType>
inline void SPC5_4rVc_Spmv_omp(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[],
                               const std::vector<ThreadInterval<ValueType>>& threadsVecs);

template <>
inline void SPC5_4rVc_Spmv_omp<double>(const SPC5Mat<double>& mat, const double x[], double y[],
                                   const std::vector<ThreadInterval<double>>& threadsVecs){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_4rVc);
    const int numThreads = int(threadsVecs.size());
#pragma omp parallel num_threads(numThreads)
    {
        memset(threadsVecs[omp_get_thread_num()].threadY.get(), 0, sizeof(double)*threadsVecs[omp_get_thread_num()].numberOfRows);

        core_SPC5_4rVc_Spmv_double(threadsVecs[omp_get_thread_num()].numberOfRows,
                                       threadsVecs[omp_get_thread_num()].threadRowsSize.get(),
                                       threadsVecs[omp_get_thread_num()].threadBlocksColumnIndexesWithMasks.get(),
                                       threadsVecs[omp_get_thread_num()].threadValues.get(),
                                       x, threadsVecs[omp_get_thread_num()].threadY.get());

        SPC5_opti_merge(&y[threadsVecs[omp_get_thread_num()].startingRow], threadsVecs[omp_get_thread_num()].threadY.get(),
                threadsVecs[omp_get_thread_num()].numberOfRows);
    }
}

template <>
inline void SPC5_4rVc_Spmv_omp<float>(const SPC5Mat<float>& mat, const float x[], float y[],
                                  const std::vector<ThreadInterval<float>>& threadsVecs){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_4rVc);
    const int numThreads = int(threadsVecs.size());
#pragma omp parallel num_threads(numThreads)
    {
        memset(threadsVecs[omp_get_thread_num()].threadY.get(), 0, sizeof(float)*threadsVecs[omp_get_thread_num()].numberOfRows);

        core_SPC5_4rVc_Spmv_float(threadsVecs[omp_get_thread_num()].numberOfRows,
                                       threadsVecs[omp_get_thread_num()].threadRowsSize.get(),
                                       threadsVecs[omp_get_thread_num()].threadBlocksColumnIndexesWithMasks.get(),
                                       threadsVecs[omp_get_thread_num()].threadValues.get(),
                                       x, threadsVecs[omp_get_thread_num()].threadY.get());

        SPC5_opti_merge(&y[threadsVecs[omp_get_thread_num()].startingRow], threadsVecs[omp_get_thread_num()].threadY.get(),
                threadsVecs[omp_get_thread_num()].numberOfRows);
    }
}


#endif


//////////////////////////////////////////////////////////////////////////////
/// 8rVc => 8 rows, VEC columns
//////////////////////////////////////////////////////////////////////////////

template <class ValueType>
inline void CSR_to_SPC5_8rVc(SPC5Mat<ValueType>* csr){
    core_CSR_to_SPC5_rVc<ValueType, 8>(csr);
    csr->format = SPC5_MATRIX_TYPE::FORMAT_8rVc;
}

template <class ValueType>
inline SPC5Mat<ValueType> COO_to_SPC5_8rVc(const int nbRows, const int nbCols,
                                           Ijv<ValueType> values[], int nbValues){
    SPC5Mat<ValueType> mat = COO_unsorted_to_CSR(nbRows, nbCols, values, nbValues);
    CSR_to_SPC5_8rVc(&mat);
    return mat;
}

template <class ValueType>
inline void SPC5_8rVc_Spmv_scalar(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[]){
    assert(mat->format == SPC5_MATRIX_TYPE::FORMAT_8rVc);
    core_SPC5_rVc_Spmv_scalar<ValueType,8>(mat, x, y);
}

template <class ValueType, class FuncType>
inline void SPC5_8rVc_iterate(const SPC5Mat<ValueType>& mat, FuncType&& func){
    assert(mat->format == SPC5_MATRIX_TYPE::FORMAT_8rVc);
    core_SPC5_rVc_iterate<ValueType,8>(mat, std::forward<FuncType>(func));
}

template <class ValueType>
inline int SPC5_8rVc_block_count(const SPC5Mat<ValueType>& csr){
    return core_SPC5_block_count<ValueType,8,ValPerVec<ValueType>::size>(csr);
}

extern "C" void core_SPC5_8rVc_Spmv_double(const long int nbRows, const int* rowsSizes,
                                           const unsigned char* blocksColumnIndexesWithMasks,
                                           const double* values,
                                           const double* x, double* y);
extern "C" void core_SPC5_8rVc_Spmv_float(const long int nbRows, const int* rowsSizes,
                                          const unsigned char* blocksColumnIndexesWithMasks,
                                          const float* values,
                                          const float* x, float* y);

template <class ValueType>
inline void SPC5_8rVc_Spmv(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[]);

template <>
inline void SPC5_8rVc_Spmv<double>(const SPC5Mat<double>& mat, const double x[], double y[]){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_8rVc);
    core_SPC5_8rVc_Spmv_double(mat.numberOfRows, mat.rowsSize.get(),
                               mat.blocksColumnIndexesWithMasks.get(), mat.values.get(),
                               x, y);
}

template <>
inline void SPC5_8rVc_Spmv<float>(const SPC5Mat<float>& mat, const float x[], float y[]){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_8rVc);
    core_SPC5_8rVc_Spmv_float(mat.numberOfRows, mat.rowsSize.get(),
                              mat.blocksColumnIndexesWithMasks.get(), mat.values.get(),
                              x, y);
}



#ifdef _OPENMP

template <class ValueType>
inline std::vector<ThreadInterval<ValueType>> SPC5_8rVc_split_omp(const SPC5Mat<ValueType>& mat, const int numThreads){
    return core_SPC5_rVc_threadsplit<ValueType,8>(mat, numThreads);
}

template <class ValueType>
inline std::vector<ThreadInterval<ValueType>> SPC5_8rVc_split_omp(const SPC5Mat<ValueType>& mat){
    return core_SPC5_rVc_threadsplit<ValueType,8>(mat, omp_get_max_threads());
}


template <class ValueType>
inline void SPC5_8rVc_Spmv_omp(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[],
                               const std::vector<ThreadInterval<ValueType>>& threadsVecs);

template <>
inline void SPC5_8rVc_Spmv_omp<double>(const SPC5Mat<double>& mat, const double x[], double y[],
                                       const std::vector<ThreadInterval<double>>& threadsVecs){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_8rVc);
    const int numThreads = int(threadsVecs.size());
#pragma omp parallel num_threads(numThreads)
    {
        memset(threadsVecs[omp_get_thread_num()].threadY.get(), 0, sizeof(double)*threadsVecs[omp_get_thread_num()].numberOfRows);

        core_SPC5_8rVc_Spmv_double(threadsVecs[omp_get_thread_num()].numberOfRows,
                                   threadsVecs[omp_get_thread_num()].threadRowsSize.get(),
                                   threadsVecs[omp_get_thread_num()].threadBlocksColumnIndexesWithMasks.get(),
                                   threadsVecs[omp_get_thread_num()].threadValues.get(),
                                   x, threadsVecs[omp_get_thread_num()].threadY.get());

        SPC5_opti_merge(&y[threadsVecs[omp_get_thread_num()].startingRow], threadsVecs[omp_get_thread_num()].threadY.get(),
                        threadsVecs[omp_get_thread_num()].numberOfRows);
    }
}

template <>
inline void SPC5_8rVc_Spmv_omp<float>(const SPC5Mat<float>& mat, const float x[], float y[],
                                      const std::vector<ThreadInterval<float>>& threadsVecs){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_8rVc);
    const int numThreads = int(threadsVecs.size());
#pragma omp parallel num_threads(numThreads)
    {
        memset(threadsVecs[omp_get_thread_num()].threadY.get(), 0, sizeof(float)*threadsVecs[omp_get_thread_num()].numberOfRows);

        core_SPC5_8rVc_Spmv_float(threadsVecs[omp_get_thread_num()].numberOfRows,
                                  threadsVecs[omp_get_thread_num()].threadRowsSize.get(),
                                  threadsVecs[omp_get_thread_num()].threadBlocksColumnIndexesWithMasks.get(),
                                  threadsVecs[omp_get_thread_num()].threadValues.get(),
                                  x, threadsVecs[omp_get_thread_num()].threadY.get());

        SPC5_opti_merge(&y[threadsVecs[omp_get_thread_num()].startingRow], threadsVecs[omp_get_thread_num()].threadY.get(),
                        threadsVecs[omp_get_thread_num()].numberOfRows);
    }
}

#endif


//////////////////////////////////////////////////////////////////////////////
/// Wrappers
//////////////////////////////////////////////////////////////////////////////

template <class ValueType>
inline void CSR_to_SPC5(SPC5Mat<ValueType>* mat, const SPC5_MATRIX_TYPE matType){
    switch(matType){
    case SPC5_MATRIX_TYPE::FORMAT_1rVc_WT :
        {
        CSR_to_SPC5_1rVc<ValueType>(mat);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_2rVc :
        {
        CSR_to_SPC5_2rVc<ValueType>(mat);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_4rVc :
        {
        CSR_to_SPC5_4rVc<ValueType>(mat);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_8rVc :
    {
        CSR_to_SPC5_8rVc<ValueType>(mat);
    }
    break;
    default :
        {
            throw std::invalid_argument("CSR_to_SPC5 : Unknown format type");
        }
    }
    assert(mat.format == matType);
}

template <class ValueType>
inline void SPC5_Spmv_scalar(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[]){
    switch(mat.format){
    case SPC5_MATRIX_TYPE::FORMAT_1rVc_WT :
        {
        SPC5_1rVc_Spmv_scalar<ValueType>(mat, x, y);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_2rVc :
        {
        SPC5_2rVc_Spmv_scalar<ValueType>(mat, x, y);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_4rVc :
        {
        SPC5_4rVc_Spmv_scalar<ValueType>(mat, x, y);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_8rVc :
    {
        SPC5_8rVc_Spmv_scalar<ValueType>(mat, x, y);
    }
    break;
    case SPC5_MATRIX_TYPE::FORMAT_CSR :
        {
        CSR_Spmv_scalar<ValueType>(mat, x, y);
        }
        break;
    default :
        {
            throw std::invalid_argument("SPC5_Spmv_scalar : Unknown format type");
        }
    }
}

template <class ValueType, class FuncType>
inline void SPC5_iterate(const SPC5Mat<ValueType>& mat, FuncType&& func){
    switch(mat.format){
    case SPC5_MATRIX_TYPE::FORMAT_1rVc_WT :
        {
        SPC5_1rVc_iterate<ValueType>(mat, std::forward<FuncType>(func));
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_2rVc :
        {
        SPC5_2rVc_iterate<ValueType>(mat, std::forward<FuncType>(func));
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_4rVc :
        {
        SPC5_4rVc_iterate<ValueType>(mat, std::forward<FuncType>(func));
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_8rVc :
    {
        SPC5_8rVc_iterate<ValueType>(mat, std::forward<FuncType>(func));
    }
    break;
    case SPC5_MATRIX_TYPE::FORMAT_CSR :
        {
        CSR_iterate<ValueType>(mat, std::forward<FuncType>(func));
        }
        break;
    default :
        {
            throw std::invalid_argument("SPC5_iterate : Unknown format type");
        }
    }
}

template <class ValueType>
inline void SPC5_Spmv(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[]){
    switch(mat.format){
    case SPC5_MATRIX_TYPE::FORMAT_1rVc_WT :
        {
        SPC5_1rVc_Spmv<ValueType>(mat, x, y);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_2rVc :
        {
        SPC5_2rVc_Spmv<ValueType>(mat, x, y);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_4rVc :
        {
        SPC5_4rVc_Spmv<ValueType>(mat, x, y);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_8rVc :
    {
        SPC5_8rVc_Spmv<ValueType>(mat, x, y);
    }
    break;
    default :
        {
            throw std::invalid_argument("SPC5_Spmv : Unknown format type");
        }
    }
}

template <class ValueType>
inline void SPC5_block_count(const SPC5Mat<ValueType>& csr, const SPC5_MATRIX_TYPE matType){
    switch(matType){
    case SPC5_MATRIX_TYPE::FORMAT_1rVc_WT :
        {
        SPC5_1rVc_block_count<ValueType>(csr);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_2rVc :
        {
        SPC5_2rVc_block_count<ValueType>(csr);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_4rVc :
        {
        SPC5_4rVc_block_count<ValueType>(csr);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_8rVc :
    {
        SPC5_8rVc_block_count<ValueType>(csr);
    }
    break;
    default :
        {
            throw std::invalid_argument("SPC5_Spmv : Unknown format type");
        }
    }
}


template <class ValueType>
std::pair<SPC5_MATRIX_TYPE, double> SPC5_find_best(const SPC5Mat<ValueType>& csr){
    const bool in_double = (sizeof(ValueType) == sizeof(double));
    SPC5_MATRIX_TYPE bestType = SPC5_MATRIX_TYPE::UNDEFINED_FORMAT;
    double estimatedSpeed = 0;

    auto polyval = [](const double coef[4], const double x) -> double{
        return coef[0]*(x*x*x) + coef[1]*(x*x) + coef[2]*x + coef[3];
    };

        {
            if(in_double){
                const double coef[4] = {2.186490e-03, -7.931533e-02 , 8.719444e-01 , 3.712380e-01  };
                const int nbBlocks = SPC5_1rVc_block_count<ValueType>(csr);
                const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
                if(thisSpeed > estimatedSpeed){
                    estimatedSpeed = thisSpeed;
                    bestType = SPC5_MATRIX_TYPE::FORMAT_1rVc_WT;
                }
            }
            else{
                const double coef[4] = {-4.039852e-03, 7.997606e-02 , 6.892119e-02 , 1.508645e+00  };
                const int nbBlocks = SPC5_1rVc_block_count<ValueType>(csr);
                const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
                if(thisSpeed > estimatedSpeed){
                    estimatedSpeed = thisSpeed;
                    bestType = SPC5_MATRIX_TYPE::FORMAT_1rVc_WT;
                }
            }
        }
    {
        if(in_double){
            const double coef[4] = { 9.822706e-04, -4.500002e-02 , 6.714119e-01 , 2.413985e-01  };
            const int nbBlocks = SPC5_2rVc_block_count<ValueType>(csr);
            const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
            if(thisSpeed > estimatedSpeed){
                estimatedSpeed = thisSpeed;
                bestType = SPC5_MATRIX_TYPE::FORMAT_2rVc;
            }
        }
        else{
            const double coef[4] = { -9.195940e-04, 4.442050e-03 , 5.273045e-01 , 1.101610e+00  };
            const int nbBlocks = SPC5_2rVc_block_count<ValueType>(csr);
            const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
            if(thisSpeed > estimatedSpeed){
                estimatedSpeed = thisSpeed;
                bestType = SPC5_MATRIX_TYPE::FORMAT_2rVc;
            }
        }
    }
    {
        if(in_double){
            const double coef[4] = { 1.320496e-04, -1.222130e-02 , 3.520409e-01 , 3.603195e-01 };
            const int nbBlocks = SPC5_4rVc_block_count<ValueType>(csr);
            const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
            if(thisSpeed > estimatedSpeed){
                estimatedSpeed = thisSpeed;
                bestType = SPC5_MATRIX_TYPE::FORMAT_4rVc;
            }
        }
        else{
            const double coef[4] = { -4.464933e-05, 1.867266e-03 , 1.520910e-01 , 1.186131e+00  };
            const int nbBlocks = SPC5_4rVc_block_count<ValueType>(csr);
            const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
            if(thisSpeed > estimatedSpeed){
                estimatedSpeed = thisSpeed;
                bestType = SPC5_MATRIX_TYPE::FORMAT_4rVc;
            }
        }
    }

    return std::pair<SPC5_MATRIX_TYPE, double>(bestType, estimatedSpeed);
}





inline const char* SPC5_type_to_string(const SPC5_MATRIX_TYPE matType){
    switch(matType){
    case SPC5_MATRIX_TYPE::FORMAT_1rVc_WT :
    {
        return "1rVc_WT";
    }
    case SPC5_MATRIX_TYPE::FORMAT_2rVc :
    {
        return "2rVc";
    }
    case SPC5_MATRIX_TYPE::FORMAT_4rVc :
    {
        return "4rVc";
    }
    case SPC5_MATRIX_TYPE::FORMAT_8rVc :
    {
        return "8rVc";
    }
    default :
        {
            return "undefined";
        }
    }
}


#ifdef _OPENMP



template <class ValueType>
std::pair<SPC5_MATRIX_TYPE, double> SPC5_find_best_omp(const SPC5Mat<ValueType>& csr,
                                                       const int nbThreads);


template <>
inline std::pair<SPC5_MATRIX_TYPE, double> SPC5_find_best_omp<double>(const SPC5Mat<double>& csr,
                                                               const int nbThreads){
    using ValueType = double;
    SPC5_MATRIX_TYPE bestType = SPC5_MATRIX_TYPE::UNDEFINED_FORMAT;
    double estimatedSpeed = 0;

    auto polyval = [nbThreads](const double coef[4], const double x) -> double{
        return coef[0] + coef[1]*nbThreads + coef[2]*x + coef[3]*nbThreads*x;
    };

    {
        const double coef[4] = {4.3764,0.0981,1.8728,0.0721};
        const int nbBlocks = SPC5_1rVc_block_count<ValueType>(csr);
        const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
        if(thisSpeed > estimatedSpeed){
            estimatedSpeed = thisSpeed;
            bestType = SPC5_MATRIX_TYPE::FORMAT_1rVc_WT;
        }
    }
    {
        const double coef[4] = { 5.2172,0.1435,0.8331,0.0556};
        const int nbBlocks = SPC5_2rVc_block_count<ValueType>(csr);
        const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
        if(thisSpeed > estimatedSpeed){
            estimatedSpeed = thisSpeed;
            bestType = SPC5_MATRIX_TYPE::FORMAT_2rVc;
        }
    }
    {
        const double coef[4] = { 5.3212,0.1921,0.5201,0.0263};
        const int nbBlocks = SPC5_4rVc_block_count<ValueType>(csr);
        const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
        if(thisSpeed > estimatedSpeed){
            estimatedSpeed = thisSpeed;
            bestType = SPC5_MATRIX_TYPE::FORMAT_4rVc;
        }
    }

    return std::pair<SPC5_MATRIX_TYPE, double>(bestType, estimatedSpeed);
}


template <class ValueType>
inline std::vector<ThreadInterval<ValueType>> SPC5_split_omp(const SPC5Mat<ValueType>& mat){
    switch(mat.format){
    case SPC5_MATRIX_TYPE::FORMAT_1rVc_WT :
        {
        return SPC5_1rVc_split_omp<ValueType>(mat);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_2rVc :
        {
        return SPC5_2rVc_split_omp<ValueType>(mat);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_4rVc :
        {
        return SPC5_4rVc_split_omp<ValueType>(mat);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_8rVc :
    {
        return SPC5_8rVc_split_omp<ValueType>(mat);
    }
    break;
    default :
        {
            throw std::invalid_argument("SPC5_Spmv : Unknown format type");
        }
    }
    return std::vector<ThreadInterval<ValueType>>();
}

template <class ValueType>
inline void SPC5_Spmv_omp(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[],
                          const std::vector<ThreadInterval<ValueType>>& threadsVecs){
    switch(mat.format){
    case SPC5_MATRIX_TYPE::FORMAT_1rVc_WT :
        {
        SPC5_1rVc_Spmv_omp<ValueType>(mat, x, y, threadsVecs);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_2rVc :
        {
        SPC5_2rVc_Spmv_omp<ValueType>(mat, x, y, threadsVecs);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_4rVc :
        {
        SPC5_4rVc_Spmv_omp<ValueType>(mat, x, y, threadsVecs);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_8rVc :
    {
        SPC5_8rVc_Spmv_omp<ValueType>(mat, x, y, threadsVecs);
    }
    break;
    default :
        {
            throw std::invalid_argument("SPC5_Spmv : Unknown format type");
        }
    }
}
#endif

////////////////////////////////////////////////////////////////////////
#ifndef USE_AVX512

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#else
#include "farm_sve.h"
#endif /* __ARM_FEATURE_SVE */


extern "C" void SPC5_opti_merge_double(double dest[], const double src[], const int nbValues){
    for(int idx = 0 ; idx < nbValues ; idx += svcntd()){
            const svbool_t predicate = svwhilelt_b64_s32(idx, nbValues);
            const svfloat64_t values = svld1(predicate, &src[idx]);
            const svfloat64_t valuesdest = svld1(predicate, &dest[idx]);
            const svfloat64_t res = svadd_z(predicate, values, valuesdest);
            svst1_f64(predicate, &dest[idx], res);
    }
}

extern "C" void SPC5_opti_merge_float(float dest[], const float src[], const int nbValues){
    for(int idx = 0 ; idx < nbValues ; idx += svcntw()){
            const svbool_t predicate = svwhilelt_b32_s32(idx, nbValues);
            const svfloat32_t values = svld1(predicate, &src[idx]);
            const svfloat32_t valuesdest = svld1(predicate, &dest[idx]);
            const svfloat32_t res = svadd_z(predicate, values, valuesdest);
            svst1_f32(predicate, &dest[idx], res);
    }
}



//////////////////////////////////////////////////////////////////////////

void core_SPC5_1rVc_Spmv_double(const long int nbRows, const int* rowSizes,
                                const unsigned char* headers,
                                const double* values,
                                const double* x, double* y){
    //const svbool_t false_vec = svpfalse();
    const svbool_t true_vec = svptrue_b64();
    const svfloat64_t zeros = svdup_n_f64(0);
    const unsigned long maskFilterValues[8] = {1<<0, 1<<1, 1<<2, 1<<3,
                                               1<<4, 1<<5, 1<<6, 1<<7};
    const svuint64_t maskFilter = svld1_u64(true_vec, maskFilterValues);

    for (int idxRow = 0; idxRow < nbRows; ++idxRow) {

            svfloat64_t sum_vec = zeros;

            for (int idxBlock = rowSizes[idxRow]; idxBlock < rowSizes[idxRow+1]; ++idxBlock) {
            const int idxCol = *((const int *)headers);
            const unsigned char mask = headers[4];

            const svuint64_t maskInVec = svdup_n_u64(mask);

            const svbool_t mask_vec = svcmpne_n_u64(true_vec, svand_u64_z(true_vec, maskFilter, maskInVec), 0);

            const uint64_t increment = svcntp_b64(mask_vec, mask_vec);

            const svfloat64_t xvals = svcompact(mask_vec, svld1(mask_vec, &x[idxCol]));

            const svfloat64_t block = svld1(svwhilelt_b64_s32(0, increment), values);
            values += increment;

            sum_vec = svmla_m(true_vec, sum_vec, block, xvals);

            headers += 5;
            }

            y[idxRow] += svaddv(true_vec, sum_vec);
    }
}

void core_SPC5_1rVc_Spmv_float(const long int nbRows, const int* rowSizes,
                               const unsigned char* headers,
                               const float* values,
                               const float* x, float* y){
    //const svbool_t false_vec = svpfalse();
    const svbool_t true_vec = svptrue_b32();
    const svfloat32_t zeros = svdup_n_f32(0);
    const unsigned int maskFilterValues[16] = {1<<0, 1<<1, 1<<2, 1<<3,
                                               1<<4, 1<<5, 1<<6, 1<<7,
                                               1<<8, 1<<9, 1<<10, 1<<11,
                                               1<<12, 1<<13, 1<<14, 1<<15};
    const svuint32_t maskFilter = svld1_u32(true_vec, maskFilterValues);

    for (int idxRow = 0; idxRow < nbRows; ++idxRow) {

            svfloat32_t sum_vec = zeros;

            for (int idxBlock = rowSizes[idxRow]; idxBlock < rowSizes[idxRow+1]; ++idxBlock) {
            const int idxCol = *((const int *)headers);
            const unsigned short mask = *(const unsigned short*)&headers[4];

            const svuint32_t maskInVec = svdup_n_u32(mask);

            const svbool_t mask_vec = svcmpne_n_u32(true_vec, svand_u32_z(true_vec, maskFilter, maskInVec), 0);

            const uint32_t increment = svcntp_b32(mask_vec, mask_vec);

            const svfloat32_t xvals = svcompact(mask_vec, svld1(mask_vec, &x[idxCol]));

            const svfloat32_t block = svld1(svwhilelt_b32_s32(0, increment), values);
            values += increment;

            sum_vec = svmla_m(true_vec, sum_vec, block, xvals);

            headers += 6;
            }

            y[idxRow] += svaddv(true_vec, sum_vec);
    }
}

//////////////////////////////////////////////////////////////////////////
void core_SPC5_2rVc_Spmv_double(const long int nbRows, const int* rowSizes,
                                const unsigned char* headers,
                                const double* values,
                                const double* x, double* y){
    //const svbool_t false_vec = svpfalse();
    const svbool_t true_vec = svptrue_b64();
    const svfloat64_t zeros = svdup_n_f64(0);
    const unsigned long maskFilterValues[8] = {1<<0, 1<<1, 1<<2, 1<<3,
                                               1<<4, 1<<5, 1<<6, 1<<7};
    const svuint64_t maskFilter = svld1_u64(true_vec, maskFilterValues);

    for (int idxRow = 0; idxRow < nbRows; idxRow += 2) {
            const int idxRowBlock = idxRow/8;

            svfloat64_t sum_vec = zeros;
            svfloat64_t sum_vec_1 = zeros;

            for (int idxBlock = rowSizes[idxRowBlock]; idxBlock < rowSizes[idxRowBlock+1]; ++idxBlock) {
            const int idxCol = *((const int *)headers);
            const unsigned char mask = headers[4];
            const unsigned char mask_1 = headers[5];

            const svuint64_t maskInVec = svdup_n_u64(mask);
            const svbool_t mask_vec = svcmpne_n_u64(true_vec, svand_u64_z(true_vec, maskFilter, maskInVec), 0);

            const svuint64_t maskInVec_1 = svdup_n_u64(mask_1);
            const svbool_t mask_vec_1 = svcmpne_n_u64(true_vec, svand_u64_z(true_vec, maskFilter, maskInVec_1), 0);

            const uint64_t increment = svcntp_b64(mask_vec, mask_vec);
            const uint64_t increment_1 = svcntp_b64(mask_vec_1, mask_vec_1);

            const svfloat64_t xvec = svld1(true_vec, &x[idxCol]);
            const svfloat64_t xvals = svcompact(mask_vec, xvec);
            const svfloat64_t xvals_1 = svcompact(mask_vec_1, xvec);

            const svfloat64_t block = svld1(svwhilelt_b64_s32(0, increment), values);
            values += increment;

            const svfloat64_t block_1 = svld1(svwhilelt_b64_s32(0, increment_1), values);
            values += increment_1;

            sum_vec = svmla_m(true_vec, sum_vec, block, xvals);
            sum_vec_1 = svmla_m(true_vec, sum_vec_1, block_1, xvals_1);

            headers += 6;
            }

            y[idxRow] += svaddv(true_vec, sum_vec);
            y[idxRow+1] += svaddv(true_vec, sum_vec_1);
    }
}



void core_SPC5_2rVc_Spmv_float(const long int nbRows, const int* rowSizes,
                               const unsigned char* headers,
                               const float* values,
                               const float* x, float* y){
    //const svbool_t false_vec = svpfalse();
    const svbool_t true_vec = svptrue_b32();
    const svfloat32_t zeros = svdup_n_f32(0);
    const unsigned int maskFilterValues[16] = {1<<0, 1<<1, 1<<2, 1<<3,
                                               1<<4, 1<<5, 1<<6, 1<<7,
                                               1<<8, 1<<9, 1<<10, 1<<11,
                                               1<<12, 1<<13, 1<<14, 1<<15};
    const svuint32_t maskFilter = svld1_u32(true_vec, maskFilterValues);

    for (int idxRow = 0; idxRow < nbRows; idxRow += 2) {
            const int idxRowBlock = idxRow/2;

            svfloat32_t sum_vec = zeros;
            svfloat32_t sum_vec_1 = zeros;

            for (int idxBlock = rowSizes[idxRowBlock]; idxBlock < rowSizes[idxRowBlock+1]; ++idxBlock) {
            const int idxCol = *((const int *)headers);
            const unsigned short mask = *(const unsigned short*)&headers[4];
            const unsigned short mask_1 = *(const unsigned short*)&headers[6];

            const svuint32_t maskInVec = svdup_n_u32(mask);
            const svbool_t mask_vec = svcmpne_n_u32(true_vec, svand_u32_z(true_vec, maskFilter, maskInVec), 0);

            const svuint32_t maskInVec_1 = svdup_n_u32(mask_1);
            const svbool_t mask_vec_1 = svcmpne_n_u32(true_vec, svand_u32_z(true_vec, maskFilter, maskInVec_1), 0);

            const uint32_t increment = svcntp_b32(mask_vec, mask_vec);
            const uint32_t increment_1 = svcntp_b32(mask_vec_1, mask_vec_1);

            const svfloat32_t xvec = svld1(true_vec, &x[idxCol]);
            const svfloat32_t xvals = svcompact(mask_vec, xvec);
            const svfloat32_t xvals_1 = svcompact(mask_vec_1, xvec);

            const svfloat32_t block = svld1(svwhilelt_b32_s32(0, increment), values);
            values += increment;

            const svfloat32_t block_1 = svld1(svwhilelt_b32_s32(0, increment_1), values);
            values += increment_1;

            sum_vec = svmla_m(true_vec, sum_vec, block, xvals);
            sum_vec_1 = svmla_m(true_vec, sum_vec_1, block_1, xvals_1);

            headers += 8;
            }

            y[idxRow] += svaddv(true_vec, sum_vec);
            y[idxRow+1] += svaddv(true_vec, sum_vec_1);
    }
}

//////////////////////////////////////////////////////////////////////////

void core_SPC5_4rVc_Spmv_double(const long int nbRows, const int* rowSizes,
                                const unsigned char* headers,
                                const double* values,
                                const double* x, double* y){
    //const svbool_t false_vec = svpfalse();
    const svbool_t true_vec = svptrue_b64();
    const svfloat64_t zeros = svdup_n_f64(0);
    const unsigned long maskFilterValues[8] = {1<<0, 1<<1, 1<<2, 1<<3,
                                               1<<4, 1<<5, 1<<6, 1<<7};
    const svuint64_t maskFilter = svld1_u64(true_vec, maskFilterValues);

    for (int idxRow = 0; idxRow < nbRows; idxRow += 4) {
            const int idxRowBlock = idxRow/4;

            svfloat64_t sum_vec = zeros;
            svfloat64_t sum_vec_1 = zeros;
            svfloat64_t sum_vec_2 = zeros;
            svfloat64_t sum_vec_3 = zeros;

            for (int idxBlock = rowSizes[idxRowBlock]; idxBlock < rowSizes[idxRowBlock+1]; ++idxBlock) {
            const int idxCol = *((const int *)headers);
            const unsigned char mask = headers[4];
            const unsigned char mask_1 = headers[5];
            const unsigned char mask_2 = headers[6];
            const unsigned char mask_3 = headers[7];

            const svuint64_t maskInVec = svdup_n_u64(mask);
            const svbool_t mask_vec = svcmpne_n_u64(true_vec, svand_u64_z(true_vec, maskFilter, maskInVec), 0);

            const svuint64_t maskInVec_1 = svdup_n_u64(mask_1);
            const svbool_t mask_vec_1 = svcmpne_n_u64(true_vec, svand_u64_z(true_vec, maskFilter, maskInVec_1), 0);

            const svuint64_t maskInVec_2 = svdup_n_u64(mask_2);
            const svbool_t mask_vec_2 = svcmpne_n_u64(true_vec, svand_u64_z(true_vec, maskFilter, maskInVec_2), 0);

            const svuint64_t maskInVec_3 = svdup_n_u64(mask_3);
            const svbool_t mask_vec_3 = svcmpne_n_u64(true_vec, svand_u64_z(true_vec, maskFilter, maskInVec_3), 0);

            const uint64_t increment = svcntp_b64(mask_vec, mask_vec);
            const uint64_t increment_1 = svcntp_b64(mask_vec_1, mask_vec_1);
            const uint64_t increment_2 = svcntp_b64(mask_vec_2, mask_vec_2);
            const uint64_t increment_3 = svcntp_b64(mask_vec_3, mask_vec_3);

            const svfloat64_t xvec = svld1(true_vec, &x[idxCol]);
            const svfloat64_t xvals = svcompact(mask_vec, xvec);
            const svfloat64_t xvals_1 = svcompact(mask_vec_1, xvec);
            const svfloat64_t xvals_2 = svcompact(mask_vec_2, xvec);
            const svfloat64_t xvals_3 = svcompact(mask_vec_3, xvec);

            const svfloat64_t block = svld1(svwhilelt_b64_s32(0, increment), values);
            values += increment;

            const svfloat64_t block_1 = svld1(svwhilelt_b64_s32(0, increment_1), values);
            values += increment_1;

            const svfloat64_t block_2 = svld1(svwhilelt_b64_s32(0, increment_2), values);
            values += increment_2;

            const svfloat64_t block_3 = svld1(svwhilelt_b64_s32(0, increment_3), values);
            values += increment_3;

            sum_vec = svmla_m(true_vec, sum_vec, block, xvals);
            sum_vec_1 = svmla_m(true_vec, sum_vec_1, block_1, xvals_1);
            sum_vec_2 = svmla_m(true_vec, sum_vec_2, block_2, xvals_2);
            sum_vec_3 = svmla_m(true_vec, sum_vec_3, block_3, xvals_3);

            headers += 8;
            }

            y[idxRow] += svaddv(true_vec, sum_vec);
            y[idxRow+1] += svaddv(true_vec, sum_vec_1);
            y[idxRow+2] += svaddv(true_vec, sum_vec_2);
            y[idxRow+3] += svaddv(true_vec, sum_vec_3);
    }
}




void core_SPC5_4rVc_Spmv_float(const long int nbRows, const int* rowSizes,
                               const unsigned char* headers,
                               const float* values,
                               const float* x, float* y){
    //const svbool_t false_vec = svpfalse();
    const svbool_t true_vec = svptrue_b32();
    const svfloat32_t zeros = svdup_n_f32(0);
    const unsigned int maskFilterValues[16] = {1<<0, 1<<1, 1<<2, 1<<3,
                                               1<<4, 1<<5, 1<<6, 1<<7,
                                               1<<8, 1<<9, 1<<10, 1<<11,
                                               1<<12, 1<<13, 1<<14, 1<<15};
    const svuint32_t maskFilter = svld1_u32(true_vec, maskFilterValues);

    for (int idxRow = 0; idxRow < nbRows; idxRow += 4) {
            const int idxRowBlock = idxRow/4;

            svfloat32_t sum_vec = zeros;
            svfloat32_t sum_vec_1 = zeros;
            svfloat32_t sum_vec_2 = zeros;
            svfloat32_t sum_vec_3 = zeros;

            for (int idxBlock = rowSizes[idxRowBlock]; idxBlock < rowSizes[idxRowBlock+1]; ++idxBlock) {
            const int idxCol = *((const int *)headers);
            const unsigned short mask = *(const unsigned short*)&headers[4];
            const unsigned short mask_1 = *(const unsigned short*)&headers[6];
            const unsigned short mask_2 = *(const unsigned short*)&headers[8];
            const unsigned short mask_3 = *(const unsigned short*)&headers[10];

            const svuint32_t maskInVec = svdup_n_u32(mask);
            const svbool_t mask_vec = svcmpne_n_u32(true_vec, svand_u32_z(true_vec, maskFilter, maskInVec), 0);

            const svuint32_t maskInVec_1 = svdup_n_u32(mask_1);
            const svbool_t mask_vec_1 = svcmpne_n_u32(true_vec, svand_u32_z(true_vec, maskFilter, maskInVec_1), 0);

            const svuint32_t maskInVec_2 = svdup_n_u32(mask_2);
            const svbool_t mask_vec_2 = svcmpne_n_u32(true_vec, svand_u32_z(true_vec, maskFilter, maskInVec_2), 0);

            const svuint32_t maskInVec_3 = svdup_n_u32(mask_3);
            const svbool_t mask_vec_3 = svcmpne_n_u32(true_vec, svand_u32_z(true_vec, maskFilter, maskInVec_3), 0);

            const uint32_t increment = svcntp_b32(mask_vec, mask_vec);
            const uint32_t increment_1 = svcntp_b32(mask_vec_1, mask_vec_1);
            const uint32_t increment_2 = svcntp_b32(mask_vec_2, mask_vec_2);
            const uint32_t increment_3 = svcntp_b32(mask_vec_3, mask_vec_3);

            const svfloat32_t xvec = svld1(true_vec, &x[idxCol]);
            const svfloat32_t xvals = svcompact(mask_vec, xvec);
            const svfloat32_t xvals_1 = svcompact(mask_vec_1, xvec);
            const svfloat32_t xvals_2 = svcompact(mask_vec_2, xvec);
            const svfloat32_t xvals_3 = svcompact(mask_vec_3, xvec);

            const svfloat32_t block = svld1(svwhilelt_b32_s32(0, increment), values);
            values += increment;

            const svfloat32_t block_1 = svld1(svwhilelt_b32_s32(0, increment_1), values);
            values += increment_1;

            const svfloat32_t block_2 = svld1(svwhilelt_b32_s32(0, increment_2), values);
            values += increment_2;

            const svfloat32_t block_3 = svld1(svwhilelt_b32_s32(0, increment_3), values);
            values += increment_3;

            sum_vec = svmla_m(true_vec, sum_vec, block, xvals);
            sum_vec_1 = svmla_m(true_vec, sum_vec_1, block_1, xvals_1);
            sum_vec_2 = svmla_m(true_vec, sum_vec_2, block_2, xvals_2);
            sum_vec_3 = svmla_m(true_vec, sum_vec_3, block_3, xvals_3);

            headers += 12;
            }

            y[idxRow] += svaddv(true_vec, sum_vec);
            y[idxRow+1] += svaddv(true_vec, sum_vec_1);
            y[idxRow+2] += svaddv(true_vec, sum_vec_2);
            y[idxRow+3] += svaddv(true_vec, sum_vec_3);
    }
}


//////////////////////////////////////////////////////////////////////////

void core_SPC5_8rVc_Spmv_double(const long int nbRows, const int* rowSizes,
                                const unsigned char* headers,
                                const double* values,
                                const double* x, double* y){
    //const svbool_t false_vec = svpfalse();
    const svbool_t true_vec = svptrue_b64();
    const svfloat64_t zeros = svdup_n_f64(0);
    const unsigned long maskFilterValues[8] = {1<<0, 1<<1, 1<<2, 1<<3,
                                               1<<4, 1<<5, 1<<6, 1<<7};
    const svuint64_t maskFilter = svld1_u64(true_vec, maskFilterValues);

    for (int idxRow = 0; idxRow < nbRows; idxRow += 8) {
            const int idxRowBlock = idxRow/8;

            svfloat64_t sum_vec = zeros;
            svfloat64_t sum_vec_1 = zeros;
            svfloat64_t sum_vec_2 = zeros;
            svfloat64_t sum_vec_3 = zeros;
            svfloat64_t sum_vec_4 = zeros;
            svfloat64_t sum_vec_5 = zeros;
            svfloat64_t sum_vec_6 = zeros;
            svfloat64_t sum_vec_7 = zeros;

            for (int idxBlock = rowSizes[idxRowBlock]; idxBlock < rowSizes[idxRowBlock+1]; ++idxBlock) {
            const int idxCol = *((const int *)headers);
            const unsigned char mask = headers[4];
            const unsigned char mask_1 = headers[5];
            const unsigned char mask_2 = headers[6];
            const unsigned char mask_3 = headers[7];
            const unsigned char mask_4 = headers[8];
            const unsigned char mask_5 = headers[9];
            const unsigned char mask_6 = headers[10];
            const unsigned char mask_7 = headers[11];

            const svuint64_t maskInVec = svdup_n_u64(mask);
            const svbool_t mask_vec = svcmpne_n_u64(true_vec, svand_u64_z(true_vec, maskFilter, maskInVec), 0);

            const svuint64_t maskInVec_1 = svdup_n_u64(mask_1);
            const svbool_t mask_vec_1 = svcmpne_n_u64(true_vec, svand_u64_z(true_vec, maskFilter, maskInVec_1), 0);

            const svuint64_t maskInVec_2 = svdup_n_u64(mask_2);
            const svbool_t mask_vec_2 = svcmpne_n_u64(true_vec, svand_u64_z(true_vec, maskFilter, maskInVec_2), 0);

            const svuint64_t maskInVec_3 = svdup_n_u64(mask_3);
            const svbool_t mask_vec_3 = svcmpne_n_u64(true_vec, svand_u64_z(true_vec, maskFilter, maskInVec_3), 0);

            const svuint64_t maskInVec_4 = svdup_n_u64(mask_4);
            const svbool_t mask_vec_4 = svcmpne_n_u64(true_vec, svand_u64_z(true_vec, maskFilter, maskInVec_4), 0);

            const svuint64_t maskInVec_5 = svdup_n_u64(mask_5);
            const svbool_t mask_vec_5 = svcmpne_n_u64(true_vec, svand_u64_z(true_vec, maskFilter, maskInVec_5), 0);

            const svuint64_t maskInVec_6 = svdup_n_u64(mask_6);
            const svbool_t mask_vec_6 = svcmpne_n_u64(true_vec, svand_u64_z(true_vec, maskFilter, maskInVec_6), 0);

            const svuint64_t maskInVec_7 = svdup_n_u64(mask_7);
            const svbool_t mask_vec_7 = svcmpne_n_u64(true_vec, svand_u64_z(true_vec, maskFilter, maskInVec_7), 0);

            const uint64_t increment = svcntp_b64(mask_vec, mask_vec);
            const uint64_t increment_1 = svcntp_b64(mask_vec_1, mask_vec_1);
            const uint64_t increment_2 = svcntp_b64(mask_vec_2, mask_vec_2);
            const uint64_t increment_3 = svcntp_b64(mask_vec_3, mask_vec_3);
            const uint64_t increment_4 = svcntp_b64(mask_vec_4, mask_vec_4);
            const uint64_t increment_5 = svcntp_b64(mask_vec_5, mask_vec_5);
            const uint64_t increment_6 = svcntp_b64(mask_vec_6, mask_vec_6);
            const uint64_t increment_7 = svcntp_b64(mask_vec_7, mask_vec_7);

            const svfloat64_t xvec = svld1(true_vec, &x[idxCol]);
            const svfloat64_t xvals = svcompact(mask_vec, xvec);
            const svfloat64_t xvals_1 = svcompact(mask_vec_1, xvec);
            const svfloat64_t xvals_2 = svcompact(mask_vec_2, xvec);
            const svfloat64_t xvals_3 = svcompact(mask_vec_3, xvec);
            const svfloat64_t xvals_4 = svcompact(mask_vec_4, xvec);
            const svfloat64_t xvals_5 = svcompact(mask_vec_5, xvec);
            const svfloat64_t xvals_6 = svcompact(mask_vec_6, xvec);
            const svfloat64_t xvals_7 = svcompact(mask_vec_7, xvec);

            const svfloat64_t block = svld1(svwhilelt_b64_s32(0, increment), values);
            values += increment;

            const svfloat64_t block_1 = svld1(svwhilelt_b64_s32(0, increment_1), values);
            values += increment_1;

            const svfloat64_t block_2 = svld1(svwhilelt_b64_s32(0, increment_2), values);
            values += increment_2;

            const svfloat64_t block_3 = svld1(svwhilelt_b64_s32(0, increment_3), values);
            values += increment_3;

            const svfloat64_t block_4 = svld1(svwhilelt_b64_s32(0, increment_4), values);
            values += increment_4;

            const svfloat64_t block_5 = svld1(svwhilelt_b64_s32(0, increment_5), values);
            values += increment_5;

            const svfloat64_t block_6 = svld1(svwhilelt_b64_s32(0, increment_6), values);
            values += increment_6;

            const svfloat64_t block_7 = svld1(svwhilelt_b64_s32(0, increment_7), values);
            values += increment_7;

            sum_vec = svmla_m(true_vec, sum_vec, block, xvals);
            sum_vec_1 = svmla_m(true_vec, sum_vec_1, block_1, xvals_1);
            sum_vec_2 = svmla_m(true_vec, sum_vec_2, block_2, xvals_2);
            sum_vec_3 = svmla_m(true_vec, sum_vec_3, block_3, xvals_3);
            sum_vec_4 = svmla_m(true_vec, sum_vec_4, block_4, xvals_4);
            sum_vec_5 = svmla_m(true_vec, sum_vec_5, block_5, xvals_5);
            sum_vec_6 = svmla_m(true_vec, sum_vec_6, block_6, xvals_6);
            sum_vec_7 = svmla_m(true_vec, sum_vec_7, block_7, xvals_7);

            headers += 12;
            }

            y[idxRow] += svaddv(true_vec, sum_vec);
            y[idxRow+1] += svaddv(true_vec, sum_vec_1);
            y[idxRow+2] += svaddv(true_vec, sum_vec_2);
            y[idxRow+3] += svaddv(true_vec, sum_vec_3);
            y[idxRow+4] += svaddv(true_vec, sum_vec_4);
            y[idxRow+5] += svaddv(true_vec, sum_vec_5);
            y[idxRow+6] += svaddv(true_vec, sum_vec_6);
            y[idxRow+7] += svaddv(true_vec, sum_vec_7);
    }
}




void core_SPC5_8rVc_Spmv_float(const long int nbRows, const int* rowSizes,
                               const unsigned char* headers,
                               const float* values,
                               const float* x, float* y){
    //const svbool_t false_vec = svpfalse();
    const svbool_t true_vec = svptrue_b32();
    const svfloat32_t zeros = svdup_n_f32(0);
    const unsigned int maskFilterValues[16] = {1<<0, 1<<1, 1<<2, 1<<3,
                                               1<<4, 1<<5, 1<<6, 1<<7,
                                               1<<8, 1<<9, 1<<10, 1<<11,
                                               1<<12, 1<<13, 1<<14, 1<<15};
    const svuint32_t maskFilter = svld1_u32(true_vec, maskFilterValues);

    for (int idxRow = 0; idxRow < nbRows; idxRow += 8) {
            const int idxRowBlock = idxRow/8;

            svfloat32_t sum_vec = zeros;
            svfloat32_t sum_vec_1 = zeros;
            svfloat32_t sum_vec_2 = zeros;
            svfloat32_t sum_vec_3 = zeros;
            svfloat32_t sum_vec_4 = zeros;
            svfloat32_t sum_vec_5 = zeros;
            svfloat32_t sum_vec_6 = zeros;
            svfloat32_t sum_vec_7 = zeros;

            for (int idxBlock = rowSizes[idxRowBlock]; idxBlock < rowSizes[idxRowBlock+1]; ++idxBlock) {
            const int idxCol = *((const int *)headers);
            const unsigned short mask = *(const unsigned short*)&headers[4];
            const unsigned short mask_1 = *(const unsigned short*)&headers[6];
            const unsigned short mask_2 = *(const unsigned short*)&headers[8];
            const unsigned short mask_3 = *(const unsigned short*)&headers[10];
            const unsigned short mask_4 = *(const unsigned short*)&headers[12];
            const unsigned short mask_5 = *(const unsigned short*)&headers[14];
            const unsigned short mask_6 = *(const unsigned short*)&headers[16];
            const unsigned short mask_7 = *(const unsigned short*)&headers[18];

            const svuint32_t maskInVec = svdup_n_u32(mask);
            const svbool_t mask_vec = svcmpne_n_u32(true_vec, svand_u32_z(true_vec, maskFilter, maskInVec), 0);

            const svuint32_t maskInVec_1 = svdup_n_u32(mask_1);
            const svbool_t mask_vec_1 = svcmpne_n_u32(true_vec, svand_u32_z(true_vec, maskFilter, maskInVec_1), 0);

            const svuint32_t maskInVec_2 = svdup_n_u32(mask_2);
            const svbool_t mask_vec_2 = svcmpne_n_u32(true_vec, svand_u32_z(true_vec, maskFilter, maskInVec_2), 0);

            const svuint32_t maskInVec_3 = svdup_n_u32(mask_3);
            const svbool_t mask_vec_3 = svcmpne_n_u32(true_vec, svand_u32_z(true_vec, maskFilter, maskInVec_3), 0);

            const svuint32_t maskInVec_4 = svdup_n_u32(mask_4);
            const svbool_t mask_vec_4 = svcmpne_n_u32(true_vec, svand_u32_z(true_vec, maskFilter, maskInVec_4), 0);

            const svuint32_t maskInVec_5 = svdup_n_u32(mask_5);
            const svbool_t mask_vec_5 = svcmpne_n_u32(true_vec, svand_u32_z(true_vec, maskFilter, maskInVec_5), 0);

            const svuint32_t maskInVec_6 = svdup_n_u32(mask_6);
            const svbool_t mask_vec_6 = svcmpne_n_u32(true_vec, svand_u32_z(true_vec, maskFilter, maskInVec_6), 0);

            const svuint32_t maskInVec_7 = svdup_n_u32(mask_7);
            const svbool_t mask_vec_7 = svcmpne_n_u32(true_vec, svand_u32_z(true_vec, maskFilter, maskInVec_7), 0);

            const uint32_t increment = svcntp_b32(mask_vec, mask_vec);
            const uint32_t increment_1 = svcntp_b32(mask_vec_1, mask_vec_1);
            const uint32_t increment_2 = svcntp_b32(mask_vec_2, mask_vec_2);
            const uint32_t increment_3 = svcntp_b32(mask_vec_3, mask_vec_3);
            const uint32_t increment_4 = svcntp_b32(mask_vec_4, mask_vec_4);
            const uint32_t increment_5 = svcntp_b32(mask_vec_5, mask_vec_5);
            const uint32_t increment_6 = svcntp_b32(mask_vec_6, mask_vec_6);
            const uint32_t increment_7 = svcntp_b32(mask_vec_7, mask_vec_7);

            const svfloat32_t xvec = svld1(true_vec, &x[idxCol]);
            const svfloat32_t xvals = svcompact(mask_vec, xvec);
            const svfloat32_t xvals_1 = svcompact(mask_vec_1, xvec);
            const svfloat32_t xvals_2 = svcompact(mask_vec_2, xvec);
            const svfloat32_t xvals_3 = svcompact(mask_vec_3, xvec);
            const svfloat32_t xvals_4 = svcompact(mask_vec_4, xvec);
            const svfloat32_t xvals_5 = svcompact(mask_vec_5, xvec);
            const svfloat32_t xvals_6 = svcompact(mask_vec_6, xvec);
            const svfloat32_t xvals_7 = svcompact(mask_vec_7, xvec);

            const svfloat32_t block = svld1(svwhilelt_b32_s32(0, increment), values);
            values += increment;

            const svfloat32_t block_1 = svld1(svwhilelt_b32_s32(0, increment_1), values);
            values += increment_1;

            const svfloat32_t block_2 = svld1(svwhilelt_b32_s32(0, increment_2), values);
            values += increment_2;

            const svfloat32_t block_3 = svld1(svwhilelt_b32_s32(0, increment_3), values);
            values += increment_3;

            const svfloat32_t block_4 = svld1(svwhilelt_b32_s32(0, increment_4), values);
            values += increment_4;

            const svfloat32_t block_5 = svld1(svwhilelt_b32_s32(0, increment_5), values);
            values += increment_5;

            const svfloat32_t block_6 = svld1(svwhilelt_b32_s32(0, increment_6), values);
            values += increment_6;

            const svfloat32_t block_7 = svld1(svwhilelt_b32_s32(0, increment_7), values);
            values += increment_7;

            sum_vec = svmla_m(true_vec, sum_vec, block, xvals);
            sum_vec_1 = svmla_m(true_vec, sum_vec_1, block_1, xvals_1);
            sum_vec_2 = svmla_m(true_vec, sum_vec_2, block_2, xvals_2);
            sum_vec_3 = svmla_m(true_vec, sum_vec_3, block_3, xvals_3);
            sum_vec_4 = svmla_m(true_vec, sum_vec_4, block_4, xvals_4);
            sum_vec_5 = svmla_m(true_vec, sum_vec_5, block_5, xvals_5);
            sum_vec_6 = svmla_m(true_vec, sum_vec_6, block_6, xvals_6);
            sum_vec_7 = svmla_m(true_vec, sum_vec_7, block_7, xvals_7);

            headers += 20;
            }

            y[idxRow] += svaddv(true_vec, sum_vec);
            y[idxRow+1] += svaddv(true_vec, sum_vec_1);
            y[idxRow+2] += svaddv(true_vec, sum_vec_2);
            y[idxRow+3] += svaddv(true_vec, sum_vec_3);
            y[idxRow+4] += svaddv(true_vec, sum_vec_4);
            y[idxRow+5] += svaddv(true_vec, sum_vec_5);
            y[idxRow+6] += svaddv(true_vec, sum_vec_6);
            y[idxRow+7] += svaddv(true_vec, sum_vec_7);
    }
}

#else // USE_AVX512

#include <immintrin.h>

extern "C" void SPC5_opti_merge_double(double dest[], const double src[], const int nbValues){
    const int nbValuesVectorized = (nbValues/8)*8;
    for(int idxVal = 0 ; idxVal < nbValuesVectorized ; idxVal += 8){
            _mm512_storeu_pd(&dest[idxVal],_mm512_add_pd(_mm512_loadu_pd(&dest[idxVal]), _mm512_loadu_pd(&src[idxVal])));
    }
    for(int idxVal = nbValuesVectorized ; idxVal < nbValues ; idxVal += 1){
            dest[idxVal] += src[idxVal];
    }
}

extern "C" void SPC5_opti_merge_float(float dest[], const float src[], const int nbValues){
    const int nbValuesVectorized = (nbValues/16)*16;
    for(int idxVal = 0 ; idxVal < nbValuesVectorized ; idxVal += 16){
            _mm512_storeu_ps(&dest[idxVal],_mm512_add_ps(_mm512_loadu_ps(&dest[idxVal]), _mm512_loadu_ps(&src[idxVal])));
    }
    for(int idxVal = nbValuesVectorized ; idxVal < nbValues ; idxVal += 1){
            dest[idxVal] += src[idxVal];
    }
}



//////////////////////////////////////////////////////////////////////////

void core_SPC5_1rVc_Spmv_double(const long int nbRows, const int* rowSizes,
                                const unsigned char* headers,
                                const double* values,
                                const double* x, double* y){
    const __m512d zeros = _mm512_set1_pd(0);

    for (int idxRow = 0; idxRow < nbRows; ++idxRow) {
            __m512d sum_vec = zeros;

            for (int idxBlock = rowSizes[idxRow]; idxBlock < rowSizes[idxRow + 1]; ++idxBlock) {
                const int idxCol = *((const int *)headers);
                const unsigned char mask = headers[4];

                const __m512d xvals = _mm512_loadu_pd(&x[idxCol]);
                const __m512d block = _mm512_maskz_expandloadu_pd(mask, values);

                const int increment = _mm_popcnt_u32(mask);
                values += increment;

                sum_vec = _mm512_fmadd_pd(block, xvals, sum_vec);

                headers += 5;
            }

            const double sum = _mm512_reduce_add_pd(sum_vec);
            y[idxRow] += sum;
    }
}

void core_SPC5_1rVc_Spmv_float(const long int nbRows, const int* rowSizes,
                               const unsigned char* headers,
                               const float* values,
                               const float* x, float* y){
    const __m512 zeros = _mm512_set1_ps(0);

    for (int idxRow = 0; idxRow < nbRows; ++idxRow) {
            __m512 sum_vec = zeros;

            for (int idxBlock = rowSizes[idxRow]; idxBlock < rowSizes[idxRow + 1]; ++idxBlock) {
                const int idxCol = *((const int *)headers);
                const unsigned short mask = *(const unsigned short *)&headers[4];

                const __m512 xvals = _mm512_loadu_ps(&x[idxCol]);
                const __m512 block = _mm512_maskz_expandloadu_ps(mask, values);

                const int increment = _mm_popcnt_u32(mask);
                values += increment;

                sum_vec = _mm512_fmadd_ps(block, xvals, sum_vec);

                headers += 6;
            }

            const float sum = _mm512_reduce_add_ps(sum_vec);
            y[idxRow] += sum;
    }
}

//////////////////////////////////////////////////////////////////////////
void core_SPC5_2rVc_Spmv_double(const long int nbRows, const int* rowSizes,
                                const unsigned char* headers,
                                const double* values,
                                const double* x, double* y){
    const __m512d zeros = _mm512_set1_pd(0);

    for (int idxRow = 0; idxRow < nbRows; idxRow += 2) {
            const int idxRowBlock = idxRow/2;
            __m512d sum_vec = zeros;
            __m512d sum_vec_1 = zeros;

            for (int idxBlock = rowSizes[idxRowBlock]; idxBlock < rowSizes[idxRowBlock + 1]; ++idxBlock) {
                const int idxCol = *((const int *)headers);
                const unsigned char mask = headers[4];
                const unsigned char mask_1 = headers[5];

                const __m512d xvals = _mm512_loadu_pd(&x[idxCol]);

                const int increment = _mm_popcnt_u32(mask);
                sum_vec = _mm512_fmadd_pd(_mm512_maskz_expandloadu_pd(mask, values), xvals, sum_vec);
                values += increment;

                const int increment_1 = _mm_popcnt_u32(mask_1);
                sum_vec_1 = _mm512_fmadd_pd(_mm512_maskz_expandloadu_pd(mask_1, values), xvals, sum_vec_1);
                values += increment_1;

                headers += 6;
            }

            y[idxRow] += _mm512_reduce_add_pd(sum_vec);
            y[idxRow+1] += _mm512_reduce_add_pd(sum_vec_1);
    }
}



void core_SPC5_2rVc_Spmv_float(const long int nbRows, const int* rowSizes,
                               const unsigned char* headers,
                               const float* values,
                               const float* x, float* y){
    const __m512 zeros = _mm512_set1_ps(0);

    for (int idxRow = 0; idxRow < nbRows; idxRow += 2) {
            const int idxRowBlock = idxRow/2;
            __m512 sum_vec = zeros;
            __m512 sum_vec_1 = zeros;

            for (int idxBlock = rowSizes[idxRowBlock]; idxBlock < rowSizes[idxRowBlock + 1]; ++idxBlock) {
                const int idxCol = *((const int *)headers);
                const unsigned short mask = *(const unsigned short *)&headers[4];
                const unsigned short mask_1 = *(const unsigned short *)&headers[6];

                const __m512 xvals = _mm512_loadu_ps(&x[idxCol]);

                const int increment = _mm_popcnt_u32(mask);
                sum_vec = _mm512_fmadd_ps(_mm512_maskz_expandloadu_ps(mask, values), xvals, sum_vec);
                values += increment;

                const int increment_1 = _mm_popcnt_u32(mask_1);
                sum_vec_1 = _mm512_fmadd_ps(_mm512_maskz_expandloadu_ps(mask_1, values), xvals, sum_vec_1);
                values += increment_1;

                headers += 8;
            }

            y[idxRow] += _mm512_reduce_add_ps(sum_vec);
            y[idxRow+1] += _mm512_reduce_add_ps(sum_vec_1);
    }
}

//////////////////////////////////////////////////////////////////////////

void core_SPC5_4rVc_Spmv_double(const long int nbRows, const int* rowSizes,
                                const unsigned char* headers,
                                const double* values,
                                const double* x, double* y){
    const __m512d zeros = _mm512_set1_pd(0);

    for (int idxRow = 0; idxRow < nbRows; idxRow += 4) {
            const int idxRowBlock = idxRow/4;
            __m512d sum_vec = zeros;
            __m512d sum_vec_1 = zeros;
            __m512d sum_vec_2 = zeros;
            __m512d sum_vec_3 = zeros;

            for (int idxBlock = rowSizes[idxRowBlock]; idxBlock < rowSizes[idxRowBlock + 1]; ++idxBlock) {
                const int idxCol = *((const int *)headers);
                const unsigned char mask = headers[4];
                const unsigned char mask_1 = headers[5];
                const unsigned char mask_2 = headers[6];
                const unsigned char mask_3 = headers[7];

                const __m512d xvals = _mm512_loadu_pd(&x[idxCol]);

                const int increment = _mm_popcnt_u32(mask);
                sum_vec = _mm512_fmadd_pd(_mm512_maskz_expandloadu_pd(mask, values), xvals, sum_vec);
                values += increment;

                const int increment_1 = _mm_popcnt_u32(mask_1);
                sum_vec_1 = _mm512_fmadd_pd(_mm512_maskz_expandloadu_pd(mask_1, values), xvals, sum_vec_1);
                values += increment_1;

                const int increment_2 = _mm_popcnt_u32(mask_2);
                sum_vec_2 = _mm512_fmadd_pd(_mm512_maskz_expandloadu_pd(mask_2, values), xvals, sum_vec_2);
                values += increment_2;

                const int increment_3 = _mm_popcnt_u32(mask_3);
                sum_vec_3 = _mm512_fmadd_pd(_mm512_maskz_expandloadu_pd(mask_3, values), xvals, sum_vec_3);
                values += increment_3;

                headers += 8;
            }

            y[idxRow] += _mm512_reduce_add_pd(sum_vec);
            y[idxRow+1] += _mm512_reduce_add_pd(sum_vec_1);
            y[idxRow+2] += _mm512_reduce_add_pd(sum_vec_2);
            y[idxRow+3] += _mm512_reduce_add_pd(sum_vec_3);
    }
}




void core_SPC5_4rVc_Spmv_float(const long int nbRows, const int* rowSizes,
                               const unsigned char* headers,
                               const float* values,
                               const float* x, float* y){
    const __m512 zeros = _mm512_set1_ps(0);

    for (int idxRow = 0; idxRow < nbRows; idxRow += 4) {
            const int idxRowBlock = idxRow/4;
            __m512 sum_vec = zeros;
            __m512 sum_vec_1 = zeros;
            __m512 sum_vec_2 = zeros;
            __m512 sum_vec_3 = zeros;

            for (int idxBlock = rowSizes[idxRowBlock]; idxBlock < rowSizes[idxRowBlock + 1]; ++idxBlock) {
                const int idxCol = *((const int *)headers);
                const unsigned short mask = *(const unsigned short *)&headers[4];
                const unsigned short mask_1 = *(const unsigned short *)&headers[6];
                const unsigned short mask_2 = *(const unsigned short *)&headers[8];
                const unsigned short mask_3 = *(const unsigned short *)&headers[10];

                const __m512 xvals = _mm512_loadu_ps(&x[idxCol]);

                const int increment = _mm_popcnt_u32(mask);
                sum_vec = _mm512_fmadd_ps(_mm512_maskz_expandloadu_ps(mask, values), xvals, sum_vec);
                values += increment;

                const int increment_1 = _mm_popcnt_u32(mask_1);
                sum_vec_1 = _mm512_fmadd_ps(_mm512_maskz_expandloadu_ps(mask_1, values), xvals, sum_vec_1);
                values += increment_1;

                const int increment_2 = _mm_popcnt_u32(mask_2);
                sum_vec_2 = _mm512_fmadd_ps(_mm512_maskz_expandloadu_ps(mask_2, values), xvals, sum_vec_2);
                values += increment_2;

                const int increment_3 = _mm_popcnt_u32(mask_3);
                sum_vec_3 = _mm512_fmadd_ps(_mm512_maskz_expandloadu_ps(mask_3, values), xvals, sum_vec_3);
                values += increment_3;

                headers += 12;
            }

            y[idxRow] += _mm512_reduce_add_ps(sum_vec);
            y[idxRow+1] += _mm512_reduce_add_ps(sum_vec_1);
            y[idxRow+2] += _mm512_reduce_add_ps(sum_vec_2);
            y[idxRow+3] += _mm512_reduce_add_ps(sum_vec_3);
    }
}


//////////////////////////////////////////////////////////////////////////

void core_SPC5_8rVc_Spmv_double(const long int nbRows, const int* rowSizes,
                                const unsigned char* headers,
                                const double* values,
                                const double* x, double* y){
    const __m512d zeros = _mm512_set1_pd(0);

    for (int idxRow = 0; idxRow < nbRows; idxRow += 8) {
            const int idxRowBlock = idxRow/8;
            __m512d sum_vec = zeros;
            __m512d sum_vec_1 = zeros;
            __m512d sum_vec_2 = zeros;
            __m512d sum_vec_3 = zeros;
            __m512d sum_vec_4 = zeros;
            __m512d sum_vec_5 = zeros;
            __m512d sum_vec_6 = zeros;
            __m512d sum_vec_7 = zeros;

            for (int idxBlock = rowSizes[idxRowBlock]; idxBlock < rowSizes[idxRowBlock + 1]; ++idxBlock) {
                const int idxCol = *((const int *)headers);
                const unsigned char mask = headers[4];
                const unsigned char mask_1 = headers[5];
                const unsigned char mask_2 = headers[6];
                const unsigned char mask_3 = headers[7];
                const unsigned char mask_4 = headers[8];
                const unsigned char mask_5 = headers[9];
                const unsigned char mask_6 = headers[10];
                const unsigned char mask_7 = headers[11];

                const __m512d xvals = _mm512_loadu_pd(&x[idxCol]);

                const int increment = _mm_popcnt_u32(mask);
                sum_vec = _mm512_fmadd_pd(_mm512_maskz_expandloadu_pd(mask, values), xvals, sum_vec);
                values += increment;

                const int increment_1 = _mm_popcnt_u32(mask_1);
                sum_vec_1 = _mm512_fmadd_pd(_mm512_maskz_expandloadu_pd(mask_1, values), xvals, sum_vec_1);
                values += increment_1;

                const int increment_2 = _mm_popcnt_u32(mask_2);
                sum_vec_2 = _mm512_fmadd_pd(_mm512_maskz_expandloadu_pd(mask_2, values), xvals, sum_vec_2);
                values += increment_2;

                const int increment_3 = _mm_popcnt_u32(mask_3);
                sum_vec_3 = _mm512_fmadd_pd(_mm512_maskz_expandloadu_pd(mask_3, values), xvals, sum_vec_3);
                values += increment_3;

                const int increment_4 = _mm_popcnt_u32(mask_4);
                sum_vec_4 = _mm512_fmadd_pd(_mm512_maskz_expandloadu_pd(mask_4, values), xvals, sum_vec_4);
                values += increment_4;

                const int increment_5 = _mm_popcnt_u32(mask_5);
                sum_vec_5 = _mm512_fmadd_pd(_mm512_maskz_expandloadu_pd(mask_5, values), xvals, sum_vec_5);
                values += increment_5;

                const int increment_6 = _mm_popcnt_u32(mask_6);
                sum_vec_6 = _mm512_fmadd_pd(_mm512_maskz_expandloadu_pd(mask_6, values), xvals, sum_vec_6);
                values += increment_6;

                const int increment_7 = _mm_popcnt_u32(mask_7);
                sum_vec_7 = _mm512_fmadd_pd(_mm512_maskz_expandloadu_pd(mask_7, values), xvals, sum_vec_7);
                values += increment_7;

                headers += 12;
            }

            y[idxRow] += _mm512_reduce_add_pd(sum_vec);
            y[idxRow+1] += _mm512_reduce_add_pd(sum_vec_1);
            y[idxRow+2] += _mm512_reduce_add_pd(sum_vec_2);
            y[idxRow+3] += _mm512_reduce_add_pd(sum_vec_3);
            y[idxRow+4] += _mm512_reduce_add_pd(sum_vec_4);
            y[idxRow+5] += _mm512_reduce_add_pd(sum_vec_5);
            y[idxRow+6] += _mm512_reduce_add_pd(sum_vec_6);
            y[idxRow+7] += _mm512_reduce_add_pd(sum_vec_7);
    }
}




void core_SPC5_8rVc_Spmv_float(const long int nbRows, const int* rowSizes,
                               const unsigned char* headers,
                               const float* values,
                               const float* x, float* y){
    const __m512 zeros = _mm512_set1_ps(0);

    for (int idxRow = 0; idxRow < nbRows; idxRow += 8) {
            const int idxRowBlock = idxRow/8;
            __m512 sum_vec = zeros;
            __m512 sum_vec_1 = zeros;
            __m512 sum_vec_2 = zeros;
            __m512 sum_vec_3 = zeros;
            __m512 sum_vec_4 = zeros;
            __m512 sum_vec_5 = zeros;
            __m512 sum_vec_6 = zeros;
            __m512 sum_vec_7 = zeros;

            for (int idxBlock = rowSizes[idxRowBlock]; idxBlock < rowSizes[idxRowBlock + 1]; ++idxBlock) {
                const int idxCol = *((const int *)headers);
                const unsigned short mask = *(const unsigned short *)&headers[4];
                const unsigned short mask_1 = *(const unsigned short *)&headers[6];
                const unsigned short mask_2 = *(const unsigned short *)&headers[8];
                const unsigned short mask_3 = *(const unsigned short *)&headers[10];
                const unsigned short mask_4 = *(const unsigned short *)&headers[12];
                const unsigned short mask_5 = *(const unsigned short *)&headers[14];
                const unsigned short mask_6 = *(const unsigned short *)&headers[16];
                const unsigned short mask_7 = *(const unsigned short *)&headers[18];

                const __m512 xvals = _mm512_loadu_ps(&x[idxCol]);

                const int increment = _mm_popcnt_u32(mask);
                sum_vec = _mm512_fmadd_ps(_mm512_maskz_expandloadu_ps(mask, values), xvals, sum_vec);
                values += increment;

                const int increment_1 = _mm_popcnt_u32(mask_1);
                sum_vec_1 = _mm512_fmadd_ps(_mm512_maskz_expandloadu_ps(mask_1, values), xvals, sum_vec_1);
                values += increment_1;

                const int increment_2 = _mm_popcnt_u32(mask_2);
                sum_vec_2 = _mm512_fmadd_ps(_mm512_maskz_expandloadu_ps(mask_2, values), xvals, sum_vec_2);
                values += increment_2;

                const int increment_3 = _mm_popcnt_u32(mask_3);
                sum_vec_3 = _mm512_fmadd_ps(_mm512_maskz_expandloadu_ps(mask_3, values), xvals, sum_vec_3);
                values += increment_3;

                const int increment_4 = _mm_popcnt_u32(mask_4);
                sum_vec_4 = _mm512_fmadd_ps(_mm512_maskz_expandloadu_ps(mask_4, values), xvals, sum_vec_4);
                values += increment_4;

                const int increment_5 = _mm_popcnt_u32(mask_5);
                sum_vec_5 = _mm512_fmadd_ps(_mm512_maskz_expandloadu_ps(mask_5, values), xvals, sum_vec_5);
                values += increment_5;

                const int increment_6 = _mm_popcnt_u32(mask_6);
                sum_vec_6 = _mm512_fmadd_ps(_mm512_maskz_expandloadu_ps(mask_6, values), xvals, sum_vec_6);
                values += increment_6;

                const int increment_7 = _mm_popcnt_u32(mask_7);
                sum_vec_7 = _mm512_fmadd_ps(_mm512_maskz_expandloadu_ps(mask_7, values), xvals, sum_vec_7);
                values += increment_7;

                headers += 20;
            }

            y[idxRow] += _mm512_reduce_add_ps(sum_vec);
            y[idxRow+1] += _mm512_reduce_add_ps(sum_vec_1);
            y[idxRow+2] += _mm512_reduce_add_ps(sum_vec_2);
            y[idxRow+3] += _mm512_reduce_add_ps(sum_vec_3);
            y[idxRow+4] += _mm512_reduce_add_ps(sum_vec_4);
            y[idxRow+5] += _mm512_reduce_add_ps(sum_vec_5);
            y[idxRow+6] += _mm512_reduce_add_ps(sum_vec_6);
            y[idxRow+7] += _mm512_reduce_add_ps(sum_vec_7);
    }
}


#endif // USE_AVX512

#endif
