#ifndef SPC5_HPP
#define SPC5_HPP

#include <memory>       // For unique_ptr
#include <algorithm>    // For sort
#include <cassert>      // For assert function
#include <limits>       // For <int>::max
#include <vector>       // For vector class
#include <stdexcept>    // For invalid argument

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#else
#include "farm_sve.h"
#endif /* __ARM_FEATURE_SVE */

#ifndef SPC5_NB_BITS_IN_VEC
#define SPC5_NB_BITS_IN_VEC 512
#endif

//////////////////////////////////////////////////////////////////////////////
/// Matrix structure
//////////////////////////////////////////////////////////////////////////////

enum SPC5_MATRIX_TYPE{
    UNDEFINED_FORMAT,
    FORMAT_CSR,
    FORMAT_1rVc_WT,
    FORMAT_2rV2c_WT,
    FORMAT_2rV2c,
    FORMAT_2rVc,
    FORMAT_4rV2c,
    FORMAT_4rVc,
    FORMAT_8rV2c
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
    std::unique_ptr<int[]> rowsSizeCpy;//< Copy of the usual "rowsSize/rowptr" (of size numberOfRows+1)

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
void core_CSR_to_SPC5_rV2c(SPC5Mat<ValueType>* csr){
    static_assert((nbRowsPerBlock&1) == 0 , "nbRowsPerBlock must multiple of 2");

    assert(csr->format == SPC5_MATRIX_TYPE::FORMAT_CSR);

    std::vector<unsigned char> blocks;
    blocks.reserve((sizeof(int)+sizeof(short))*csr->numberOfNNZ/ValPerVec<ValueType>::size);

    csr->rowsSizeCpy.reset(new int[csr->numberOfRows+1]);
    memcpy(csr->rowsSizeCpy.get(), csr->rowsSize.get(), sizeof(int)*(csr->numberOfRows+1));

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


            typename SPC5Mat_Mask<ValueType>::type valMask[nbRowsPerBlock/2] = {0u};
            for(int idxSubRow = 0 ; idxSubRow < nbRowsPerBlock ; ++idxSubRow){
                if(idxRow + idxSubRow < csr->numberOfRows){
                    while(idxVal[idxSubRow] < csr->rowsSize[idxRow+idxSubRow+1]
                          && csr->valuesColumnIndexes[idxVal[idxSubRow]] < idxCol+ValPerVec<ValueType>::size/2){
                        assert(globalIdxValues < csr->numberOfNNZ);
                        newValues[globalIdxValues++] = csr->values[idxVal[idxSubRow]];
                        valMask[idxSubRow/2] |= (1u << (csr->valuesColumnIndexes[idxVal[idxSubRow]]-idxCol
                                                 + (idxSubRow%2)*ValPerVec<ValueType>::size/2));
                        idxVal[idxSubRow] += 1;
                    }
                }
            }

            blocks.insert(blocks.end(), (unsigned char*)&idxCol, (unsigned char*)(&idxCol+1));
            blocks.insert(blocks.end(), (unsigned char*)&valMask[0], (unsigned char*)(&valMask[nbRowsPerBlock/2]));
            currentNbBlocks += 1;
        }

        csr->rowsSize[idxRow] = previousNbBlocks;
        previousNbBlocks += currentNbBlocks;

        for(int idxSubRow = 1 ; idxSubRow < nbRowsPerBlock ; ++idxSubRow){
            if(idxRow + idxSubRow < csr->numberOfRows){
                csr->rowsSize[idxRow+idxSubRow] = previousNbBlocks;
            }
        }
    }

    csr->numberOfBlocks = previousNbBlocks;
    csr->values = std::move(newValues);
    csr->rowsSize[csr->numberOfRows] = previousNbBlocks;
    csr->blocksColumnIndexesWithMasks = (ToUniquePtr(blocks));
}


template <class ValueType, int nbRowsPerBlock>
inline void core_CSR_to_SPC5_rVc(SPC5Mat<ValueType>* csr){
    assert(csr->format == SPC5_MATRIX_TYPE::FORMAT_CSR);
    std::vector<unsigned char> blocks;
    blocks.reserve((sizeof(int)+sizeof(short))*csr->numberOfNNZ/ValPerVec<ValueType>::size);

    std::unique_ptr<ValueType[]> newValues(new ValueType[csr->numberOfNNZ]);
    int globalIdxValues = 0;

    csr->rowsSizeCpy.reset(new int[csr->numberOfRows+1]);
    memcpy(csr->rowsSizeCpy.get(), csr->rowsSize.get(), sizeof(int)*(csr->numberOfRows+1));

    int previousNbBlocks = 0;
    for(int idxRow = 0 ; idxRow < csr->numberOfRows ; idxRow += nbRowsPerBlock){
        int currentNbBlocks = 0;
        int idxVal[nbRowsPerBlock] = {0};
        for(int idxSubRow = 0 ; idxSubRow < nbRowsPerBlock ; ++idxSubRow){
            if(idxRow + idxSubRow < csr->numberOfRows){
                idxVal[idxSubRow] = csr->rowsSize[idxRow+idxSubRow];
            }
        }

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
                        valMask[idxSubRow] |= (1u << (csr->valuesColumnIndexes[idxVal[idxSubRow]]-idxCol));
                        idxVal[idxSubRow] += 1;
                    }
                }
            }

            blocks.insert(blocks.end(), (unsigned char*)&idxCol, (unsigned char*)(&idxCol+1));
            blocks.insert(blocks.end(), (unsigned char*)&valMask[0], (unsigned char*)(&valMask[nbRowsPerBlock]));
            currentNbBlocks += 1;
        }

        csr->rowsSize[idxRow] = previousNbBlocks;
        previousNbBlocks += currentNbBlocks;

        for(int idxSubRow = 1 ; idxSubRow < nbRowsPerBlock ; ++idxSubRow){
            if(idxRow + idxSubRow < csr->numberOfRows){
                csr->rowsSize[idxRow+idxSubRow] = previousNbBlocks;
            }
        }
    }

    csr->numberOfBlocks = previousNbBlocks;
    csr->values = std::move(newValues);
    csr->rowsSize[csr->numberOfRows] = previousNbBlocks;
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

template <class ValueType, int nbRowsPerBlock>
inline void core_SPC5_rV2c_Spmv_scalar(const SPC5Mat<ValueType>& mat, const ValueType *x , ValueType *y ){
    static_assert((nbRowsPerBlock&1) == 0 , "nbRowsPerBlock must multiple of 2");

    int idxVal = 0;
    for(int idxRow = 0 ; idxRow < mat.numberOfRows ; idxRow += nbRowsPerBlock){
        ValueType sum[nbRowsPerBlock] = {0};

        for(int idxBlock = mat.rowsSize[idxRow]; idxBlock < mat.rowsSize[idxRow+1] ; ++idxBlock){
            const int idxCol = *(int*)&mat.blocksColumnIndexesWithMasks[idxBlock*(sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*nbRowsPerBlock/2)];

            for(int idxRowBlock = 0 ; idxRowBlock < nbRowsPerBlock ; idxRowBlock += 2){
                const typename SPC5Mat_Mask<ValueType>::type valMask = *(typename SPC5Mat_Mask<ValueType>::type*)&mat.blocksColumnIndexesWithMasks[idxBlock*(sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*nbRowsPerBlock/2) + sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*(idxRowBlock/2)];

                for(int idxvv = 0 ; idxvv < ValPerVec<ValueType>::size/2 ; ++idxvv){
                    if((1 << idxvv) & valMask){
                        sum[idxRowBlock] += x[idxCol+idxvv] * mat.values[idxVal];
                        idxVal += 1;
                    }
                }
                for(int idxvv = ValPerVec<ValueType>::size/2 ; idxvv < ValPerVec<ValueType>::size ; ++idxvv){
                    if((1 << idxvv) & valMask){
                        sum[idxRowBlock+1] += x[idxCol+idxvv-ValPerVec<ValueType>::size/2] * mat.values[idxVal];
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
        for(int idxBlock = mat.rowsSize[idxRow]; idxBlock < mat.rowsSize[idxRow+1] ; ++idxBlock){
            const int idxCol = *(int*)&mat.blocksColumnIndexesWithMasks[idxBlock*(sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*nbRowsPerBlock)];
            for(int idxRowBlock = 0 ; idxRowBlock < nbRowsPerBlock ; idxRowBlock += 1){
                const typename SPC5Mat_Mask<ValueType>::type valMask = *(typename SPC5Mat_Mask<ValueType>::type*)&mat.blocksColumnIndexesWithMasks[idxBlock*(sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*nbRowsPerBlock) + sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*(idxRowBlock)];
                for(int idxvv = 0 ; idxvv < ValPerVec<ValueType>::size ; ++idxvv){
                    if((1 << idxvv) & valMask){
                        func(idxRow+idxRowBlock, idxCol+idxvv, mat.values[idxVal]);
                        idxVal += 1;
                    }
                }
            }
        }
    }
}

template <class ValueType, int nbRowsPerBlock, class FuncType>
inline void core_SPC5_rV2c_iterate(SPC5Mat<ValueType>& mat, const FuncType&& func){
    static_assert((nbRowsPerBlock&1) == 0 , "nbRowsPerBlock must multiple of 2");

    int idxVal = 0;
    for(int idxRow = 0 ; idxRow < mat.numberOfRows ; idxRow += nbRowsPerBlock){
        for(int idxBlock = mat.rowsSize[idxRow]; idxBlock < mat.rowsSize[idxRow+1] ; ++idxBlock){
            const int idxCol = *(int*)&mat.blocksColumnIndexesWithMasks[idxBlock*(sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*nbRowsPerBlock/2)];

            for(int idxRowBlock = 0 ; idxRowBlock < nbRowsPerBlock ; idxRowBlock += 2){
                const typename SPC5Mat_Mask<ValueType>::type valMask = *(typename SPC5Mat_Mask<ValueType>::type*)&mat.blocksColumnIndexesWithMasks[idxBlock*(sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*nbRowsPerBlock/2) + sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*(idxRowBlock/2)];

                for(int idxvv = 0 ; idxvv < ValPerVec<ValueType>::size/2 ; ++idxvv){
                    if((1 << idxvv) & valMask){
                        func(idxRow+idxRowBlock, idxCol+idxvv,  mat.values[idxVal]);
                        idxVal += 1;
                    }
                }
                for(int idxvv = ValPerVec<ValueType>::size/2 ; idxvv < ValPerVec<ValueType>::size ; ++idxvv){
                    if((1 << idxvv) & valMask){
                        func(idxRow+idxRowBlock+1, idxCol+idxvv-ValPerVec<ValueType>::size/2,  mat.values[idxVal]);
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
#ifdef SPLIT_NUMA
    std::unique_ptr<ValueType[]> threadValues;
    std::unique_ptr<int[]> threadRowsSize;
    std::unique_ptr<int[]> threadValuesColumnIndexes;
    std::unique_ptr<unsigned char[]> threadBlocksColumnIndexesWithMasks;
#endif
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
                && std::abs((double(idxCurrentThread+1)*blocksPerThreads)-mat.rowsSize[idxRow])
                    < std::abs((double(idxCurrentThread+1)*blocksPerThreads)-mat.rowsSize[idxRow+1])){
            intervals[idxCurrentThread].numberOfRows = idxRow - intervals[idxCurrentThread].startingRow;

            idxCurrentThread += 1;
            intervals[idxCurrentThread].startingRow = idxRow;
            intervals[idxCurrentThread].valuesOffset = idxVal;
        }

        for(int idxBlock = mat.rowsSize[idxRow]; idxBlock < mat.rowsSize[idxRow+1] ; ++idxBlock){
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
        intervals[omp_get_thread_num()].threadY.reset(new ValueType[intervals[omp_get_thread_num()].numberOfRows + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y]());
#ifdef SPLIT_NUMA
        intervals[omp_get_thread_num()].threadRowsSize.reset(new int[intervals[omp_get_thread_num()].numberOfRows + 1]());
        memcpy(intervals[omp_get_thread_num()].threadRowsSize.get(),
                mat.rowsSize.get()+intervals[omp_get_thread_num()].startingRow,
                sizeof(int)*(intervals[omp_get_thread_num()].numberOfRows + 1));

        const int nbBlocks = mat.rowsSize[intervals[omp_get_thread_num()].startingRow + intervals[omp_get_thread_num()].numberOfRows]
                           - mat.rowsSize[intervals[omp_get_thread_num()].startingRow];
        const int maskSize = (sizeof(ValueType)==4?2:1);

        intervals[omp_get_thread_num()].threadBlocksColumnIndexesWithMasks.reset(new unsigned char[nbBlocks*(4+maskSize*nbRowsPerBlock)]());
        memcpy(intervals[omp_get_thread_num()].threadBlocksColumnIndexesWithMasks.get(),
                mat.blocksColumnIndexesWithMasks.get()+mat.rowsSize[intervals[omp_get_thread_num()].startingRow]*(4+maskSize*nbRowsPerBlock),
                sizeof(unsigned char)*(nbBlocks*(4+maskSize*nbRowsPerBlock)));

        const int nbValues = (omp_get_thread_num()+1 != nbThreads ? intervals[omp_get_thread_num()+1].valuesOffset : idxVal)
                - intervals[omp_get_thread_num()].valuesOffset;

        intervals[omp_get_thread_num()].threadValues.reset(new ValueType[nbValues]());
        memcpy(intervals[omp_get_thread_num()].threadValues.get(),
                mat.values.get()+intervals[omp_get_thread_num()].valuesOffset,
                sizeof(ValueType)*nbValues);
#endif
    }

    return intervals;
}

template <class ValueType, int nbRowsPerBlock>
inline std::vector<ThreadInterval<ValueType>> core_SPC5_rV2c_threadsplit(const SPC5Mat<ValueType>& mat, const int nbThreads){
    static_assert((nbRowsPerBlock&1) == 0 , "nbRowsPerBlock must multiple of 2");

    std::vector<ThreadInterval<ValueType>> intervals(nbThreads);
    int idxCurrentThread = 0;
    const double blocksPerThreads = double(mat.numberOfBlocks)/double(nbThreads);
    intervals[0].startingRow = 0;
    intervals[0].valuesOffset = 0;

    int idxVal = 0;
    for(int idxRow = 0 ; idxRow < mat.numberOfRows ; idxRow += nbRowsPerBlock){
        if(idxCurrentThread != nbThreads-1 && idxRow // not the last thread
                && std::abs((double(idxCurrentThread+1)*blocksPerThreads)-mat.rowsSize[idxRow])
                    < std::abs((double(idxCurrentThread+1)*blocksPerThreads)-mat.rowsSize[idxRow+1])){
            intervals[idxCurrentThread].numberOfRows = idxRow - intervals[idxCurrentThread].startingRow;

            idxCurrentThread += 1;
            intervals[idxCurrentThread].startingRow = idxRow;
            intervals[idxCurrentThread].valuesOffset = idxVal;
        }

        for(int idxBlock = mat.rowsSize[idxRow]; idxBlock < mat.rowsSize[idxRow+1] ; ++idxBlock){
            //const int idxCol = *(int*)&mat.blocksColumnIndexesWithMasks[idxBlock*(sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*nbRowsPerBlock/2)];

            for(int idxRowBlock = 0 ; idxRowBlock < nbRowsPerBlock ; idxRowBlock += 2){
                const typename SPC5Mat_Mask<ValueType>::type valMask = *(typename SPC5Mat_Mask<ValueType>::type*)&mat.blocksColumnIndexesWithMasks[idxBlock*(sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*nbRowsPerBlock/2) + sizeof(int)+sizeof(typename SPC5Mat_Mask<ValueType>::type)*(idxRowBlock/2)];

                for(int idxvv = 0 ; idxvv < ValPerVec<ValueType>::size/2 ; ++idxvv){
                    if((1 << idxvv) & valMask){
                        idxVal += 1;
                    }
                }
                for(int idxvv = ValPerVec<ValueType>::size/2 ; idxvv < ValPerVec<ValueType>::size ; ++idxvv){
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
        intervals[omp_get_thread_num()].threadY.reset(new ValueType[intervals[omp_get_thread_num()].numberOfRows + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y]());
#ifdef SPLIT_NUMA
        intervals[omp_get_thread_num()].threadRowsSize.reset(new int[intervals[omp_get_thread_num()].numberOfRows + 1]());
        memcpy(intervals[omp_get_thread_num()].threadRowsSize.get(),
                mat.rowsSize.get()+intervals[omp_get_thread_num()].startingRow,
                sizeof(int)*(intervals[omp_get_thread_num()].numberOfRows + 1));

        const int nbBlocks = mat.rowsSize[intervals[omp_get_thread_num()].startingRow + intervals[omp_get_thread_num()].numberOfRows]
                           - mat.rowsSize[intervals[omp_get_thread_num()].startingRow];
        const int maskSize = (sizeof(ValueType)==4?2:1);

        intervals[omp_get_thread_num()].threadBlocksColumnIndexesWithMasks.reset(new unsigned char[nbBlocks*(4+maskSize*nbRowsPerBlock/2)]());
        memcpy(intervals[omp_get_thread_num()].threadBlocksColumnIndexesWithMasks.get(),
                mat.blocksColumnIndexesWithMasks.get()+mat.rowsSize[intervals[omp_get_thread_num()].startingRow]*(4+maskSize*nbRowsPerBlock/2),
                sizeof(unsigned char)*(nbBlocks*(4+maskSize*nbRowsPerBlock/2)));

        const int nbValues = (omp_get_thread_num()+1 != nbThreads ? intervals[omp_get_thread_num()+1].valuesOffset : idxVal)
                - intervals[omp_get_thread_num()].valuesOffset;

        intervals[omp_get_thread_num()].threadValues.reset(new ValueType[nbValues]());
        memcpy(intervals[omp_get_thread_num()].threadValues.get(),
                mat.values.get()+intervals[omp_get_thread_num()].valuesOffset,
                sizeof(ValueType)*nbValues);
#endif
    }

    return intervals;
}

template <class ValueType>
void SPC5_opti_merge(ValueType dest[], const ValueType src[], const int nbValues);

template <>
inline void SPC5_opti_merge<double>(double dest[], const double src[], const int nbValues){
    for(int idxVal = 0 ; idxVal < nbValues ; idxVal += 1){
        dest[idxVal] += src[idxVal];
    }
}

template <>
inline void SPC5_opti_merge<float>(float dest[], const float src[], const int nbValues){
    for(int idxVal = 0 ; idxVal < nbValues ; idxVal += 1){
        dest[idxVal] += src[idxVal];
    }
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


    csr->rowsSizeCpy.reset(new int[csr->numberOfRows+1]);
    memcpy(csr->rowsSizeCpy.get(), csr->rowsSize.get(), sizeof(int)*(csr->numberOfRows+1));


    int previousNbBlocks = 0;
    for(int idxRow = 0 ; idxRow < csr->numberOfRows ; ++idxRow){
        int currentNbBlocks = 0;
        int idxVal = csr->rowsSize[idxRow];
        while(idxVal < csr->rowsSize[idxRow+1]){
            int idxCol = csr->valuesColumnIndexes[idxVal];
            typename SPC5Mat_Mask<ValueType>::type valMask = 1u;
            idxVal += 1;
            while(idxVal < csr->rowsSize[idxRow+1] && csr->valuesColumnIndexes[idxVal] < idxCol+ValPerVec<ValueType>::size){
                valMask |= (static_cast<typename SPC5Mat_Mask<ValueType>::type>(1u) << (csr->valuesColumnIndexes[idxVal]-idxCol));
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


extern "C" void core_SPC5_1rVc_Spmv_double_v2(const long int nbRows, const int* rowsSizes,
                                           const unsigned char* blocksColumnIndexesWithMasks,
                                           const double* values,
                                           const double* x, double* y,
                                              const int* rowptr);
extern "C" void core_SPC5_1rVc_Spmv_float_v2(const long int nbRows, const int* rowsSizes,
                                          const unsigned char* blocksColumnIndexesWithMasks,
                                          const float* values,
                                          const float* x, float* y,
                                             const int* rowptr);

template <class ValueType>
inline void SPC5_1rVc_Spmv_v2(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[]);

template <>
inline void SPC5_1rVc_Spmv_v2<double>(const SPC5Mat<double>& mat, const double x[], double y[]){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_1rVc_WT);
    core_SPC5_1rVc_Spmv_double_v2(mat.numberOfRows, mat.rowsSize.get(),
                               mat.blocksColumnIndexesWithMasks.get(), mat.values.get(),
                                  x, y, mat.rowsSizeCpy.get());
}

template <>
inline void SPC5_1rVc_Spmv_v2<float>(const SPC5Mat<float>& mat, const float x[], float y[]){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_1rVc_WT);
    core_SPC5_1rVc_Spmv_float_v2(mat.numberOfRows, mat.rowsSize.get(),
                              mat.blocksColumnIndexesWithMasks.get(), mat.values.get(),
                              x, y, mat.rowsSizeCpy.get());
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
#ifdef SPLIT_NUMA
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
#else
template <>
inline void SPC5_1rVc_Spmv_omp<double>(const SPC5Mat<double>& mat, const double x[], double y[],
                                   const std::vector<ThreadInterval<double>>& threadsVecs){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_1rVc_WT);
    const int numThreads = int(threadsVecs.size());
#pragma omp parallel num_threads(numThreads)
    {
        memset(threadsVecs[omp_get_thread_num()].threadY.get(), 0, sizeof(double)*threadsVecs[omp_get_thread_num()].numberOfRows);

        core_SPC5_1rVc_Spmv_double(threadsVecs[omp_get_thread_num()].numberOfRows,
                                       mat.rowsSize.get()+threadsVecs[omp_get_thread_num()].startingRow,
                                       mat.blocksColumnIndexesWithMasks.get()+mat.rowsSize[threadsVecs[omp_get_thread_num()].startingRow]*5,
                                       mat.values.get()+threadsVecs[omp_get_thread_num()].valuesOffset,
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
                                       mat.rowsSize.get()+threadsVecs[omp_get_thread_num()].startingRow,
                                       mat.blocksColumnIndexesWithMasks.get()+mat.rowsSize[threadsVecs[omp_get_thread_num()].startingRow]*6,
                                       mat.values.get()+threadsVecs[omp_get_thread_num()].valuesOffset,
                                       x, threadsVecs[omp_get_thread_num()].threadY.get());

        SPC5_opti_merge(&y[threadsVecs[omp_get_thread_num()].startingRow], threadsVecs[omp_get_thread_num()].threadY.get(),
                threadsVecs[omp_get_thread_num()].numberOfRows);
    }
}
#endif

#endif


//////////////////////////////////////////////////////////////////////////////
/// 2rV2c_wt => 2 rows, VEC/2 columns
//////////////////////////////////////////////////////////////////////////////

template <class ValueType>
inline void CSR_to_SPC5_2rV2c_wt(SPC5Mat<ValueType>* csr){
    core_CSR_to_SPC5_rV2c<ValueType, 2>(csr);
    csr->format = SPC5_MATRIX_TYPE::FORMAT_2rV2c_WT;
}

template <class ValueType>
inline SPC5Mat<ValueType> COO_to_SPC5_2rV2c_wt(const int nbRows, const int nbCols,
                                    Ijv<ValueType> values[], int nbValues){
    SPC5Mat<ValueType> mat = COO_unsorted_to_CSR(nbRows, nbCols, values, nbValues);
    CSR_to_SPC5_2rV2c_wt(&mat);
    return mat;
}

template <class ValueType>
inline void SPC5_2rV2c_wt_Spmv_scalar(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[]){
    assert(mat->format == SPC5_MATRIX_TYPE::FORMAT_2rV2c_WT);
    core_SPC5_rV2c_Spmv_scalar<ValueType,2>(mat, x, y);
}

template <class ValueType, class FuncType>
inline void SPC5_2rV2c_wt_iterate(const SPC5Mat<ValueType>& mat, FuncType&& func){
    assert(mat->format == SPC5_MATRIX_TYPE::FORMAT_2rV2c_WT);
    core_SPC5_rV2c_iterate<ValueType,2>(mat, std::forward<FuncType>(func));
}

template <class ValueType>
inline int SPC5_2rV2c_wt_block_count(const SPC5Mat<ValueType>& csr){
    return core_SPC5_block_count<ValueType,2,ValPerVec<ValueType>::size/2>(csr);
}



#ifdef _OPENMP

template <class ValueType>
inline std::vector<ThreadInterval<ValueType>> SPC5_2rV2c_wt_split_omp(const SPC5Mat<ValueType>& mat, const int numThreads){
    return core_SPC5_rV2c_threadsplit<ValueType,2>(mat, numThreads);
}

template <class ValueType>
inline std::vector<ThreadInterval<ValueType>> SPC5_2rV2c_wt_split_omp(const SPC5Mat<ValueType>& mat){
    return core_SPC5_rV2c_threadsplit<ValueType,2>(mat, omp_get_max_threads());
}



#endif


//////////////////////////////////////////////////////////////////////////////
/// 2rV2c => 2 rows, VEC/2 columns
//////////////////////////////////////////////////////////////////////////////

template <class ValueType>
inline void CSR_to_SPC5_2rV2c(SPC5Mat<ValueType>* csr){
    core_CSR_to_SPC5_rV2c<ValueType, 2>(csr);
    csr->format = SPC5_MATRIX_TYPE::FORMAT_2rV2c;
}

template <class ValueType>
inline SPC5Mat<ValueType> COO_to_SPC5_2rV2c(const int nbRows, const int nbCols,
                                    Ijv<ValueType> values[], int nbValues){
    SPC5Mat<ValueType> mat = COO_unsorted_to_CSR(nbRows, nbCols, values, nbValues);
    CSR_to_SPC5_2rV2c(&mat);
    return mat;
}

template <class ValueType>
inline void SPC5_2rV2c_Spmv_scalar(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[]){
    assert(mat->format == SPC5_MATRIX_TYPE::FORMAT_2rV2c);
    core_SPC5_rV2c_Spmv_scalar<ValueType,2>(mat, x, y);
}

template <class ValueType, class FuncType>
inline void SPC5_2rV2c_iterate(const SPC5Mat<ValueType>& mat, FuncType&& func){
    assert(mat->format == SPC5_MATRIX_TYPE::FORMAT_2rV2c);
    core_SPC5_rV2c_iterate<ValueType,2>(mat, std::forward<FuncType>(func));
}

template <class ValueType>
inline int SPC5_2rV2c_block_count(const SPC5Mat<ValueType>& csr){
    return core_SPC5_block_count<ValueType,2,ValPerVec<ValueType>::size/2>(csr);
}


#ifdef _OPENMP

template <class ValueType>
inline std::vector<ThreadInterval<ValueType>> SPC5_2rV2c_split_omp(const SPC5Mat<ValueType>& mat, const int numThreads){
    return core_SPC5_rV2c_threadsplit<ValueType,2>(mat, numThreads);
}

template <class ValueType>
inline std::vector<ThreadInterval<ValueType>> SPC5_2rV2c_split_omp(const SPC5Mat<ValueType>& mat){
    return core_SPC5_rV2c_threadsplit<ValueType,2>(mat, omp_get_max_threads());
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


extern "C" void core_SPC5_2rVc_Spmv_double_v2(const long int nbRows, const int* rowsSizes,
                                           const unsigned char* blocksColumnIndexesWithMasks,
                                           const double* values,
                                           const double* x, double* y,
                                              const int* rowptr);
extern "C" void core_SPC5_2rVc_Spmv_float_v2(const long int nbRows, const int* rowsSizes,
                                          const unsigned char* blocksColumnIndexesWithMasks,
                                          const float* values,
                                          const float* x, float* y,
                                             const int* rowptr);

template <class ValueType>
inline void SPC5_2rVc_Spmv_v2(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[]);

template <>
inline void SPC5_2rVc_Spmv_v2<double>(const SPC5Mat<double>& mat, const double x[], double y[]){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_2rVc);
    core_SPC5_2rVc_Spmv_double_v2(mat.numberOfRows, mat.rowsSize.get(),
                               mat.blocksColumnIndexesWithMasks.get(), mat.values.get(),
                               x, y, mat.rowsSizeCpy.get());
}

template <>
inline void SPC5_2rVc_Spmv_v2<float>(const SPC5Mat<float>& mat, const float x[], float y[]){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_2rVc);
    core_SPC5_2rVc_Spmv_float_v2(mat.numberOfRows, mat.rowsSize.get(),
                              mat.blocksColumnIndexesWithMasks.get(), mat.values.get(),
                              x, y, mat.rowsSizeCpy.get());
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
#ifdef SPLIT_NUMA
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
#else
template <>
inline void SPC5_2rVc_Spmv_omp<double>(const SPC5Mat<double>& mat, const double x[], double y[],
                                   const std::vector<ThreadInterval<double>>& threadsVecs){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_2rVc);
    const int numThreads = int(threadsVecs.size());
#pragma omp parallel num_threads(numThreads)
    {
        memset(threadsVecs[omp_get_thread_num()].threadY.get(), 0, sizeof(double)*threadsVecs[omp_get_thread_num()].numberOfRows);

        core_SPC5_2rVc_Spmv_double(threadsVecs[omp_get_thread_num()].numberOfRows,
                                       mat.rowsSize.get()+threadsVecs[omp_get_thread_num()].startingRow,
                                       mat.blocksColumnIndexesWithMasks.get()+mat.rowsSize[threadsVecs[omp_get_thread_num()].startingRow]*6,
                                       mat.values.get()+threadsVecs[omp_get_thread_num()].valuesOffset,
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
                                       mat.rowsSize.get()+threadsVecs[omp_get_thread_num()].startingRow,
                                       mat.blocksColumnIndexesWithMasks.get()+mat.rowsSize[threadsVecs[omp_get_thread_num()].startingRow]*8,
                                       mat.values.get()+threadsVecs[omp_get_thread_num()].valuesOffset,
                                       x, threadsVecs[omp_get_thread_num()].threadY.get());

        SPC5_opti_merge(&y[threadsVecs[omp_get_thread_num()].startingRow], threadsVecs[omp_get_thread_num()].threadY.get(),
                threadsVecs[omp_get_thread_num()].numberOfRows);
    }
}
#endif

#endif

//////////////////////////////////////////////////////////////////////////////
/// 4rV2c => 4 rows, VEC/2 columns
//////////////////////////////////////////////////////////////////////////////


template <class ValueType>
inline void CSR_to_SPC5_4rV2c(SPC5Mat<ValueType>* csr){
    core_CSR_to_SPC5_rV2c<ValueType, 4>(csr);
    csr->format = SPC5_MATRIX_TYPE::FORMAT_4rV2c;
}

template <class ValueType>
inline SPC5Mat<ValueType> COO_to_SPC5_4rV2c(const int nbRows, const int nbCols,
                                    Ijv<ValueType> values[], int nbValues){
    SPC5Mat<ValueType> mat = COO_unsorted_to_CSR(nbRows, nbCols, values, nbValues);
    CSR_to_SPC5_4rV2c(&mat);
    return mat;
}

template <class ValueType>
inline void SPC5_4rV2c_Spmv_scalar(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[]){
    assert(mat->format == SPC5_MATRIX_TYPE::FORMAT_4rV2c);
    core_SPC5_rV2c_Spmv_scalar<ValueType,4>(mat, x, y);
}

template <class ValueType, class FuncType>
inline void SPC5_4rV2c_iterate(const SPC5Mat<ValueType>& mat, FuncType&& func){
    assert(mat->format == SPC5_MATRIX_TYPE::FORMAT_4rV2c);
    core_SPC5_rV2c_iterate<ValueType,4>(mat, std::forward<FuncType>(func));
}

template <class ValueType>
inline int SPC5_4rV2c_block_count(const SPC5Mat<ValueType>& csr){
    return core_SPC5_block_count<ValueType,4,ValPerVec<ValueType>::size/2>(csr);
}


#ifdef _OPENMP

template <class ValueType>
inline std::vector<ThreadInterval<ValueType>> SPC5_4rV2c_split_omp(const SPC5Mat<ValueType>& mat, const int numThreads){
    return core_SPC5_rV2c_threadsplit<ValueType,4>(mat, numThreads);
}

template <class ValueType>
inline std::vector<ThreadInterval<ValueType>> SPC5_4rV2c_split_omp(const SPC5Mat<ValueType>& mat){
    return core_SPC5_rV2c_threadsplit<ValueType,4>(mat, omp_get_max_threads());
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

extern "C" void core_SPC5_4rVc_Spmv_double_v2(const long int nbRows, const int* rowsSizes,
                                           const unsigned char* blocksColumnIndexesWithMasks,
                                           const double* values,
                                           const double* x, double* y,
                                              const int* rowptr);
extern "C" void core_SPC5_4rVc_Spmv_float_v2(const long int nbRows, const int* rowsSizes,
                                          const unsigned char* blocksColumnIndexesWithMasks,
                                          const float* values,
                                          const float* x, float* y,
                                             const int* rowptr);

template <class ValueType>
inline void SPC5_4rVc_Spmv_v2(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[]);

template <>
inline void SPC5_4rVc_Spmv_v2<double>(const SPC5Mat<double>& mat, const double x[], double y[]){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_4rVc);
    core_SPC5_4rVc_Spmv_double_v2(mat.numberOfRows, mat.rowsSize.get(),
                               mat.blocksColumnIndexesWithMasks.get(), mat.values.get(),
                                  x, y, mat.rowsSizeCpy.get());
}

template <>
inline void SPC5_4rVc_Spmv_v2<float>(const SPC5Mat<float>& mat, const float x[], float y[]){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_4rVc);
    core_SPC5_4rVc_Spmv_float_v2(mat.numberOfRows, mat.rowsSize.get(),
                              mat.blocksColumnIndexesWithMasks.get(), mat.values.get(),
                              x, y, mat.rowsSizeCpy.get());
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
#ifdef SPLIT_NUMA
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
#else
template <>
inline void SPC5_4rVc_Spmv_omp<double>(const SPC5Mat<double>& mat, const double x[], double y[],
                                   const std::vector<ThreadInterval<double>>& threadsVecs){
    assert(mat.format == SPC5_MATRIX_TYPE::FORMAT_4rVc);
    const int numThreads = int(threadsVecs.size());
#pragma omp parallel num_threads(numThreads)
    {
        memset(threadsVecs[omp_get_thread_num()].threadY.get(), 0, sizeof(double)*threadsVecs[omp_get_thread_num()].numberOfRows);

        core_SPC5_4rVc_Spmv_double(threadsVecs[omp_get_thread_num()].numberOfRows,
                                       mat.rowsSize.get()+threadsVecs[omp_get_thread_num()].startingRow,
                                       mat.blocksColumnIndexesWithMasks.get()+mat.rowsSize[threadsVecs[omp_get_thread_num()].startingRow]*8,
                                       mat.values.get()+threadsVecs[omp_get_thread_num()].valuesOffset,
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
                                       mat.rowsSize.get()+threadsVecs[omp_get_thread_num()].startingRow,
                                       mat.blocksColumnIndexesWithMasks.get()+mat.rowsSize[threadsVecs[omp_get_thread_num()].startingRow]*12,
                                       mat.values.get()+threadsVecs[omp_get_thread_num()].valuesOffset,
                                       x, threadsVecs[omp_get_thread_num()].threadY.get());

        SPC5_opti_merge(&y[threadsVecs[omp_get_thread_num()].startingRow], threadsVecs[omp_get_thread_num()].threadY.get(),
                threadsVecs[omp_get_thread_num()].numberOfRows);
    }
}
#endif

#endif


//////////////////////////////////////////////////////////////////////////////
/// 8rV2c => 8 rows, VEC/2 columns
//////////////////////////////////////////////////////////////////////////////

template <class ValueType>
inline void CSR_to_SPC5_8rV2c(SPC5Mat<ValueType>* csr){
    core_CSR_to_SPC5_rV2c<ValueType, 8>(csr);
    csr->format = SPC5_MATRIX_TYPE::FORMAT_8rV2c;
}

template <class ValueType>
inline SPC5Mat<ValueType> COO_to_SPC5_8rV2c(const int nbRows, const int nbCols,
                                    Ijv<ValueType> values[], int nbValues){
    SPC5Mat<ValueType> mat = COO_unsorted_to_CSR(nbRows, nbCols, values, nbValues);
    CSR_to_SPC5_8rV2c(&mat);
    return mat;
}

template <class ValueType>
inline void SPC5_8rV2c_Spmv_scalar(const SPC5Mat<ValueType>& mat, const ValueType x[], ValueType y[]){
    assert(mat->format == SPC5_MATRIX_TYPE::FORMAT_8rV2c);
    core_SPC5_rV2c_Spmv_scalar<ValueType,8>(mat, x, y);
}

template <class ValueType, class FuncType>
inline void SPC5_8rV2c_iterate(const SPC5Mat<ValueType>& mat, FuncType&& func){
    assert(mat->format == SPC5_MATRIX_TYPE::FORMAT_8rV2c);
    core_SPC5_rV2c_iterate<ValueType,8>(mat, std::forward<FuncType>(func));
}

template <class ValueType>
inline int SPC5_8rV2c_block_count(const SPC5Mat<ValueType>& csr){
    return core_SPC5_block_count<ValueType,8,ValPerVec<ValueType>::size/2>(csr);
}


#ifdef _OPENMP

template <class ValueType>
inline std::vector<ThreadInterval<ValueType>> SPC5_8rV2c_split_omp(const SPC5Mat<ValueType>& mat, const int numThreads){
    return core_SPC5_rV2c_threadsplit<ValueType,8>(mat, numThreads);
}

template <class ValueType>
inline std::vector<ThreadInterval<ValueType>> SPC5_8rV2c_split_omp(const SPC5Mat<ValueType>& mat){
    return core_SPC5_rV2c_threadsplit<ValueType,8>(mat, omp_get_max_threads());
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
    case SPC5_MATRIX_TYPE::FORMAT_2rV2c_WT :
        {
        CSR_to_SPC5_2rV2c_wt<ValueType>(mat);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_2rV2c :
        {
        CSR_to_SPC5_2rV2c<ValueType>(mat);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_2rVc :
        {
        CSR_to_SPC5_2rVc<ValueType>(mat);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_4rV2c :
        {
        CSR_to_SPC5_4rV2c<ValueType>(mat);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_4rVc :
        {
        CSR_to_SPC5_4rVc<ValueType>(mat);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_8rV2c :
        {
        CSR_to_SPC5_8rV2c<ValueType>(mat);
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
    case SPC5_MATRIX_TYPE::FORMAT_2rV2c_WT :
        {
        SPC5_2rV2c_wt_Spmv_scalar<ValueType>(mat, x, y);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_2rV2c :
        {
        SPC5_2rV2c_Spmv_scalar<ValueType>(mat, x, y);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_2rVc :
        {
        SPC5_2rVc_Spmv_scalar<ValueType>(mat, x, y);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_4rV2c :
        {
        SPC5_4rV2c_Spmv_scalar<ValueType>(mat, x, y);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_4rVc :
        {
        SPC5_4rVc_Spmv_scalar<ValueType>(mat, x, y);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_8rV2c :
        {
        SPC5_8rV2c_Spmv_scalar<ValueType>(mat, x, y);
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
    case SPC5_MATRIX_TYPE::FORMAT_2rV2c_WT :
        {
        SPC5_2rV2c_wt_iterate<ValueType>(mat, std::forward<FuncType>(func));
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_2rV2c :
        {
        SPC5_2rV2c_iterate<ValueType>(mat, std::forward<FuncType>(func));
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_2rVc :
        {
        SPC5_2rVc_iterate<ValueType>(mat, std::forward<FuncType>(func));
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_4rV2c :
        {
        SPC5_4rV2c_iterate<ValueType>(mat, std::forward<FuncType>(func));
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_4rVc :
        {
        SPC5_4rVc_iterate<ValueType>(mat, std::forward<FuncType>(func));
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_8rV2c :
        {
        SPC5_8rV2c_iterate<ValueType>(mat, std::forward<FuncType>(func));
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
    case SPC5_MATRIX_TYPE::FORMAT_2rV2c_WT :
        {
        SPC5_2rV2c_wt_block_count<ValueType>(csr);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_2rV2c :
        {
        SPC5_2rV2c_block_count<ValueType>(csr);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_2rVc :
        {
        SPC5_2rVc_block_count<ValueType>(csr);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_4rV2c :
        {
        SPC5_4rV2c_block_count<ValueType>(csr);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_4rVc :
        {
        SPC5_4rVc_block_count<ValueType>(csr);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_8rV2c :
        {
        SPC5_8rV2c_block_count<ValueType>(csr);
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
            const double coef[4] = { 8.266286e-03, -1.626724e-01 , 1.293833e+00 , -5.437822e-01  };
            const int nbBlocks = SPC5_2rV2c_block_count<ValueType>(csr);
            const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
            if(thisSpeed > estimatedSpeed){
                estimatedSpeed = thisSpeed;
                bestType = SPC5_MATRIX_TYPE::FORMAT_2rV2c_WT;
            }
        }
        else{
            const double coef[4] = { -9.195940e-04, 4.442050e-03 , 5.273045e-01 , 1.101610e+00  };
            const int nbBlocks = SPC5_2rV2c_block_count<ValueType>(csr);
            const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
            if(thisSpeed > estimatedSpeed){
                estimatedSpeed = thisSpeed;
                bestType = SPC5_MATRIX_TYPE::FORMAT_2rV2c_WT;
            }
        }
    }
    {
        if(in_double){
            const double coef[4] = { 7.787745e-03, -1.570643e-01 , 1.231532e+00 , -1.800484e-01  };
            const int nbBlocks = SPC5_2rV2c_block_count<ValueType>(csr);
            const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
            if(thisSpeed > estimatedSpeed){
                estimatedSpeed = thisSpeed;
                bestType = SPC5_MATRIX_TYPE::FORMAT_2rV2c;
            }
        }
        else{
            const double coef[4] = { -9.195940e-04, 4.442050e-03 , 5.273045e-01 , 1.101610e+00  };
            const int nbBlocks = SPC5_2rV2c_block_count<ValueType>(csr);
            const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
            if(thisSpeed > estimatedSpeed){
                estimatedSpeed = thisSpeed;
                bestType = SPC5_MATRIX_TYPE::FORMAT_2rV2c;
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
            const double coef[4] = { 5.138277e-04, -3.127338e-02 , 5.740454e-01 , 2.086101e-01 };
            const int nbBlocks = SPC5_4rV2c_block_count<ValueType>(csr);
            const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
            if(thisSpeed > estimatedSpeed){
                estimatedSpeed = thisSpeed;
                bestType = SPC5_MATRIX_TYPE::FORMAT_4rV2c;
            }
        }
        else{
            const double coef[4] = {  -2.181318e-05, -5.198402e-03 , 3.855267e-01 , 8.700233e-01 };
            const int nbBlocks = SPC5_4rV2c_block_count<ValueType>(csr);
            const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
            if(thisSpeed > estimatedSpeed){
                estimatedSpeed = thisSpeed;
                bestType = SPC5_MATRIX_TYPE::FORMAT_4rV2c;
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
    {
        if(in_double){
            const double coef[4] = { 3.403517e-05, -7.292415e-03 , 2.941192e-01 , 2.756599e-01 };
            const int nbBlocks = SPC5_8rV2c_block_count<ValueType>(csr);
            const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
            if(thisSpeed > estimatedSpeed){
                estimatedSpeed = thisSpeed;
                bestType = SPC5_MATRIX_TYPE::FORMAT_8rV2c;
            }
        }
        else{
            const double coef[4] = { -2.097467e-05, -4.264717e-04 , 2.086814e-01 , 6.871666e-01 };
            const int nbBlocks = SPC5_8rV2c_block_count<ValueType>(csr);
            const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
            if(thisSpeed > estimatedSpeed){
                estimatedSpeed = thisSpeed;
                bestType = SPC5_MATRIX_TYPE::FORMAT_8rV2c;
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
    case SPC5_MATRIX_TYPE::FORMAT_2rV2c_WT :
    {
        return "2rV2c_WT";
    }
    case SPC5_MATRIX_TYPE::FORMAT_2rV2c :
    {
        return "2rV2c";
    }
    case SPC5_MATRIX_TYPE::FORMAT_2rVc :
    {
        return "2rVc";
    }
    case SPC5_MATRIX_TYPE::FORMAT_4rV2c :
    {
        return "4rV2c";
    }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_4rVc :
    {
        return "4rVc";
    }
    case SPC5_MATRIX_TYPE::FORMAT_8rV2c :
    {
        return "8rV2c";
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
        const double coef[4] = { 2.0839,0.0600,2.3371,0.0778};
        const int nbBlocks = SPC5_2rV2c_block_count<ValueType>(csr);
        const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
        if(thisSpeed > estimatedSpeed){
            estimatedSpeed = thisSpeed;
            bestType = SPC5_MATRIX_TYPE::FORMAT_2rV2c_WT;
        }
    }
    {
        const double coef[4] = {  3.5188,0.1213,2.1070,0.0738};
        const int nbBlocks = SPC5_2rV2c_block_count<ValueType>(csr);
        const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
        if(thisSpeed > estimatedSpeed){
            estimatedSpeed = thisSpeed;
            bestType = SPC5_MATRIX_TYPE::FORMAT_2rV2c;
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
        const double coef[4] = { 4.7097,0.1107,1.1185,0.0439 };
        const int nbBlocks = SPC5_4rV2c_block_count<ValueType>(csr);
        const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
        if(thisSpeed > estimatedSpeed){
            estimatedSpeed = thisSpeed;
            bestType = SPC5_MATRIX_TYPE::FORMAT_4rV2c;
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
    {
        const double coef[4] = {4.7149,0.1311,0.6598,0.0223};
        const int nbBlocks = SPC5_8rV2c_block_count<ValueType>(csr);
        const double thisSpeed = polyval(coef, double(csr.numberOfNNZ)/double(nbBlocks));
        if(thisSpeed > estimatedSpeed){
            estimatedSpeed = thisSpeed;
            bestType = SPC5_MATRIX_TYPE::FORMAT_8rV2c;
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
    case SPC5_MATRIX_TYPE::FORMAT_2rV2c_WT :
        {
        return SPC5_2rV2c_wt_split_omp<ValueType>(mat);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_2rVc :
        {
        return SPC5_2rVc_split_omp<ValueType>(mat);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_2rV2c :
        {
        return SPC5_2rV2c_split_omp<ValueType>(mat);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_4rV2c :
        {
        return SPC5_4rV2c_split_omp<ValueType>(mat);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_4rVc :
        {
        return SPC5_4rVc_split_omp<ValueType>(mat);
        }
        break;
    case SPC5_MATRIX_TYPE::FORMAT_8rV2c :
        {
        return SPC5_8rV2c_split_omp<ValueType>(mat);
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
    default :
        {
            throw std::invalid_argument("SPC5_Spmv : Unknown format type");
        }
    }
}
#endif


#endif
