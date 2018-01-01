#include "spc5.hpp"

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

//////////////////////////////////////////////////////////////////////////////
/// Timer class
//////////////////////////////////////////////////////////////////////////////

#include <chrono>   // just for timer

class dtimer {
    using double_second_time = std::chrono::duration<double, std::ratio<1, 1>>;

    std::chrono::high_resolution_clock::time_point
    m_start;  ///< m_start time (start)
    std::chrono::high_resolution_clock::time_point m_end;  ///< stop time (stop)
    std::chrono::nanoseconds m_cumulate;  ///< the m_cumulate time

public:
    /// Constructor
    dtimer() : m_cumulate(std::chrono::nanoseconds()) { start(); }

    /// Copy constructor
    dtimer(const dtimer& other) = delete;
    /// Copies an other timer
    dtimer& operator=(const dtimer& other) = delete;
    /// Move constructor
    dtimer(dtimer&& other) = delete;
    /// Copies an other timer
    dtimer& operator=(dtimer&& other) = delete;

    /** Rest all the values, and apply start */
    void reset() {
        m_start = std::chrono::high_resolution_clock::time_point();
        m_end = std::chrono::high_resolution_clock::time_point();
        m_cumulate = std::chrono::nanoseconds();
        start();
    }

    /** Start the timer */
    void start() {
        m_start = std::chrono::high_resolution_clock::now();
    }

    /** Stop the current timer */
    void stop() {
        m_end = std::chrono::high_resolution_clock::now();
        m_cumulate += std::chrono::duration_cast<std::chrono::nanoseconds>(m_end - m_start);
    }

    /** Return the elapsed time between start and stop (in second) */
    double getElapsed() const {
        return std::chrono::duration_cast<double_second_time>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(m_end - m_start)).count();
    }

    /** Return the total counted time */
    double getCumulated() const {
        return std::chrono::duration_cast<double_second_time>(m_cumulate).count();
    }

    /** End the current counter (stop) and return the elapsed time */
    double stopAndGetElapsed() {
        stop();
        return getElapsed();
    }
};


//////////////////////////////////////////////////////////////////////////////
/// MKL CSR
//////////////////////////////////////////////////////////////////////////////

#ifdef USE_MKL

#include <mkl.h>

template <class ValueClass, typename DoubleFoncClass, typename SingleFoncClass, typename ... Args>
inline typename std::enable_if<std::is_same<ValueClass, float>::value, void>::type
CallDoubleOrSingle(DoubleFoncClass /*doubleFonc*/, SingleFoncClass singleFonc, Args ... args){
    singleFonc(args...);
}
template <class ValueClass, typename DoubleFoncClass, typename SingleFoncClass, typename ... Args>
inline typename std::enable_if<std::is_same<ValueClass, double>::value, void>::type
CallDoubleOrSingle(DoubleFoncClass doubleFonc, SingleFoncClass /*singleFonc*/, Args ... args){
    doubleFonc(args...);
}


template <class ValueType>
struct CsrMKL{
    MKL_INT m_row;  //< the dim of the matrix
    MKL_INT m_col;  //< the dim of the matrix
    MKL_INT nnz;//< the number of nnz (== ia[m])
    ValueType *a;  //< the values (of size NNZ)
    MKL_INT *ia;//< the usual rowptr (of size m+1)
    MKL_INT *ja;//< the colidx of each NNZ (of size nnz)

    CsrMKL(){
        m_row = 0;
        m_col = 0;
        nnz = 0;
        a = NULL;
        ia = NULL;
        ja= NULL;
    }

    ~CsrMKL(){
        delete[] a;
        delete[] ia;
        delete[] ja;
    }
};

/** See https://software.intel.com/fr-fr/node/520849#449CA855-CE5B-4061-B003-70D078CA5E05 */
template <class ValueType>
CsrMKL<ValueType> COO_to_CsrMKL(const int nbRows, const int nbCols,
                                const Ijv<ValueType> values[], int nbValues){
    CsrMKL<ValueType> csr;

    MKL_INT job[6] = {1,//if job(1)=1, the matrix in the coordinate format is converted to the CRS format.
                      0,//If job(2)=0, zero-based indexing for the matrix in CRS format is used;
                      0,//If job(3)=0, zero-based indexing for the matrix in coordinate format is used;
                      0,
                      nbValues,//job(5)=nnz - sets number of the non-zero elements of the matrix A if job(1)=1.
                      0 //If job(6)=0, all arrays acsr, ja, ia are filled in for the output storage.
                     };
    // Init crs
    csr.m_row = nbRows;
    csr.m_col = nbCols;
    csr.nnz = nbValues;
    csr.a = new ValueType[csr.nnz];
    csr.ia = new MKL_INT[csr.m_row+1];
    csr.ja = new MKL_INT[csr.nnz];
    MKL_INT nnz = nbValues;
    MKL_INT info;

    std::unique_ptr<MKL_INT[]> rowind(new MKL_INT[nbValues]);
    std::unique_ptr<MKL_INT[]> colind(new MKL_INT[nbValues]);
    std::unique_ptr<ValueType[]> val(new ValueType[nbValues]);

    for(int idxVal = 0 ; idxVal < nbValues ; ++idxVal){
        rowind[idxVal] = values[idxVal].i;
        colind[idxVal] = values[idxVal].j;
        val[idxVal] = values[idxVal].v;
    }

    CallDoubleOrSingle<ValueType>(mkl_dcsrcoo, mkl_scsrcoo,job , &csr.m_row,
                csr.a , csr.ja , csr.ia , &nnz ,
                val.get(), rowind.get(), colind.get(), &info );

    return csr;
}

/** See https://software.intel.com/fr-fr/node/520815#D840F0E5-E41A-4E91-94D2-FEB320F93E91 */

template <class ValueType>
void compute_CsrMKL( CsrMKL<ValueType>& crs, ValueType *x , ValueType *y){
    char transa = 'N';
    // void mkl_cspblas_dcsrgemv (const char *transa , const MKL_INT *m , const ValueType *a , const MKL_INT *ia , const MKL_INT *ja , const ValueType *x , ValueType *y );
    CallDoubleOrSingle<ValueType>(mkl_cspblas_dcsrgemv, mkl_cspblas_scsrgemv, &transa, &crs.m_row , crs.a , crs.ia , crs.ja , x, y);
}


#endif


template <class ValueType>
ValueType ChechAccuracy(const ValueType y1[], const ValueType y2[], const int size){
    ValueType maxDiff = 0;
    for(int idx = 0 ; idx < size ; ++idx){
        if(y1[idx] == 0.0 || y2[idx] == 0.0){
            maxDiff = std::max(maxDiff, std::abs(y1[idx] + y2[idx]));
        }
        else{
            maxDiff = std::max(maxDiff, std::abs((y1[idx]-y2[idx])/y1[idx]));
        }
    }
    return maxDiff;
}


//////////////////////////////////////////////////////////////////////////////
/// Load MM matrix
//////////////////////////////////////////////////////////////////////////////

#include <memory>
#include <string>
#include <cstdio>
#include <cstring>

template <class ValueType>
void loadMM(const std::string& filename,
            std::unique_ptr<Ijv<ValueType>[]>* values,
            int* nbRows, int* nbCols, int* nbValues){
    FILE* fmm = fopen(filename.c_str(),"r");

    if(fmm == nullptr){
        fprintf(stderr, "[ERROR] Cannot open %s\n", filename.c_str());
        throw;
    }

    // Use large buffer please
    const size_t bufferSize = 1024*1024*30; // ~30MB
    std::unique_ptr<char[]> buffer(new char[bufferSize]);
    const int buffSetRet = setvbuf(fmm, buffer.get(), _IOFBF, bufferSize);

    if(buffSetRet != 0){
        fprintf(stderr, "[ERROR] buffer cannot be set\n");
        throw;
    }

    // read %%MatrixMarket matrix coordinate real general
    char matrix[512];
    char coordinate[512];
    char real[512];
    char general[512];

    char* line = (char*)::malloc(1024);
    size_t lineLength = 1024;

    long int retval = getline(&line, &lineLength, fmm);
    if(retval == -1){
        fprintf(stderr,"[ERROR] Failled to read header\n");
        throw;
    }
    sscanf(line, "%%%%MatrixMarket %s %s %s %s", matrix, coordinate, real, general);

    if(strcmp(real,"real") != 0 && strcmp(real,"pattern") != 0 && strcmp(real,"integer") != 0){
        fprintf(stderr,"[ERROR] Error only real matrices are supported (%s given)\n", real);
        throw;
    }

    const bool is_symmetric = (strcmp(general,"symmetric") == 0);
    const bool is_skew_symmetric = (strcmp(general,"skew-symmetric") == 0);
    const bool is_pattern = (strcmp(real,"pattern") == 0);
    const bool is_integer = (strcmp(real,"integer") == 0);

    while((retval = getline(&line, &lineLength, fmm)) != -1
          && line[0] == '%'){
    }

    if(retval == -1){
        fprintf(stderr,"[ERROR] Failled to read message header\n");
        throw;
    }

    int dim1, dim2, nbValuesInFile;
    sscanf(line, "%d %d %d", &dim1, &dim2, &nbValuesInFile);

    int nbValuesInMatrix = nbValuesInFile;
    if(is_symmetric || is_skew_symmetric){
        nbValuesInMatrix += nbValuesInFile;
    }
    (*values).reset(new Ijv<ValueType>[nbValuesInMatrix]);

    double val = 1.;

    long int progressStep = -1;

    int currentValueIdx = 0;

    for(int idx = 0 ; idx < nbValuesInFile ; ++idx){
        // Show progress
        if((((idx+1)*100L)/nbValuesInFile) != progressStep){
           progressStep = (((idx+1)*100L)/nbValuesInFile);
           printf("\r%12d/%d : %3ld%% [", idx, nbValuesInFile, progressStep);
           for(long int idxBar = 0 ; idxBar < (progressStep/5)-1 ; ++idxBar){
               printf("=");
           }
           printf(">");
           for(long int idxBar = (progressStep/5) ; idxBar < 20 ; ++idxBar){
               printf(" ");
           }
           printf("]");
           fflush(stdout);
        }

        retval = getline(&line, &lineLength, fmm);
        if(retval == -1){
            fprintf(stderr,"[ERROR] Failled to read value %d\n", idx);
            throw;
        }
        int row, col;
        if(is_pattern){
            sscanf(line, "%d %d", &row, &col);
        }
        else if(is_integer){
            int tmp;
            sscanf(line, "%d %d %d", &row, &col, &tmp);
            val = tmp;
        }
        else{
            sscanf(line, "%d %d %lf", &row, &col, &val);
        }
        Ijv<ValueType> value;
        value.i = row-1;
        value.j = col-1;
        value.v = val;

        assert(0 <= value.j);
        assert(value.j < dim2);
        assert(0 <= value.i);
        assert(value.i < dim1);

        (*values)[currentValueIdx] = value;
        currentValueIdx += 1;

        if(is_symmetric && value.i != value.j){
            std::swap(value.i, value.j);
            (*values)[currentValueIdx] = value;
            currentValueIdx += 1;
        }
        if(is_skew_symmetric && value.i != value.j){
            std::swap(value.i, value.j);
            value.v = -value.v;
            (*values)[currentValueIdx] = value;
            currentValueIdx += 1;
        }
    }
    printf("\n"); // After the progress bar

    // Close the file
    fclose(fmm);
    ::free(line);

    std::sort(&(*values)[0], &(*values)[currentValueIdx], [](const Ijv<ValueType>& v1, const Ijv<ValueType>& v2){
        return v1.i < v2.i || (v1.i == v2.i && v1.j < v2.j);
    });

    (*nbRows) = dim1;
    (*nbCols) = dim2;
    (*nbValues) = currentValueIdx;
}


template <class ValueType>
void builddense(const int dim,
            std::unique_ptr<Ijv<ValueType>[]>* values,
            int* nbRows, int* nbCols, int* nbValues){
    *nbRows = dim;
    *nbCols = dim;
    *nbValues = dim*dim;
    values->reset(new Ijv<ValueType>[dim*dim]);

    for(int idxRow = 0 ; idxRow < dim; ++idxRow){
        for(int idxCol = 0 ; idxCol < dim; ++idxCol){
            Ijv<ValueType> value;
            value.i = idxRow;
            value.j = idxCol;
            value.v = 1;
            (*values)[idxRow*dim + idxCol] = value;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
/// Load MM matrix
//////////////////////////////////////////////////////////////////////////////

#include <iostream>

//#define USEDENSE

int main(int argc, char** argv){
    using ValueType = double;
#ifndef USEDENSE
    if(argc != 2){
        std::cerr << "[ERROR] You should pass a matrix market file as argument, exiting..." << std::endl;
        return 1;
    }

    std::cout << "Will now open " << argv[1] << std::endl;
#endif
    std::unique_ptr<Ijv<ValueType>[]> values;
    int nbRows;
    int nbCols;
    int nbValues;
#ifndef USEDENSE
    loadMM<ValueType>(argv[1], &values, &nbRows, &nbCols, &nbValues);
#else
    builddense<ValueType>(8000, &values, &nbRows, &nbCols, &nbValues);
#endif

    std::cout << "-> number of rows = " << nbRows << std::endl;
    std::cout << "-> number of columns = " << nbCols << std::endl;
    std::cout << "-> number of values = " << nbValues << std::endl;
    std::cout << "-> number of values per row = " << double(nbValues)/double(nbRows) << std::endl;

    const int nbLoops = 16;
    std::cout << "-> number loops to smooth the timing if > 1 =  " << nbLoops << std::endl;

    const long int flops = nbValues*2L*nbLoops;
    std::cout << "-> number of flops to do per spmv = " << flops << std::endl;

    std::unique_ptr<ValueType[]> x(new ValueType[nbCols + SPC5_VEC_PADDING::SPC5_VEC_PADDING_X]());
    for(int idxX = 0 ; idxX < nbCols ; ++idxX){
        x[idxX] = 1.0;
    }

    std::unique_ptr<ValueType[]> ycsr(new ValueType[nbCols + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y]());
    {
        std::cout << "Start usual CSR: " << nbRows << std::endl;
        dtimer timerConversion;
        SPC5Mat<ValueType> csr = COO_sorted_to_CSR<ValueType>(nbRows, nbCols, values.get(), nbValues);
        timerConversion.stop();
        std::cout << "Conversion in : " << timerConversion.getElapsed() << "s\n";
        dtimer timer;

        for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
            CSR_Spmv_scalar<ValueType>(csr, x.get(), ycsr.get());
        }

        timer.stop();
        std::cout << "-> Done in " << timer.getElapsed() << "s\n";
        std::cout << "-> GFlops " << double(flops)/timer.getElapsed()/1e9 << "s\n";

        std::pair<SPC5_MATRIX_TYPE, double> esimation = SPC5_find_best<ValueType>(csr);
        std::cout << "-> Esimated performance are " << esimation.second << " for " << SPC5_type_to_string(esimation.first) << "\n";
    }
    std::unique_ptr<ValueType[]> y(new ValueType[nbCols + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y]());

#ifdef USE_MKL
    {
        memset(y.get(), 0, sizeof(ValueType)*(nbCols + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y));

        std::cout << "Start MKL CSR: "<< std::endl;
        dtimer timerConversion;
        CsrMKL<ValueType> csrMkl = COO_to_CsrMKL<ValueType>(nbRows, nbCols, values.get(), nbValues);
        timerConversion.stop();
        std::cout << "Conversion in : " << timerConversion.getElapsed() << "s\n";
        dtimer timer;

        for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
            compute_CsrMKL<ValueType>(csrMkl, x.get(), y.get());
        }

        timer.stop();
        std::cout << "-> Done in " << timer.getElapsed() << "s\n";
        std::cout << "-> GFlops " << double(flops)/timer.getElapsed()/1e9 << "s\n";

        // Our kernels accumulate the results
        for(int idxX = 0 ; idxX < nbCols ; ++idxX){
            y[idxX] *= nbLoops;
        }

        std::cout << "-> Max Difference in Accuracy " << ChechAccuracy(ycsr.get(), y.get(), nbCols) << "\n";
    }
#endif
    {
        memset(y.get(), 0, sizeof(ValueType)*(nbCols + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y));

        std::cout << "Start usual 1rVc: "<< std::endl;
        dtimer timerConversion;
        SPC5Mat<ValueType> csr = COO_to_SPC5_1rVc<ValueType>(nbRows, nbCols, values.get(), nbValues);
        timerConversion.stop();
        std::cout << "Conversion in : " << timerConversion.getElapsed() << "s\n";
        dtimer timer;

        for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
            SPC5_1rVc_Spmv<ValueType>(csr, x.get(), y.get());
        }

        timer.stop();
        std::cout << "-> Done in " << timer.getElapsed() << "s\n";
        std::cout << "-> Number of blocks " << csr.numberOfBlocks << "( avg. " << double(csr.numberOfNNZ)/double(csr.numberOfBlocks)<< " values per block)\n";
        std::cout << "-> GFlops " << double(flops)/timer.getElapsed()/1e9 << "s\n";
        std::cout << "-> Max Difference in Accuracy " << ChechAccuracy(ycsr.get(), y.get(), nbCols) << "\n";
    }
    {
        memset(y.get(), 0, sizeof(ValueType)*(nbCols + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y));

        std::cout << "Start usual 2rV2c with test: "<< std::endl;
        dtimer timerConversion;
        SPC5Mat<ValueType> csr = COO_to_SPC5_2rV2c_wt<ValueType>(nbRows, nbCols, values.get(), nbValues);
        timerConversion.stop();
        std::cout << "Conversion in : " << timerConversion.getElapsed() << "s\n";
        dtimer timer;

        for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
            SPC5_2rV2c_wt_Spmv<ValueType>(csr, x.get(), y.get());
        }

        timer.stop();
        std::cout << "-> Done in " << timer.getElapsed() << "s\n";
        std::cout << "-> Number of blocks " << csr.numberOfBlocks << "( avg. " << double(csr.numberOfNNZ)/double(csr.numberOfBlocks)<< " values per block)\n";
        std::cout << "-> GFlops " << double(flops)/timer.getElapsed()/1e9 << "s\n";
        std::cout << "-> Max Difference in Accuracy " << ChechAccuracy(ycsr.get(), y.get(), nbCols) << "\n";
    }
    {
        memset(y.get(), 0, sizeof(ValueType)*(nbCols + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y));

        std::cout << "Start usual 2rV2c: "<< std::endl;
        dtimer timerConversion;
        SPC5Mat<ValueType> csr = COO_to_SPC5_2rV2c<ValueType>(nbRows, nbCols, values.get(), nbValues);
        timerConversion.stop();
        std::cout << "Conversion in : " << timerConversion.getElapsed() << "s\n";
        dtimer timer;

        for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
            SPC5_2rV2c_Spmv<ValueType>(csr, x.get(), y.get());
        }

        timer.stop();
        std::cout << "-> Done in " << timer.getElapsed() << "s\n";
        std::cout << "-> Number of blocks " << csr.numberOfBlocks << "( avg. " << double(csr.numberOfNNZ)/double(csr.numberOfBlocks)<< " values per block)\n";
        std::cout << "-> GFlops " << double(flops)/timer.getElapsed()/1e9 << "s\n";
        std::cout << "-> Max Difference in Accuracy " << ChechAccuracy(ycsr.get(), y.get(), nbCols) << "\n";
    }
    {
        memset(y.get(), 0, sizeof(ValueType)*(nbCols + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y));

        std::cout << "Start usual 2rVc: "<< std::endl;
        dtimer timerConversion;
        SPC5Mat<ValueType> csr = COO_to_SPC5_2rVc<ValueType>(nbRows, nbCols, values.get(), nbValues);
        timerConversion.stop();
        std::cout << "Conversion in : " << timerConversion.getElapsed() << "s\n";
        dtimer timer;

        for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
            SPC5_2rVc_Spmv<ValueType>(csr, x.get(), y.get());
        }

        timer.stop();
        std::cout << "-> Done in " << timer.getElapsed() << "s\n";
        std::cout << "-> Number of blocks " << csr.numberOfBlocks << "( avg. " << double(csr.numberOfNNZ)/double(csr.numberOfBlocks)<< " values per block)\n";
        std::cout << "-> GFlops " << double(flops)/timer.getElapsed()/1e9 << "s\n";
        std::cout << "-> Max Difference in Accuracy " << ChechAccuracy(ycsr.get(), y.get(), nbCols) << "\n";
    }
    {
        memset(y.get(), 0, sizeof(ValueType)*(nbCols + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y));

        std::cout << "Start usual 4rV2c: "<< std::endl;
        dtimer timerConversion;
        SPC5Mat<ValueType> csr = COO_to_SPC5_4rV2c<ValueType>(nbRows, nbCols, values.get(), nbValues);
        timerConversion.stop();
        std::cout << "Conversion in : " << timerConversion.getElapsed() << "s\n";
        dtimer timer;

        for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
            SPC5_4rV2c_Spmv<ValueType>(csr, x.get(), y.get());
        }

        timer.stop();
        std::cout << "-> Done in " << timer.getElapsed() << "s\n";
        std::cout << "-> Number of blocks " << csr.numberOfBlocks << "( avg. " << double(csr.numberOfNNZ)/double(csr.numberOfBlocks)<< " values per block)\n";
        std::cout << "-> GFlops " << double(flops)/timer.getElapsed()/1e9 << "s\n";
        std::cout << "-> Max Difference in Accuracy " << ChechAccuracy(ycsr.get(), y.get(), nbCols) << "\n";
    }
    {
        memset(y.get(), 0, sizeof(ValueType)*(nbCols + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y));

        std::cout << "Start usual 4rVc: "<< std::endl;
        dtimer timerConversion;
        SPC5Mat<ValueType> csr = COO_to_SPC5_4rVc<ValueType>(nbRows, nbCols, values.get(), nbValues);
        timerConversion.stop();
        std::cout << "Conversion in : " << timerConversion.getElapsed() << "s\n";
        dtimer timer;

        for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
            SPC5_4rVc_Spmv<ValueType>(csr, x.get(), y.get());
        }

        timer.stop();
        std::cout << "-> Done in " << timer.getElapsed() << "s\n";
        std::cout << "-> Number of blocks " << csr.numberOfBlocks << "( avg. " << double(csr.numberOfNNZ)/double(csr.numberOfBlocks)<< " values per block)\n";
        std::cout << "-> GFlops " << double(flops)/timer.getElapsed()/1e9 << "s\n";
        std::cout << "-> Max Difference in Accuracy " << ChechAccuracy(ycsr.get(), y.get(), nbCols) << "\n";
    }
    {
        memset(y.get(), 0, sizeof(ValueType)*(nbCols + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y));

        std::cout << "Start usual 8rV2c: "<< std::endl;
        dtimer timerConversion;
        SPC5Mat<ValueType> csr = COO_to_SPC5_8rV2c<ValueType>(nbRows, nbCols, values.get(), nbValues);
        timerConversion.stop();
        std::cout << "Conversion in : " << timerConversion.getElapsed() << "s\n";
        dtimer timer;

        for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
            SPC5_8rV2c_Spmv<ValueType>(csr, x.get(), y.get());
        }

        timer.stop();
        std::cout << "-> Done in " << timer.getElapsed() << "s\n";
        std::cout << "-> Number of blocks " << csr.numberOfBlocks << "( avg. " << double(csr.numberOfNNZ)/double(csr.numberOfBlocks)<< " values per block)\n";
        std::cout << "-> GFlops " << double(flops)/timer.getElapsed()/1e9 << "s\n";
        std::cout << "-> Max Difference in Accuracy " << ChechAccuracy(ycsr.get(), y.get(), nbCols) << "\n";
    }
#ifdef _OPENMP
    std::cout << "===================================================" << std::endl;
    std::cout << "Openmp is enabled with " << omp_get_max_threads() << " threads per default" << std::endl;
    {
        std::cout << "Estimate for omp: " << nbRows << std::endl;
        SPC5Mat<ValueType> csr = COO_sorted_to_CSR<ValueType>(nbRows, nbCols, values.get(), nbValues);
        std::pair<SPC5_MATRIX_TYPE, double> esimation_omp = SPC5_find_best_omp<ValueType>(csr, omp_get_max_threads());
        std::cout << "-> Esimated threaded performance are " << esimation_omp.second << " for " << SPC5_type_to_string(esimation_omp.first)
                  << " with " << omp_get_max_threads() << "\n";
    }
    {
        memset(y.get(), 0, sizeof(ValueType)*(nbCols + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y));

        std::cout << "Start openmp 1rVc: "<< std::endl;
        SPC5Mat<ValueType> csr = COO_to_SPC5_1rVc<ValueType>(nbRows, nbCols, values.get(), nbValues);
        std::vector<ThreadInterval<ValueType>> intervals = SPC5_1rVc_split_omp(csr);
        dtimer timer;

        for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
            SPC5_1rVc_Spmv_omp<ValueType>(csr, x.get(), y.get(), intervals);
        }

        timer.stop();
        std::cout << "-> Done in " << timer.getElapsed() << "s\n";
        std::cout << "-> Number of blocks " << csr.numberOfBlocks << "( avg. " << double(csr.numberOfNNZ)/double(csr.numberOfBlocks)<< " values per block)\n";
        std::cout << "-> GFlops " << double(flops)/timer.getElapsed()/1e9 << "s\n";
        std::cout << "-> Max Difference in Accuracy " << ChechAccuracy(ycsr.get(), y.get(), nbCols) << "\n";
    }
    {
        memset(y.get(), 0, sizeof(ValueType)*(nbCols + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y));

        std::cout << "Start openmp 2rV2c with test: "<< std::endl;
        SPC5Mat<ValueType> csr = COO_to_SPC5_2rV2c_wt<ValueType>(nbRows, nbCols, values.get(), nbValues);
        std::vector<ThreadInterval<ValueType>> intervals = SPC5_2rV2c_wt_split_omp(csr);
        dtimer timer;

        for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
            SPC5_2rV2c_wt_Spmv_omp<ValueType>(csr, x.get(), y.get(), intervals);
        }

        timer.stop();
        std::cout << "-> Done in " << timer.getElapsed() << "s\n";
        std::cout << "-> Number of blocks " << csr.numberOfBlocks << "( avg. " << double(csr.numberOfNNZ)/double(csr.numberOfBlocks)<< " values per block)\n";
        std::cout << "-> GFlops " << double(flops)/timer.getElapsed()/1e9 << "s\n";
        std::cout << "-> Max Difference in Accuracy " << ChechAccuracy(ycsr.get(), y.get(), nbCols) << "\n";
    }
    {
        memset(y.get(), 0, sizeof(ValueType)*(nbCols + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y));

        std::cout << "Start openmp 2rV2c: "<< std::endl;
        SPC5Mat<ValueType> csr = COO_to_SPC5_2rV2c<ValueType>(nbRows, nbCols, values.get(), nbValues);
        std::vector<ThreadInterval<ValueType>> intervals = SPC5_2rV2c_split_omp(csr);
        dtimer timer;

        for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
            SPC5_2rV2c_Spmv_omp<ValueType>(csr, x.get(), y.get(), intervals);
        }

        timer.stop();
        std::cout << "-> Done in " << timer.getElapsed() << "s\n";
        std::cout << "-> Number of blocks " << csr.numberOfBlocks << "( avg. " << double(csr.numberOfNNZ)/double(csr.numberOfBlocks)<< " values per block)\n";
        std::cout << "-> GFlops " << double(flops)/timer.getElapsed()/1e9 << "s\n";
        std::cout << "-> Max Difference in Accuracy " << ChechAccuracy(ycsr.get(), y.get(), nbCols) << "\n";
    }
    {
        memset(y.get(), 0, sizeof(ValueType)*(nbCols + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y));

        std::cout << "Start openmp 2rVc: "<< std::endl;
        SPC5Mat<ValueType> csr = COO_to_SPC5_2rVc<ValueType>(nbRows, nbCols, values.get(), nbValues);
        std::vector<ThreadInterval<ValueType>> intervals = SPC5_2rVc_split_omp(csr);
        dtimer timer;

        for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
            SPC5_2rVc_Spmv_omp<ValueType>(csr, x.get(), y.get(), intervals);
        }

        timer.stop();
        std::cout << "-> Done in " << timer.getElapsed() << "s\n";
        std::cout << "-> Number of blocks " << csr.numberOfBlocks << "( avg. " << double(csr.numberOfNNZ)/double(csr.numberOfBlocks)<< " values per block)\n";
        std::cout << "-> GFlops " << double(flops)/timer.getElapsed()/1e9 << "s\n";
        std::cout << "-> Max Difference in Accuracy " << ChechAccuracy(ycsr.get(), y.get(), nbCols) << "\n";
    }
    {
        memset(y.get(), 0, sizeof(ValueType)*(nbCols + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y));

        std::cout << "Start usual 4rV2c: "<< std::endl;
        SPC5Mat<ValueType> csr = COO_to_SPC5_4rV2c<ValueType>(nbRows, nbCols, values.get(), nbValues);
        std::vector<ThreadInterval<ValueType>> intervals = SPC5_4rV2c_split_omp(csr);
        dtimer timer;

        for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
            SPC5_4rV2c_Spmv_omp<ValueType>(csr, x.get(), y.get(), intervals);
        }

        timer.stop();
        std::cout << "-> Done in " << timer.getElapsed() << "s\n";
        std::cout << "-> Number of blocks " << csr.numberOfBlocks << "( avg. " << double(csr.numberOfNNZ)/double(csr.numberOfBlocks)<< " values per block)\n";
        std::cout << "-> GFlops " << double(flops)/timer.getElapsed()/1e9 << "s\n";
        std::cout << "-> Max Difference in Accuracy " << ChechAccuracy(ycsr.get(), y.get(), nbCols) << "\n";
    }
    {
        memset(y.get(), 0, sizeof(ValueType)*(nbCols + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y));

        std::cout << "Start openmp 4rVc: "<< std::endl;
        SPC5Mat<ValueType> csr = COO_to_SPC5_4rVc<ValueType>(nbRows, nbCols, values.get(), nbValues);
        std::vector<ThreadInterval<ValueType>> intervals = SPC5_4rVc_split_omp(csr);
        dtimer timer;

        for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
            SPC5_4rVc_Spmv_omp<ValueType>(csr, x.get(), y.get(), intervals);
        }

        timer.stop();
        std::cout << "-> Done in " << timer.getElapsed() << "s\n";
        std::cout << "-> Number of blocks " << csr.numberOfBlocks << "( avg. " << double(csr.numberOfNNZ)/double(csr.numberOfBlocks)<< " values per block)\n";
        std::cout << "-> GFlops " << double(flops)/timer.getElapsed()/1e9 << "s\n";
        std::cout << "-> Max Difference in Accuracy " << ChechAccuracy(ycsr.get(), y.get(), nbCols) << "\n";
    }
    {
        memset(y.get(), 0, sizeof(ValueType)*(nbCols + SPC5_VEC_PADDING::SPC5_VEC_PADDING_Y));

        std::cout << "Start openmp 8rV2c: "<< std::endl;
        SPC5Mat<ValueType> csr = COO_to_SPC5_8rV2c<ValueType>(nbRows, nbCols, values.get(), nbValues);
        std::vector<ThreadInterval<ValueType>> intervals = SPC5_8rV2c_split_omp(csr);
        dtimer timer;

        for(int idxLoop = 0 ; idxLoop < nbLoops ; ++idxLoop){
            SPC5_8rV2c_Spmv_omp<ValueType>(csr, x.get(), y.get(), intervals);
        }

        timer.stop();
        std::cout << "-> Done in " << timer.getElapsed() << "s\n";
        std::cout << "-> Number of blocks " << csr.numberOfBlocks << "( avg. " << double(csr.numberOfNNZ)/double(csr.numberOfBlocks)<< " values per block)\n";
        std::cout << "-> GFlops " << double(flops)/timer.getElapsed()/1e9 << "s\n";
        std::cout << "-> Max Difference in Accuracy " << ChechAccuracy(ycsr.get(), y.get(), nbCols) << "\n";
    }
#endif
    return 0;
}
