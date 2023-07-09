#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#else
#include "farm_sve.h"
#endif /* __ARM_FEATURE_SVE */

#include "spc5.hpp"

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

            const svfloat64_t xvals = svcompact(mask_vec, svld1(mask_vec, &x[idxCol]));
            const svfloat64_t xvals_1 = svcompact(mask_vec_1, svld1(mask_vec_1, &x[idxCol]));

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

            const svfloat32_t xvals = svcompact(mask_vec, svld1(mask_vec, &x[idxCol]));
            const svfloat32_t xvals_1 = svcompact(mask_vec_1, svld1(mask_vec_1, &x[idxCol]));

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

            const svfloat64_t xvals = svcompact(mask_vec, svld1(mask_vec, &x[idxCol]));
            const svfloat64_t xvals_1 = svcompact(mask_vec_1, svld1(mask_vec_1, &x[idxCol]));
            const svfloat64_t xvals_2 = svcompact(mask_vec_2, svld1(mask_vec_2, &x[idxCol]));
            const svfloat64_t xvals_3 = svcompact(mask_vec_3, svld1(mask_vec_3, &x[idxCol]));

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

            const svfloat32_t xvals = svcompact(mask_vec, svld1(mask_vec, &x[idxCol]));
            const svfloat32_t xvals_1 = svcompact(mask_vec_1, svld1(mask_vec_1, &x[idxCol]));
            const svfloat32_t xvals_2 = svcompact(mask_vec_2, svld1(mask_vec_2, &x[idxCol]));
            const svfloat32_t xvals_3 = svcompact(mask_vec_3, svld1(mask_vec_3, &x[idxCol]));

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
