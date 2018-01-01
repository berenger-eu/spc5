#include "spc5.hpp"


//////////////////////////////////////////////////////////////////////////

extern "C" void core_SPC5_1rVc_Spmv_asm_double(const long int nbRows, const int* rowsSizes,
                                   const unsigned char* headers,
                                                   const double* values,
                                                   const double* x, double* y);

/* (nbRows rdi , rowsSizes rsi , headers rdx , values rcx, x r8, y r9 ) */
/* r10 can be used. Must be saved : r11/r12/r13/r14/r15 */
__asm__(
".global core_SPC5_1rVc_Spmv_asm_double \n"
"core_SPC5_1rVc_Spmv_asm_double: \n"
// Save the callee registers
"pushq   %r15;\n"
"pushq   %r14;\n"
"pushq   %r13;\n"
"pushq   %r12;\n"
"pushq   %r11;\n"
"pushq   %r10;\n"
"pushq   %rax;\n"
"pushq   %rbx;\n"

"xorq %r12,%r12; \n"    // valIdx = 0

// if no rows in the matrix, jump to end
"test %rdi, %rdi;\n"
"jz   compute_Spmv512_avx_asm_double_out_exp;\n"
    "xorq %r10, %r10; \n" // idxRow/r10 = 0

    "compute_Spmv512_avx_asm_double_loop_exp:\n"

    "movslq 4(%rsi,%r10,4),%r11; \n"   // rowsSizes[idxRow+1]
    "subl 0(%rsi,%r10,4),%r11d; \n"    // nbBlocks = rowsSizes[idxRow+1]-rowsSizes[idxRow]
    // if no blocks for this row, jump to next interval
    "jz compute_Spmv512_avx_asm_double_outrow_exp; \n"

    "vpxorq %zmm0,%zmm0,%zmm0; \n"    // %zmm0 sum = 0
//#ifndef NO1TEST
    "vpxorq %zmm10,%zmm10,%zmm10; \n" // %zmm10 sum = 0

    "compute_Spmv512_avx_asm_double_inrow_exp_NOmorethan1:"
        "movslq 0(%rdx),%r13; \n"    // colIdx = *rdx or *headers
        "movzbl  4(%rdx), %r14d;\n"  // mask = *(rdx+4) or *(headers+4)

        "cmp $1,%r14d;\n" // if mask is not equal to one go to the vector based loop
        "jne compute_Spmv512_avx_asm_double_inrow_exp_morethan1;\n"
        "compute_Spmv512_avx_asm_double_inrow_exp_doNOmorethan1:\n"

        "vmovsd (%rcx,%r12,8), %xmm7; \n"  // values
        #ifndef NO_FMADD
        "vfmadd231sd	(%r8,%r13,8), %xmm7, %xmm10;\n" // mul add to sum
        #else
        "mulpd (%r8,%r13,8), %xmm7,  %xmm7;\n"
        "addpd %xmm7, %xmm10, %xmm10; \n"  // add to sum
        #endif

        "addq $1,%r12;\n"     // valIdx += 1 because there is only one bit in the mask

        "addq	$5, %rdx; \n" // headers += 1 int + 1 byte mask

        "subq $1,%r11; \n"    // nbBlocks -=1, if equal zero go to end of interval
        "jnz compute_Spmv512_avx_asm_double_inrow_exp_NOmorethan1; \n"
        "jmp compute_Spmv512_avx_asm_double_inrow_exp_stop;\n"
//#endif
    "compute_Spmv512_avx_asm_double_inrow_exp:"
            "movslq 0(%rdx),%r13; \n"    // colIdx = *rdx or *headers
            "movzbl  4(%rdx), %r14d;\n"  // mask = *(rdx+4) or *(headers+4)
//#ifndef NO1TEST
            "cmp $1,%r14d;\n" // if mask is equal to one go to the scalar based loop
            "je compute_Spmv512_avx_asm_double_inrow_exp_doNOmorethan1;\n"
            "compute_Spmv512_avx_asm_double_inrow_exp_morethan1:\n"
//#endif

            "kmovw   %r14d, %k1;\n"      // mask

            "vexpandpd (%rcx,%r12,8), %zmm1{%k1}{z}; \n"    // values (only some of them)
            #ifndef NO_FMADD
            "vfmadd231pd (%r8,%r13,8), %zmm1, %zmm0;\n" // mul add to sum
            #else
            "vmulpd (%r8,%r13,8), %zmm1,  %zmm1;\n"
            "vaddpd %zmm1, %zmm0, %zmm0; \n"            // add to sum
            #endif

            "popcntw %r14w, %r14w;\n" // count the number of bits in the mask
            "addq %r14,%r12;\n" // valIdx += number of bits(mask)

            "addq	$5, %rdx; \n" // headers += 1 * int + 1 mask

            "subq $1,%r11; \n"    // nbBlocks -=1, if equal zero go to end of interval
    "jnz compute_Spmv512_avx_asm_double_inrow_exp; \n"

    "compute_Spmv512_avx_asm_double_inrow_exp_stop:\n"
//#ifndef NO1TEST
    // Add the contribution from the scalar loop
    "vaddpd  %zmm10, %zmm0, %zmm0;\n"
//#endif
    // Horizontal sum
    "vmovapd %ymm0, %ymm1;\n"
    "vextractf64x4   $0x1, %zmm0, %ymm0;\n"
    "vaddpd  %ymm0, %ymm1, %ymm1;\n"

    "vextractf128    $0x1, %ymm1, %xmm2;\n"
    "vaddpd  %xmm1, %xmm2, %xmm1;\n"

    "vpermilpd       $1, %xmm1, %xmm0;\n"
    "vaddpd  %xmm1, %xmm0, %xmm0;\n"
    // add to y
    "vaddsd  (%r9,%r10,8), %xmm0, %xmm0;\n"
    "vmovsd  %xmm0, (%r9,%r10,8);\n"

    "compute_Spmv512_avx_asm_double_outrow_exp:\n"

    "addq $1, %r10;\n"   // idxRow += 1

    "cmp %rdi, %r10;\n" // idxRow == nbRows
    "jne compute_Spmv512_avx_asm_double_loop_exp;\n" // if nbBlocks != 0 go to beginning

    "NOP;\n"

"compute_Spmv512_avx_asm_double_out_exp:\n"

"popq    %rbx;\n"
"popq    %rax;\n"
"popq    %r10;\n"
"popq    %r11;\n"
"popq    %r12;\n"
"popq    %r13;\n"
"popq    %r14;\n"
"popq    %r15;\n"

"ret;\n"
);

extern "C" void core_SPC5_1rVc_Spmv_asm_float(const long int nbRows, const int* rowsSizes,
                                                 const unsigned char* headers,
                                                                 const float* values,
                                                                 const float* x, float* y);

/* (nbRows rdi , rowsSizes rsi , headers rdx , values rcx, x r8, y r9 ) */
/* r10. must be saved r11/r12/r13/r14/r15 */
__asm__(
".global core_SPC5_1rVc_Spmv_asm_float \n"
"core_SPC5_1rVc_Spmv_asm_float: \n"

"pushq   %r15;\n"
"pushq   %r14;\n"
"pushq   %r13;\n"
"pushq   %r12;\n"
"pushq   %r11;\n"
"pushq   %r10;\n"
"pushq   %rax;\n"
"pushq   %rbx;\n"

"xorq %r12,%r12; \n"    // valIdx

// if no rows, jump to end
"test %rdi, %rdi;\n"
"jz   compute_Spmv512_avx_asm_float_out_exp;\n"
    "xorq %r10, %r10; \n" // r10 = 0 idxRow

    "compute_Spmv512_avx_asm_float_loop_exp:\n"

    "movslq 4(%rsi,%r10,4),%r11; \n"  // rowsSizes[idxRow+1]
    "subl 0(%rsi,%r10,4),%r11d; \n"   // rowsSizes[idxRow+1]-rowsSizes[idxRow]
    // if no blocks for this row, jumpb
    "jz compute_Spmv512_avx_asm_float_outrow_exp; \n"

    "vpxorq %zmm0,%zmm0,%zmm0; \n" // %zmm0 sum

    "compute_Spmv512_avx_asm_float_inrow_exp:"
        "movslq 0(%rdx),%r13; \n"    // colIdx
        "movzwl  4(%rdx), %r14d;\n"  // mask
        "kmovw   %r14d, %k1;\n"      // mask

        "vexpandps (%rcx,%r12,4), %zmm1{%k1}{z}; \n"  // values
        //"vmovdqu64 (%r8,%r13,4), %zmm2; \n"           // x
        #ifndef NO_FMADD
        "vfmadd231ps	(%r8,%r13,4), %zmm1, %zmm0;\n"
        #else
        "vmulps (%r8,%r13,4), %zmm1,  %zmm1;\n"
        "vaddps %zmm1, %zmm0, %zmm0; \n"  //values
        #endif

        "popcntw %r14w, %r14w;\n"
        "addq %r14,%r12;\n"

        "addq	$6, %rdx; \n" // headers += 1 * c + 2 * i => 1 + 8 but rounded to 12

        "subq $1,%r11; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]-1...
    "jnz compute_Spmv512_avx_asm_float_inrow_exp; \n"
    "compute_Spmv512_avx_asm_float_inrow_exp_stop:\n"

    "vmovaps %ymm0, %ymm1;\n"
    "vextractf64x4   $0x1, %zmm0, %ymm0;\n"
    "vaddps  %ymm0, %ymm1, %ymm1;\n"

    "vextractf128    $0x1, %ymm1, %xmm0;\n"
    "vaddps  %xmm1, %xmm0, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  (%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, (%r9,%r10,4);\n"

    "compute_Spmv512_avx_asm_float_outrow_exp:\n"

    "addq $1, %r10;\n"   // idxRow += 1

    "cmp %rdi, %r10;\n" // idxRow == nbRows
    "jne compute_Spmv512_avx_asm_float_loop_exp;\n" // if nbBlocks != 0 go to beginning

    "NOP;\n" // To avoid the presence of ret just after the jne

"compute_Spmv512_avx_asm_float_out_exp:\n"

"popq    %rbx;\n"
"popq    %rax;\n"
"popq    %r10;\n"
"popq    %r11;\n"
"popq    %r12;\n"
"popq    %r13;\n"
"popq    %r14;\n"
"popq    %r15;\n"

"ret;\n"
);


//////////////////////////////////////////////////////////////////////////

extern "C" void core_SPC5_2rV2c_wt_Spmv_asm_double(const long int nbRows, const int* rowsSizes,
                                   const unsigned char* headers,
                                                   const double* values,
                                                   const double* x, double* y);

/* (nbRows rdi , rowsSizes rsi , headers rdx , values rcx, x r8, y r9 ) */
/* r10. must be saved r11/r12/r13/r14/r15 */
__asm__(
".global core_SPC5_2rV2c_wt_Spmv_asm_double \n"
"core_SPC5_2rV2c_wt_Spmv_asm_double: \n"

"pushq   %r15;\n"
"pushq   %r14;\n"
"pushq   %r13;\n"
"pushq   %r12;\n"
"pushq   %r11;\n"
"pushq   %r10;\n"
"pushq   %rax;\n"
"pushq   %rbx;\n"

"xorq %r12,%r12; \n"    // valIdx

// if no rows, jump to end
"test %rdi, %rdi;\n"
"jz   compute_2r_Spmv512_avx_asm_double_out_exp;\n"
    "xorq %r10, %r10; \n" // r10 = 0 idxRow

    "compute_2r_Spmv512_avx_asm_double_loop_exp:\n"

    "movslq 4(%rsi,%r10,4),%r11; \n"  // rowsSizes[idxRow+1]
    "subl 0(%rsi,%r10,4),%r11d; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]
    // if no blocks for this row, jumpb
    "jz compute_2r_Spmv512_avx_asm_double_outrow_exp; \n"

    "vpxor %ymm0,%ymm0,%ymm0; \n" // %zmm0 sum
    "vpxor %ymm3,%ymm3,%ymm3; \n" // %zmm0 sum
//#ifndef NO1TEST
    "vpxor %ymm10,%ymm10,%ymm10; \n" // %zmm0 sum
    "vpxor %ymm11,%ymm11,%ymm11; \n" // %zmm0 sum

    "compute_2r_Spmv512_avx_asm_double_inrow_exp_nomorethan1_loop:"
        "movslq 0(%rdx),%r13; \n"    // colIdx
        "movzbl  4(%rdx), %r14d;\n"  // mask
        "cmp $16,%r14d;\n"
        "je compute_2r_Spmv512_avx_asm_double_inrow_exp_nomorethan1_row2;\n"
        "cmp $1,%r14d;\n"
        "jne compute_2r_Spmv512_avx_asm_double_inrow_exp_morethan1;\n"
        "compute_2r_Spmv512_avx_asm_double_inrow_exp_nomorethan1:\n"
        "vmovsd (%rcx,%r12,8), %xmm7; \n"  // values
        #ifndef NO_FMADD
        "vfmadd231sd	(%r8,%r13,8), %xmm7, %xmm10;\n"
        #else
        "mulpd (%r8,%r13,8), %xmm7,  %xmm7;\n"
        "addpd %xmm7, %xmm10, %xmm10; \n"  //values
        #endif

        "addq $1,%r12;\n"

        "addq	$5, %rdx; \n" // headers += 1 * c + 2 * i => 1 + 8 but rounded to 12

        "subq $1,%r11; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]-1...
        "jnz compute_2r_Spmv512_avx_asm_double_inrow_exp_nomorethan1_loop; \n"
        "jmp compute_2r_Spmv512_avx_asm_double_inrow_exp_stop;\n"

    "compute_2r_Spmv512_avx_asm_double_inrow_exp_nomorethan1_row2_loop:"
        "movslq 0(%rdx),%r13; \n"    // colIdx
        "movzbl  4(%rdx), %r14d;\n"  // mask
        "cmp $1,%r14d;\n"
        "je compute_2r_Spmv512_avx_asm_double_inrow_exp_nomorethan1;\n"
        "cmp $16,%r14d;\n"
        "jne compute_2r_Spmv512_avx_asm_double_inrow_exp_morethan1;\n"
        "compute_2r_Spmv512_avx_asm_double_inrow_exp_nomorethan1_row2:\n"
        "vmovsd (%rcx,%r12,8), %xmm7; \n"  // values
        #ifndef NO_FMADD
        "vfmadd231sd	(%r8,%r13,8), %xmm7, %xmm11;\n"
        #else
        "mulpd (%r8,%r13,8), %xmm7,  %xmm7;\n"
        "addpd %xmm7, %xmm11, %xmm11; \n"  //values
        #endif

        "addq $1,%r12;\n"

        "addq	$5, %rdx; \n" // headers += 1 * c + 2 * i => 1 + 8 but rounded to 12

        "subq $1,%r11; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]-1...
        "jnz compute_2r_Spmv512_avx_asm_double_inrow_exp_nomorethan1_row2_loop; \n"
        "jmp compute_2r_Spmv512_avx_asm_double_inrow_exp_stop;\n"
//#endif
    "compute_2r_Spmv512_avx_asm_double_inrow_exp:"
        "movslq 0(%rdx),%r13; \n"    // colIdx
        "movzbl  4(%rdx), %r14d;\n"  // mask
//#ifndef NO1TEST
        "cmp $1,%r14d;\n"
        "je compute_2r_Spmv512_avx_asm_double_inrow_exp_nomorethan1;\n"
        "cmp $16,%r14d;\n"
        "je compute_2r_Spmv512_avx_asm_double_inrow_exp_nomorethan1_row2;\n"
        "compute_2r_Spmv512_avx_asm_double_inrow_exp_morethan1:\n"
//#endif
        "kmovw   %r14d, %k1;\n"      // mask

        "vexpandpd (%rcx,%r12,8), %zmm1{%k1}{z}; \n"  // values
        "vmovupd (%r8,%r13,8), %ymm2; \n"           // x
        "vextractf64x4   $0x1, %zmm1, %ymm4;\n"

        #ifndef NO_FMADD
        "vfmadd231pd	%ymm2, %ymm1, %ymm0;\n"
        "vfmadd231pd	%ymm2, %ymm4, %ymm3;\n"
        #else
        "vmulpd %ymm2, %ymm1,  %ymm1;\n"
        "vaddpd %ymm1, %ymm0, %ymm0; \n"  //values
        "vmulpd %ymm2, %ymm4,  %ymm4;\n"
        "vaddpd %ymm4, %ymm3, %ymm3; \n"  //values
        #endif

        "popcntw %r14w, %r14w;\n"
        "addq %r14,%r12;\n"

        "addq	$5, %rdx; \n" // headers += 1 * c + 2 * i => 1 + 8 but rounded to 12

        "subq $1,%r11; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]-1...
    "jnz compute_2r_Spmv512_avx_asm_double_inrow_exp; \n"
    "compute_2r_Spmv512_avx_asm_double_inrow_exp_stop:\n"
//#ifndef NO1TEST
    "vaddpd  %ymm10, %ymm0, %ymm0;\n"
    "vaddpd  %ymm11, %ymm3, %ymm3;\n"
//#endif
    "vhaddpd    %ymm3, %ymm0, %ymm0;\n"
    "vextractf128    $0x1, %ymm0, %xmm4;\n"
    "vaddpd  %xmm4, %xmm0, %xmm0;\n"
    "vaddpd  (%r9,%r10,8), %xmm0, %xmm0;\n"
    "vmovupd  %xmm0, (%r9,%r10,8);\n"

    "compute_2r_Spmv512_avx_asm_double_outrow_exp:\n"

    "addq $2, %r10;\n"   // idxRow += 2

    "cmp %rdi, %r10;\n" // idxRow == nbRows
    "jl compute_2r_Spmv512_avx_asm_double_loop_exp;\n" // if nbBlocks != 0 go to beginning

    "NOP;\n" // To avoid the presence of ret just after the jne

"compute_2r_Spmv512_avx_asm_double_out_exp:\n"

"popq    %rbx;\n"
"popq    %rax;\n"
"popq    %r10;\n"
"popq    %r11;\n"
"popq    %r12;\n"
"popq    %r13;\n"
"popq    %r14;\n"
"popq    %r15;\n"

"ret;\n"
);


extern "C" void core_SPC5_2rV2c_Spmv_asm_double(const long int nbRows, const int* rowsSizes,
                                   const unsigned char* headers,
                                                   const double* values,
                                                   const double* x, double* y);

/* (nbRows rdi , rowsSizes rsi , headers rdx , values rcx, x r8, y r9 ) */
/* r10. must be saved r11/r12/r13/r14/r15 */
__asm__(
".global core_SPC5_2rV2c_Spmv_asm_double \n"
"core_SPC5_2rV2c_Spmv_asm_double: \n"

"pushq   %r15;\n"
"pushq   %r14;\n"
"pushq   %r13;\n"
"pushq   %r12;\n"
"pushq   %r11;\n"
"pushq   %r10;\n"
"pushq   %rax;\n"
"pushq   %rbx;\n"

"xorq %r12,%r12; \n"    // valIdx

// if no rows, jump to end
"test %rdi, %rdi;\n"
"jz   compute_2r_notest_Spmv512_avx_asm_double_out_exp;\n"
    "xorq %r10, %r10; \n" // r10 = 0 idxRow

    "compute_2r_notest_Spmv512_avx_asm_double_loop_exp:\n"

    "movslq 4(%rsi,%r10,4),%r11; \n"  // rowsSizes[idxRow+1]
    "subl 0(%rsi,%r10,4),%r11d; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]
    // if no blocks for this row, jumpb
    "jz compute_2r_notest_Spmv512_avx_asm_double_outrow_exp; \n"

    "vpxor %ymm0,%ymm0,%ymm0; \n" // %zmm0 sum
    "vpxor %ymm3,%ymm3,%ymm3; \n" // %zmm0 sum

    "compute_2r_notest_Spmv512_avx_asm_double_inrow_exp:"
        "movslq 0(%rdx),%r13; \n"    // colIdx
        "movzbl  4(%rdx), %r14d;\n"  // mask
        "kmovw   %r14d, %k1;\n"      // mask

        "vexpandpd (%rcx,%r12,8), %zmm1{%k1}{z}; \n"  // values
        "vmovupd (%r8,%r13,8), %ymm2; \n"           // x
        "vextractf64x4   $0x1, %zmm1, %ymm4;\n"

        #ifndef NO_FMADD
        "vfmadd231pd	%ymm2, %ymm1, %ymm0;\n"
        "vfmadd231pd	%ymm2, %ymm4, %ymm3;\n"
        #else
        "vmulpd %ymm2, %ymm1,  %ymm1;\n"
        "vaddpd %ymm1, %ymm0, %ymm0; \n"  //values
        "vmulpd %ymm2, %ymm4,  %ymm4;\n"
        "vaddpd %ymm4, %ymm3, %ymm3; \n"  //values
        #endif

        "popcntw %r14w, %r14w;\n"
        "addq %r14,%r12;\n"

        "addq	$5, %rdx; \n" // headers += 1 * c + 2 * i => 1 + 8 but rounded to 12

        "subq $1,%r11; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]-1...
    "jnz compute_2r_notest_Spmv512_avx_asm_double_inrow_exp; \n"
    "compute_2r_notest_Spmv512_avx_asm_double_inrow_exp_stop:\n"

    "vhaddpd    %ymm3, %ymm0, %ymm0;\n"
    "vextractf128    $0x1, %ymm0, %xmm4;\n"
    "vaddpd  %xmm4, %xmm0, %xmm0;\n"
    "vaddpd  (%r9,%r10,8), %xmm0, %xmm0;\n"
    "vmovupd  %xmm0, (%r9,%r10,8);\n"

    "compute_2r_notest_Spmv512_avx_asm_double_outrow_exp:\n"

    "addq $2, %r10;\n"   // idxRow += 2

    "cmp %rdi, %r10;\n" // idxRow == nbRows
    "jl compute_2r_notest_Spmv512_avx_asm_double_loop_exp;\n" // if nbBlocks != 0 go to beginning

    "NOP;\n" // To avoid the presence of ret just after the jne

"compute_2r_notest_Spmv512_avx_asm_double_out_exp:\n"

"popq    %rbx;\n"
"popq    %rax;\n"
"popq    %r10;\n"
"popq    %r11;\n"
"popq    %r12;\n"
"popq    %r13;\n"
"popq    %r14;\n"
"popq    %r15;\n"

"ret;\n"
);

extern "C" void core_SPC5_2rV2c_Spmv_asm_float(const long int nbRows, const int* rowsSizes,
                                   const unsigned char* headers,
                                                   const float* values,
                                                   const float* x, float* y);

/* (nbRows rdi , rowsSizes rsi , headers rdx , values rcx, x r8, y r9 ) */
/* r10. must be saved r11/r12/r13/r14/r15 */
__asm__(
".global core_SPC5_2rV2c_Spmv_asm_float \n"
"core_SPC5_2rV2c_Spmv_asm_float: \n"

"pushq   %r15;\n"
"pushq   %r14;\n"
"pushq   %r13;\n"
"pushq   %r12;\n"
"pushq   %r11;\n"
"pushq   %r10;\n"
"pushq   %rax;\n"
"pushq   %rbx;\n"

"xorq %r12,%r12; \n"    // valIdx

// if no rows, jump to end
"test %rdi, %rdi;\n"
"jz   compute_2r_Spmv512_avx_asm_float_out_exp;\n"
    "xorq %r10, %r10; \n" // r10 = 0 idxRow

    "compute_2r_Spmv512_avx_asm_float_loop_exp:\n"

    "movslq 4(%rsi,%r10,4),%r11; \n"  // rowsSizes[idxRow+1]
    "subl 0(%rsi,%r10,4),%r11d; \n"   // rowsSizes[idxRow+1]-rowsSizes[idxRow]
    // if no blocks for this row, jumpb
    "jz compute_2r_Spmv512_avx_asm_float_outrow_exp; \n"

    "vpxor %ymm0,%ymm0,%ymm0; \n" // %zmm0 sum
    "vpxor %ymm3,%ymm3,%ymm3; \n" // %zmm0 sum

    "compute_2r_Spmv512_avx_asm_float_inrow_exp:"
        "movslq 0(%rdx),%r13; \n"    // colIdx
        "movzwl  4(%rdx), %r14d;\n"  // mask
        "kmovw   %r14d, %k1;\n"      // mask

        "vexpandps (%rcx,%r12,4), %zmm1{%k1}{z}; \n"  // values
        "vmovupd (%r8,%r13,4), %ymm2; \n"           // x
        "vextractf64x4   $0x1, %zmm1, %ymm4;\n"

        #ifndef NO_FMADD
        "vfmadd231ps	%ymm2, %ymm1, %ymm0;\n"
        "vfmadd231ps	%ymm2, %ymm4, %ymm3;\n"
        #else
        "vmulps %ymm2, %ymm1,  %ymm1;\n"
        "vaddps %ymm1, %ymm0, %ymm0; \n"  //values
        "vmulps %ymm2, %ymm4,  %ymm4;\n"
        "vaddps %ymm4, %ymm3, %ymm3; \n"  //values
        #endif

        "popcntw %r14w, %r14w;\n"
        "addq %r14,%r12;\n"

        "addq	$6, %rdx; \n" // headers += 1 * c + 2 * i => 1 + 8 but rounded to 12

        "subq $1,%r11; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]-1...
    "jnz compute_2r_Spmv512_avx_asm_float_inrow_exp; \n"
    "compute_2r_Spmv512_avx_asm_float_inrow_exp_stop:\n"

    "vextractf128    $0x1, %ymm0, %xmm1;\n"
    "vaddps  %xmm1, %xmm0, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  (%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, (%r9,%r10,4);\n"

    "vextractf128    $0x1, %ymm3, %xmm1;\n"
    "vaddps  %xmm1, %xmm3, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  4(%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, 4(%r9,%r10,4);\n"

    "compute_2r_Spmv512_avx_asm_float_outrow_exp:\n"

    "addq $2, %r10;\n"   // idxRow += 1

    "cmp %rdi, %r10;\n" // idxRow == nbRows
    "jl compute_2r_Spmv512_avx_asm_float_loop_exp;\n" // if nbBlocks != 0 go to beginning

    "NOP;\n" // To avoid the presence of ret just after the jne

"compute_2r_Spmv512_avx_asm_float_out_exp:\n"

"popq    %rbx;\n"
"popq    %rax;\n"
"popq    %r10;\n"
"popq    %r11;\n"
"popq    %r12;\n"
"popq    %r13;\n"
"popq    %r14;\n"
"popq    %r15;\n"

"ret;\n"
);

//////////////////////////////////////////////////////////////////////////
extern "C" void core_SPC5_2rVc_Spmv_asm_double(const long int nbRows, const int* rowsSizes,
                                   const unsigned char* headers,
                                                   const double* values,
                                                   const double* x, double* y);

/* (nbRows rdi , rowsSizes rsi , headers rdx , values rcx, x r8, y r9 ) */
/* r10. must be saved r11/r12/r13/r14/r15 */
__asm__(
".global core_SPC5_2rVc_Spmv_asm_double \n"
"core_SPC5_2rVc_Spmv_asm_double: \n"

"pushq   %r15;\n"
"pushq   %r14;\n"
"pushq   %r13;\n"
"pushq   %r12;\n"
"pushq   %r11;\n"
"pushq   %r10;\n"
"pushq   %rax;\n"
"pushq   %rbx;\n"

"xorq %r12,%r12; \n"    // valIdx

// if no rows, jump to end
"test %rdi, %rdi;\n"
"jz   compute_2r2_Spmv512_avx_asm_double_out_exp;\n"
    "xorq %r10, %r10; \n" // r10 = 0 idxRow

    "compute_2r2_Spmv512_avx_asm_double_loop_exp:\n"

    "movslq 4(%rsi,%r10,4),%r11; \n"  // rowsSizes[idxRow+1]
    "subl 0(%rsi,%r10,4),%r11d; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]
    // if no blocks for this row, jumpb
    "jz compute_2r2_Spmv512_avx_asm_double_outrow_exp; \n"

    "vpxorq %zmm0,%zmm0,%zmm0; \n" // %zmm0 sum
    "vpxorq %zmm3,%zmm3,%zmm3; \n" // %zmm0 sum

    "compute_2r2_Spmv512_avx_asm_double_inrow_exp:"

        "movslq 0(%rdx),%r13; \n"    // colIdx
        "movzbl  4(%rdx), %r14d;\n"  // mask
        "movzbl  5(%rdx), %r15d;\n"  // mask

        "kmovw   %r14d, %k1;\n"      // mask
        "kmovw   %r15d, %k2;\n"      // mask

        "vexpandpd (%rcx,%r12,8), %zmm1{%k1}{z}; \n"  // values
        "vmovdqu64 (%r8,%r13,8), %zmm2; \n"           // x

        "popcntw %r14w, %r14w;\n"
        "addq %r14,%r12;\n"

        "vexpandpd (%rcx,%r12,8), %zmm4{%k2}{z}; \n"  // values
        "popcntw %r15w, %r15w;\n"
        "addq %r15,%r12;\n"

        #ifndef NO_FMADD
        "vfmadd231pd	%zmm2, %zmm1, %zmm0;\n"
        "vfmadd231pd	%zmm2, %zmm4, %zmm3;\n"
        #else
        "vmulpd %zmm2, %zmm1,  %zmm1;\n"
        "vaddpd %zmm1, %zmm0, %zmm0; \n"  //values
        "vmulpd %zmm2, %zmm4,  %zmm4;\n"
        "vaddpd %zmm4, %zmm3, %zmm3; \n"  //values
        #endif

        "addq	$6, %rdx; \n" // headers += 1 * c + 2 * i => 1 + 8 but rounded to 12

        "subq $1,%r11; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]-1...
    "jnz compute_2r2_Spmv512_avx_asm_double_inrow_exp; \n"
    "compute_2r2_Spmv512_avx_asm_double_inrow_exp_stop:\n"

    "vextractf64x4   $0x1, %zmm0, %ymm1;\n"
    "vextractf64x4   $0x1, %zmm3, %ymm4;\n"

    "vaddpd  %ymm1, %ymm0, %ymm0;\n"
    "vaddpd  %ymm4, %ymm3, %ymm3;\n"

    "vhaddpd    %ymm3, %ymm0, %ymm0;\n"
    "vextractf128    $0x1, %ymm0, %xmm4;\n"
    "vaddpd  %xmm4, %xmm0, %xmm0;\n"

    "vaddpd  (%r9,%r10,8), %xmm0, %xmm0;\n"
    "vmovupd  %xmm0, (%r9,%r10,8);\n"

    "compute_2r2_Spmv512_avx_asm_double_outrow_exp:\n"

    "addq $2, %r10;\n"   // idxRow += 2

    "cmp %rdi, %r10;\n" // idxRow == nbRows
    "jl compute_2r2_Spmv512_avx_asm_double_loop_exp;\n" // if nbBlocks != 0 go to beginning

    "NOP;\n" // To avoid the presence of ret just after the jne

"compute_2r2_Spmv512_avx_asm_double_out_exp:\n"

"popq    %rbx;\n"
"popq    %rax;\n"
"popq    %r10;\n"
"popq    %r11;\n"
"popq    %r12;\n"
"popq    %r13;\n"
"popq    %r14;\n"
"popq    %r15;\n"

"ret;\n"
);

extern "C" void core_SPC5_2rVc_Spmv_asm_float(const long int nbRows, const int* rowsSizes,
                                   const unsigned char* headers,
                                                   const float* values,
                                                   const float* x, float* y);

/* (nbRows rdi , rowsSizes rsi , headers rdx , values rcx, x r8, y r9 ) */
/* r10. must be saved r11/r12/r13/r14/r15 */
__asm__(
".global core_SPC5_2rVc_Spmv_asm_float \n"
"core_SPC5_2rVc_Spmv_asm_float: \n"

"pushq   %r15;\n"
"pushq   %r14;\n"
"pushq   %r13;\n"
"pushq   %r12;\n"
"pushq   %r11;\n"
"pushq   %r10;\n"
"pushq   %rax;\n"
"pushq   %rbx;\n"

"xorq %r12,%r12; \n"    // valIdx

// if no rows, jump to end
"test %rdi, %rdi;\n"
"jz   compute_2r2_Spmv512_avx_asm_float_out_exp;\n"
    "xorq %r10, %r10; \n" // r10 = 0 idxRow

    "compute_2r2_Spmv512_avx_asm_float_loop_exp:\n"

    "movslq 4(%rsi,%r10,4),%r11; \n"  // rowsSizes[idxRow+1]
    "subl 0(%rsi,%r10,4),%r11d; \n"   // rowsSizes[idxRow+1]-rowsSizes[idxRow]
    // if no blocks for this row, jumpb
    "jz compute_2r2_Spmv512_avx_asm_float_outrow_exp; \n"

    "vpxorq %zmm0,%zmm0,%zmm0; \n" // %zmm0 sum
    "vpxorq %zmm3,%zmm3,%zmm3; \n" // %zmm0 sum

    "compute_2r2_Spmv512_avx_asm_float_inrow_exp:"
        "movslq 0(%rdx),%r13; \n"    // colIdx
        "movzwl  4(%rdx), %r14d;\n"  // mask
        "kmovw   %r14d, %k1;\n"      // mask

        "movzwl  6(%rdx), %r15d;\n"  // mask
        "kmovw   %r15d, %k2;\n"      // mask

        "vexpandps (%rcx,%r12,4), %zmm1{%k1}{z}; \n"  // values
        "vmovdqu64 (%r8,%r13,4), %zmm2; \n"           // x

        "popcntw %r14w, %r14w;\n"
        "addq %r14,%r12;\n"

        "vexpandps (%rcx,%r12,4), %zmm6{%k2}{z}; \n"  // values
        "popcntw %r15w, %r15w;\n"
        "addq %r15,%r12;\n"

        #ifndef NO_FMADD
        "vfmadd231ps	%zmm2, %zmm1, %zmm0;\n"
        "vfmadd231ps	%zmm2, %zmm6, %zmm3;\n"
        #else
        "vmulps %zmm2, %zmm1,  %zmm1;\n"
        "vaddps %zmm1, %zmm0, %zmm0; \n"  //values
        "vmulps %zmm2, %zmm6,  %zmm6;\n"
        "vaddps %zmm6, %zmm3, %zmm3; \n"  //values
        #endif

        "addq	$8, %rdx; \n" // headers += 1 * c + 2 * i => 1 + 8 but rounded to 12

        "subq $1,%r11; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]-1...
    "jnz compute_2r2_Spmv512_avx_asm_float_inrow_exp; \n"
    "compute_2r2_Spmv512_avx_asm_float_inrow_exp_stop:\n"

    "vmovaps %ymm0, %ymm1;\n"
    "vextractf64x4   $0x1, %zmm0, %ymm0;\n"
    "vaddps  %ymm0, %ymm1, %ymm0;\n"

    "vextractf128    $0x1, %ymm0, %xmm1;\n"
    "vaddps  %xmm1, %xmm0, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  (%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, (%r9,%r10,4);\n"

    "vmovaps %ymm3, %ymm1;\n"
    "vextractf64x4   $0x1, %zmm3, %ymm3;\n"
    "vaddps  %ymm3, %ymm1, %ymm3;\n"

    "vextractf128    $0x1, %ymm3, %xmm1;\n"
    "vaddps  %xmm1, %xmm3, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  4(%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, 4(%r9,%r10,4);\n"

    "compute_2r2_Spmv512_avx_asm_float_outrow_exp:\n"

    "addq $2, %r10;\n"   // idxRow += 1

    "cmp %rdi, %r10;\n" // idxRow == nbRows
    "jl compute_2r2_Spmv512_avx_asm_float_loop_exp;\n" // if nbBlocks != 0 go to beginning

    "NOP;\n" // To avoid the presence of ret just after the jne

"compute_2r2_Spmv512_avx_asm_float_out_exp:\n"

"popq    %rbx;\n"
"popq    %rax;\n"
"popq    %r10;\n"
"popq    %r11;\n"
"popq    %r12;\n"
"popq    %r13;\n"
"popq    %r14;\n"
"popq    %r15;\n"

"ret;\n"
);

extern "C" void core_SPC5_4rV2c_Spmv_asm_double(const long int nbRows, const int* rowsSizes,
                                   const unsigned char* headers,
                                                   const double* values,
                                                   const double* x, double* y);

/* (nbRows rdi , rowsSizes rsi , headers rdx , values rcx, x r8, y r9 ) */
/* r10. must be saved r11/r12/r13/r14/r15 */
__asm__(
".global core_SPC5_4rV2c_Spmv_asm_double \n"
"core_SPC5_4rV2c_Spmv_asm_double: \n"

"pushq   %r15;\n"
"pushq   %r14;\n"
"pushq   %r13;\n"
"pushq   %r12;\n"
"pushq   %r11;\n"
"pushq   %r10;\n"
"pushq   %rax;\n"
"pushq   %rbx;\n"

"xorq %r12,%r12; \n"    // valIdx

// if no rows, jump to end
"test %rdi, %rdi;\n"
"jz   compute_4r_Spmv512_avx_asm_double_out_exp;\n"
    "xorq %r10, %r10; \n" // r10 = 0 idxRow
    "mov %rcx,%rbx;\n"

    "compute_4r_Spmv512_avx_asm_double_loop_exp:\n"

    "movslq 4(%rsi,%r10,4),%r11; \n"  // rowsSizes[idxRow+1]
    "subl 0(%rsi,%r10,4),%r11d; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]
    // if no blocks for this row, jumpb
    "jz compute_4r_Spmv512_avx_asm_double_outrow_exp; \n"

    "vpxor %ymm0,%ymm0,%ymm0; \n" // %zmm0 sum
    "vpxor %ymm3,%ymm3,%ymm3; \n" // %zmm0 sum

    "vpxor %ymm5,%ymm5,%ymm5; \n" // %zmm0 sum
    "vpxor %ymm7,%ymm7,%ymm7; \n" // %zmm0 sum
/*#ifndef NO1TEST
    "vpxorq %zmm10,%zmm10,%zmm10; \n" // %zmm0 sum

    "compute_4r_Spmv512_avx_asm_double_inrow_exp_nomorethan1_loop:"
        "movzwl  4(%rdx), %ecx;\n"     // mask
        "popcntl %ecx,%eax;\n"
        "cmp $1,%eax;\n"
        "jne compute_4r_Spmv512_avx_asm_double_inrow_exp_morethan1;\n"
        "compute_4r_Spmv512_avx_asm_double_inrow_exp_nomorethan1:\n"

        "movslq 0(%rdx),%r13; \n"    // colIdx

        "vmovsd (%rbx,%r12,8), %xmm8; \n"  // values
        "vmulsd (%r8,%r13,8), %xmm8,  %xmm8;\n"

        "bsfl %ecx,%ecx;\n"
        "shrl $2,%ecx;\n" // /4
        "movl $1,%eax;\n"
        "shll %cl,%eax;\n"
        "vbroadcastsd %xmm8, %ymm8;\n"
        "kmovw   %eax, %k1;\n"
#ifdef USE_KNL
        "vaddpd %zmm8,%zmm10,%zmm10{%k1};\n"
#else
        "vaddpd %ymm8,%ymm10,%ymm10{%k1};\n"
#endif

        "addq $1,%r12;\n"

        "addq	$6, %rdx; \n" // headers += 1 * c + 2 * i => 1 + 8 but rounded to 12

        "subq $1,%r11; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]-1...
        "jnz compute_4r_Spmv512_avx_asm_double_inrow_exp_nomorethan1_loop; \n"
        "jmp compute_4r_Spmv512_avx_asm_double_inrow_exp_stop;\n"
#endif*/
    "compute_4r_Spmv512_avx_asm_double_inrow_exp:"
/*#ifndef NO1TEST
        "movzwl  4(%rdx), %ecx;\n"     // mask
        "popcntl %ecx,%eax;\n"
        "cmp $1,%eax;\n"
        "je compute_4r_Spmv512_avx_asm_double_inrow_exp_nomorethan1;\n"
        "compute_4r_Spmv512_avx_asm_double_inrow_exp_morethan1:"
#endif*/
        "movslq 0(%rdx),%r13; \n"    // colIdx
        "movzbl  4(%rdx), %r14d;\n"  // mask
        "movzbl  5(%rdx), %r15d;\n"  // mask

        "kmovw   %r14d, %k1;\n"      // mask
        "kmovw   %r15d, %k2;\n"      // mask

        "vexpandpd (%rbx,%r12,8), %zmm1{%k1}{z}; \n"  // values
        "vmovupd (%r8,%r13,8), %ymm2; \n"           // x
        "vextractf64x4   $0x1, %zmm1, %ymm4;\n"

        "popcntw %r14w, %r14w;\n"
        "addq %r14,%r12;\n"

        "vexpandpd (%rbx,%r12,8), %zmm6{%k2}{z}; \n"  // values
        "popcntw %r15w, %r15w;\n"
        "addq %r15,%r12;\n"
        "vextractf64x4   $0x1, %zmm6, %ymm8;\n"


        #ifndef NO_FMADD
        "vfmadd231pd	%ymm2, %ymm1, %ymm0;\n"
        "vfmadd231pd	%ymm2, %ymm4, %ymm3;\n"
        "vfmadd231pd	%ymm2, %ymm6, %ymm5;\n"
        "vfmadd231pd	%ymm2, %ymm8, %ymm7;\n"
        #else
        "vmulpd %ymm2, %ymm1,  %ymm1;\n"
        "vaddpd %ymm1, %ymm0, %ymm0; \n"  //values
        "vmulpd %ymm2, %ymm4,  %ymm4;\n"
        "vaddpd %ymm4, %ymm3, %ymm3; \n"  //values
        "vmulpd %ymm2, %ymm6,  %ymm6;\n"
        "vaddpd %ymm6, %ymm5, %ymm5; \n"  //values
        "vmulpd %ymm2, %ymm8,  %ymm8;\n"
        "vaddpd %ymm8, %ymm7, %ymm7; \n"  //values
        #endif

        "addq	$6, %rdx; \n" // headers += 1 * c + 2 * i => 1 + 8 but rounded to 12

        "subq $1,%r11; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]-1...
    "jnz compute_4r_Spmv512_avx_asm_double_inrow_exp; \n"
    "compute_4r_Spmv512_avx_asm_double_inrow_exp_stop:\n"

    "vhaddpd %ymm5, %ymm0, %ymm0;\n"
    "vhaddpd %ymm7, %ymm3, %ymm3;\n"

    "vpermpd $216, %ymm0, %ymm0;\n"
    "vpermpd $216, %ymm3, %ymm3;\n"

    "vhaddpd %ymm3, %ymm0, %ymm0;\n"

/*#ifndef NO1TEST
    "vaddpd  %ymm10, %ymm0, %ymm0;\n"
#endif*/

    "vaddpd  (%r9,%r10,8), %ymm0, %ymm0;\n" // Y[idxRow] += sum
    "vmovupd  %ymm0, (%r9,%r10,8);\n"

    "compute_4r_Spmv512_avx_asm_double_outrow_exp:\n"

    "addq $4, %r10;\n"   // idxRow += 4

    "cmp %rdi, %r10;\n" // idxRow == nbRows
    "jl compute_4r_Spmv512_avx_asm_double_loop_exp;\n" // if nbBlocks != 0 go to beginning

    "NOP;\n" // To avoid the presence of ret just after the jne

"compute_4r_Spmv512_avx_asm_double_out_exp:\n"

"popq    %rbx;\n"
"popq    %rax;\n"
"popq    %r10;\n"
"popq    %r11;\n"
"popq    %r12;\n"
"popq    %r13;\n"
"popq    %r14;\n"
"popq    %r15;\n"

"ret;\n"
);

extern "C" void core_SPC5_4rV2c_Spmv_asm_float(const long int nbRows, const int* rowsSizes,
                                   const unsigned char* headers,
                                                   const float* values,
                                                   const float* x, float* y);

/* (nbRows rdi , rowsSizes rsi , headers rdx , values rcx, x r8, y r9 ) */
/* r10. must be saved r11/r12/r13/r14/r15 */
__asm__(
".global core_SPC5_4rV2c_Spmv_asm_float \n"
"core_SPC5_4rV2c_Spmv_asm_float: \n"

"pushq   %r15;\n"
"pushq   %r14;\n"
"pushq   %r13;\n"
"pushq   %r12;\n"
"pushq   %r11;\n"
"pushq   %r10;\n"
"pushq   %rax;\n"
"pushq   %rbx;\n"

"xorq %r12,%r12; \n"    // valIdx

// if no rows, jump to end
"test %rdi, %rdi;\n"
"jz   compute_4r_Spmv512_avx_asm_float_out_exp;\n"
    "xorq %r10, %r10; \n" // r10 = 0 idxRow

    "compute_4r_Spmv512_avx_asm_float_loop_exp:\n"

    "movslq 4(%rsi,%r10,4),%r11; \n"  // rowsSizes[idxRow+1]
    "subl 0(%rsi,%r10,4),%r11d; \n"   // rowsSizes[idxRow+1]-rowsSizes[idxRow]
    // if no blocks for this row, jumpb
    "jz compute_4r_Spmv512_avx_asm_float_outrow_exp; \n"

    "vpxor %ymm0,%ymm0,%ymm0; \n" // %zmm0 sum
    "vpxor %ymm3,%ymm3,%ymm3; \n" // %zmm0 sum

    "vpxor %ymm5,%ymm5,%ymm5; \n" // %zmm0 sum
    "vpxor %ymm7,%ymm7,%ymm7; \n" // %zmm0 sum

    "compute_4r_Spmv512_avx_asm_float_inrow_exp:"
        "movslq 0(%rdx),%r13; \n"    // colIdx
        "movzwl  4(%rdx), %r14d;\n"  // mask
        "kmovw   %r14d, %k1;\n"      // mask

        "movzwl  6(%rdx), %r15d;\n"  // mask
        "kmovw   %r15d, %k2;\n"      // mask

        "vexpandps (%rcx,%r12,4), %zmm1{%k1}{z}; \n"  // values
        "vmovups (%r8,%r13,4), %ymm2; \n"           // x
        "vextractf64x4   $0x1, %zmm1, %ymm4;\n"

        "popcntw %r14w, %r14w;\n"
        "addq %r14,%r12;\n"

        "vexpandps (%rcx,%r12,4), %zmm6{%k2}{z}; \n"  // values
        "popcntw %r15w, %r15w;\n"
        "addq %r15,%r12;\n"
        "vextractf64x4   $0x1, %zmm6, %ymm8;\n"

        #ifndef NO_FMADD
        "vfmadd231ps	%ymm2, %ymm1, %ymm0;\n"
        "vfmadd231ps	%ymm2, %ymm4, %ymm3;\n"
        "vfmadd231ps	%ymm2, %ymm6, %ymm5;\n"
        "vfmadd231ps	%ymm2, %ymm8, %ymm7;\n"
        #else
        "vmulps %ymm2, %ymm1,  %ymm1;\n"
        "vaddps %ymm1, %ymm0, %ymm0; \n"  //values
        "vmulps %ymm2, %ymm4,  %ymm4;\n"
        "vaddps %ymm4, %ymm3, %ymm3; \n"  //values
        "vmulps %ymm2, %ymm6,  %ymm6;\n"
        "vaddps %ymm6, %ymm5, %ymm5; \n"  //values
        "vmulps %ymm2, %ymm8,  %ymm8;\n"
        "vaddps %ymm8, %ymm7, %ymm7; \n"  //values
        #endif


        "addq	$8, %rdx; \n" // headers += 1 * c + 2 * i => 1 + 8 but rounded to 12

        "subq $1,%r11; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]-1...
    "jnz compute_4r_Spmv512_avx_asm_float_inrow_exp; \n"
    "compute_4r_Spmv512_avx_asm_float_inrow_exp_stop:\n"

    "vextractf128    $0x1, %ymm0, %xmm1;\n"
    "vaddps  %xmm1, %xmm0, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  (%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, (%r9,%r10,4);\n"

    "vextractf128    $0x1, %ymm3, %xmm1;\n"
    "vaddps  %xmm1, %xmm3, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  4(%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, 4(%r9,%r10,4);\n"

    "vextractf128    $0x1, %ymm5, %xmm1;\n"
    "vaddps  %xmm1, %xmm5, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  8(%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, 8(%r9,%r10,4);\n"

    "vextractf128    $0x1, %ymm7, %xmm1;\n"
    "vaddps  %xmm1, %xmm7, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  12(%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, 12(%r9,%r10,4);\n"

    "compute_4r_Spmv512_avx_asm_float_outrow_exp:\n"

    "addq $4, %r10;\n"   // idxRow += 1

    "cmp %rdi, %r10;\n" // idxRow == nbRows
    "jl compute_4r_Spmv512_avx_asm_float_loop_exp;\n" // if nbBlocks != 0 go to beginning

    "NOP;\n" // To avoid the presence of ret just after the jne

"compute_4r_Spmv512_avx_asm_float_out_exp:\n"

"popq    %rbx;\n"
"popq    %rax;\n"
"popq    %r10;\n"
"popq    %r11;\n"
"popq    %r12;\n"
"popq    %r13;\n"
"popq    %r14;\n"
"popq    %r15;\n"

"ret;\n"
);


//////////////////////////////////////////////////////////////////////////

extern "C" void core_SPC5_4rVc_Spmv_asm_double(const long int nbRows, const int* rowsSizes,
                                   const unsigned char* headers,
                                                   const double* values,
                                                   const double* x, double* y);

/* (nbRows rdi , rowsSizes rsi , headers rdx , values rcx, x r8, y r9 ) */
/* r10. must be saved r11/r12/r13/r14/r15 */
__asm__(
".global core_SPC5_4rVc_Spmv_asm_double \n"
"core_SPC5_4rVc_Spmv_asm_double: \n"

"pushq   %r15;\n"
"pushq   %r14;\n"
"pushq   %r13;\n"
"pushq   %r12;\n"
"pushq   %r11;\n"
"pushq   %r10;\n"
"pushq   %rax;\n"
"pushq   %rbx;\n"

"xorq %r12,%r12; \n"    // valIdx

// if no rows, jump to end
"test %rdi, %rdi;\n"
"jz   compute_4r2_Spmv512_avx_asm_double_out_exp;\n"
    "xorq %r10, %r10; \n" // r10 = 0 idxRow
    "mov %rcx,%rbx;\n"

    "compute_4r2_Spmv512_avx_asm_double_loop_exp:\n"

    "movslq 4(%rsi,%r10,4),%r11; \n"  // rowsSizes[idxRow+1]
    "subl 0(%rsi,%r10,4),%r11d; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]
    // if no blocks for this row, jumpb
    "jz compute_4r2_Spmv512_avx_asm_double_outrow_exp; \n"

    "vpxorq %zmm0,%zmm0,%zmm0; \n" // %zmm0 sum
    "vpxorq %zmm3,%zmm3,%zmm3; \n" // %zmm0 sum

    "vpxorq %zmm5,%zmm5,%zmm5; \n" // %zmm0 sum
    "vpxorq %zmm7,%zmm7,%zmm7; \n" // %zmm0 sum

/*#ifndef NO1TEST
    "vpxorq %zmm10,%zmm10,%zmm10; \n" // %zmm0 sum

    "compute_4r2_Spmv512_avx_asm_double_inrow_exp_nomorethan1_loop:"
        "movl  4(%rdx), %ecx;\n"     // mask
        "popcnt %ecx,%eax;\n"
        "cmp $1,%eax;\n"
        "jne compute_4r2_Spmv512_avx_asm_double_inrow_exp_morethan1;\n"
        "compute_4r2_Spmv512_avx_asm_double_inrow_exp_nomorethan1:"

        "movslq 0(%rdx),%r13; \n"    // colIdx

        "vmovsd (%rbx,%r12,8), %xmm8; \n"  // values
        "vmulsd (%r8,%r13,8), %xmm8,  %xmm8;\n"

        "bsf %ecx,%ecx;\n"
        "shrl $3,%ecx;\n" // /8
        "movl $1,%eax;\n"
        "shll %cl,%eax;\n"
        "vbroadcastsd %xmm8, %ymm8;\n"
        "kmovw   %eax, %k1;\n"
#ifdef USE_KNL
        "vaddpd %zmm8,%zmm10,%zmm10{%k1};\n"
#else
        "vaddpd %ymm8,%ymm10,%ymm10{%k1};\n"
#endif

        "addq $1,%r12;\n"

        "addq	$8, %rdx; \n" // headers += 1 * c + 2 * i => 1 + 8 but rounded to 12

        "subq $1,%r11; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]-1...
        "jnz compute_4r2_Spmv512_avx_asm_double_inrow_exp_nomorethan1_loop; \n"
        "jmp compute_4r2_Spmv512_avx_asm_double_inrow_exp_stop;\n"
#endif*/

    "compute_4r2_Spmv512_avx_asm_double_inrow_exp:"
/*#ifndef NO1TEST
        "movl  4(%rdx), %ecx;\n"     // mask
        "popcnt %ecx,%eax;\n"
        "cmp $1,%rax;\n"
        "je compute_4r2_Spmv512_avx_asm_double_inrow_exp_nomorethan1;\n"
        "compute_4r2_Spmv512_avx_asm_double_inrow_exp_morethan1:"
#endif*/
        "movslq 0(%rdx),%r13; \n"    // colIdx
        "movzbl  4(%rdx), %r14d;\n"  // mask
        "movzbl  5(%rdx), %r15d;\n"  // mask

        "kmovw   %r14d, %k1;\n"      // mask
        "kmovw   %r15d, %k2;\n"      // mask

        "vexpandpd (%rbx,%r12,8), %zmm1{%k1}{z}; \n"  // values
        "vmovdqu64 (%r8,%r13,8), %zmm2; \n"           // x

        "popcntw %r14w, %r14w;\n"
        "addq %r14,%r12;\n"

        "vexpandpd (%rbx,%r12,8), %zmm4{%k2}{z}; \n"  // values
        "popcntw %r15w, %r15w;\n"
        "addq %r15,%r12;\n"


        #ifndef NO_FMADD
        "vfmadd231pd	%zmm2, %zmm1, %zmm0;\n"
        "vfmadd231pd	%zmm2, %zmm4, %zmm3;\n"
        #else
        "vmulpd %zmm2, %zmm1,  %zmm1;\n"
        "vaddpd %zmm1, %zmm0, %zmm0; \n"  //values
        "vmulpd %zmm2, %zmm4,  %zmm4;\n"
        "vaddpd %zmm4, %zmm3, %zmm3; \n"  //values
        #endif

        "movzbl  6(%rdx), %r14d;\n"  // mask
        "movzbl  7(%rdx), %r15d;\n"  // mask

        "kmovw   %r14d, %k1;\n"      // mask
        "kmovw   %r15d, %k2;\n"      // mask

        "vexpandpd (%rbx,%r12,8), %zmm6{%k1}{z}; \n"  // values

        "popcntw %r14w, %r14w;\n"
        "addq %r14,%r12;\n"

        "vexpandpd (%rbx,%r12,8), %zmm8{%k2}{z}; \n"  // values
        "popcntw %r15w, %r15w;\n"
        "addq %r15,%r12;\n"


        #ifndef NO_FMADD
        "vfmadd231pd	%zmm2, %zmm6, %zmm5;\n"
        "vfmadd231pd	%zmm2, %zmm8, %zmm7;\n"
        #else
        "vmulpd %zmm2, %zmm6,  %zmm6;\n"
        "vaddpd %zmm6, %zmm5, %zmm5; \n"  //values
        "vmulpd %zmm2, %zmm8,  %zmm8;\n"
        "vaddpd %zmm8, %zmm7, %zmm7; \n"  //values
        #endif

        "addq	$8, %rdx; \n" // headers += 1 * c + 2 * i => 1 + 8 but rounded to 12

        "subq $1,%r11; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]-1...
    "jnz compute_4r2_Spmv512_avx_asm_double_inrow_exp; \n"
    "compute_4r2_Spmv512_avx_asm_double_inrow_exp_stop:\n"

    "vextractf64x4   $0x1, %zmm0, %ymm1;\n"
    "vextractf64x4   $0x1, %zmm3, %ymm4;\n"
    "vextractf64x4   $0x1, %zmm5, %ymm6;\n"
    "vextractf64x4   $0x1, %zmm7, %ymm8;\n"

    "vaddpd  %ymm1, %ymm0, %ymm0;\n"
    "vaddpd  %ymm4, %ymm3, %ymm3;\n"
    "vaddpd  %ymm6, %ymm5, %ymm5;\n"
    "vaddpd  %ymm8, %ymm7, %ymm7;\n"

    "vhaddpd %ymm5, %ymm0, %ymm0;\n"
    "vhaddpd %ymm7, %ymm3, %ymm3;\n"

    "vpermpd $216, %ymm0, %ymm0;\n"
    "vpermpd $216, %ymm3, %ymm3;\n"

    "vhaddpd %ymm3, %ymm0, %ymm0;\n"

/*#ifndef NO1TEST
    "vaddpd  %ymm10, %ymm0, %ymm0;\n"
#endif*/

    "vaddpd  (%r9,%r10,8), %ymm0, %ymm0;\n" // Y[idxRow] += sum
    "vmovupd  %ymm0, (%r9,%r10,8);\n"

    "compute_4r2_Spmv512_avx_asm_double_outrow_exp:\n"

    "addq $4, %r10;\n"   // idxRow += 4

    "cmp %rdi, %r10;\n" // idxRow == nbRows
    "jl compute_4r2_Spmv512_avx_asm_double_loop_exp;\n" // if nbBlocks != 0 go to beginning

    "NOP;\n" // To avoid the presence of ret just after the jne

"compute_4r2_Spmv512_avx_asm_double_out_exp:\n"

"popq    %rbx;\n"
"popq    %rax;\n"
"popq    %r10;\n"
"popq    %r11;\n"
"popq    %r12;\n"
"popq    %r13;\n"
"popq    %r14;\n"
"popq    %r15;\n"

"ret;\n"
);

extern "C" void core_SPC5_4rVc_Spmv_asm_float(const long int nbRows, const int* rowsSizes,
                                   const unsigned char* headers,
                                                   const float* values,
                                                   const float* x, float* y);

/* (nbRows rdi , rowsSizes rsi , headers rdx , values rcx, x r8, y r9 ) */
/* r10. must be saved r11/r12/r13/r14/r15 */
__asm__(
".global core_SPC5_4rVc_Spmv_asm_float \n"
"core_SPC5_4rVc_Spmv_asm_float: \n"

"pushq   %r15;\n"
"pushq   %r14;\n"
"pushq   %r13;\n"
"pushq   %r12;\n"
"pushq   %r11;\n"
"pushq   %r10;\n"
"pushq   %rax;\n"
"pushq   %rbx;\n"

"xorq %r12,%r12; \n"    // valIdx

// if no rows, jump to end
"test %rdi, %rdi;\n"
"jz   compute_4r2_Spmv512_avx_asm_float_out_exp;\n"
    "xorq %r10, %r10; \n" // r10 = 0 idxRow

    "compute_4r2_Spmv512_avx_asm_float_loop_exp:\n"

    "movslq 4(%rsi,%r10,4),%r11; \n"  // rowsSizes[idxRow+1]
    "subl 0(%rsi,%r10,4),%r11d; \n"   // rowsSizes[idxRow+1]-rowsSizes[idxRow]
    // if no blocks for this row, jumpb
    "jz compute_4r2_Spmv512_avx_asm_float_outrow_exp; \n"

    "vpxorq %zmm0,%zmm0,%zmm0; \n" // %zmm0 sum
    "vpxorq %zmm3,%zmm3,%zmm3; \n" // %zmm0 sum

    "vpxorq %zmm5,%zmm5,%zmm5; \n" // %zmm0 sum
    "vpxorq %zmm7,%zmm7,%zmm7; \n" // %zmm0 sum

    "compute_4r2_Spmv512_avx_asm_float_inrow_exp:"
        "movslq 0(%rdx),%r13; \n"    // colIdx
        "movzwl  4(%rdx), %r14d;\n"  // mask
        "kmovw   %r14d, %k1;\n"      // mask

        "movzwl  6(%rdx), %r15d;\n"  // mask
        "kmovw   %r15d, %k2;\n"      // mask

        "vexpandps (%rcx,%r12,4), %zmm1{%k1}{z}; \n"  // values
        "vmovdqu64 (%r8,%r13,4), %zmm2; \n"           // x

        "popcntw %r14w, %r14w;\n"
        "addq %r14,%r12;\n"

        "vexpandps (%rcx,%r12,4), %zmm6{%k2}{z}; \n"  // values
        "popcntw %r15w, %r15w;\n"
        "addq %r15,%r12;\n"

        #ifndef NO_FMADD
        "vfmadd231ps	%zmm2, %zmm1, %zmm0;\n"
        "vfmadd231ps	%zmm2, %zmm6, %zmm3;\n"
        #else
        "vmulps %zmm2, %zmm1,  %zmm1;\n"
        "vaddps %zmm1, %zmm0, %zmm0; \n"  //values
        "vmulps %zmm2, %zmm6,  %zmm6;\n"
        "vaddps %zmm6, %zmm3, %zmm3; \n"  //values
        #endif

        "movzwl  8(%rdx), %r14d;\n"  // mask
        "kmovw   %r14d, %k1;\n"      // mask

        "movzwl  10(%rdx), %r15d;\n"  // mask
        "kmovw   %r15d, %k2;\n"      // mask

        "vexpandps (%rcx,%r12,4), %zmm1{%k1}{z}; \n"  // values
        "vmovdqu64 (%r8,%r13,4), %zmm2; \n"           // x

        "popcntw %r14w, %r14w;\n"
        "addq %r14,%r12;\n"

        "vexpandps (%rcx,%r12,4), %zmm6{%k2}{z}; \n"  // values
        "popcntw %r15w, %r15w;\n"
        "addq %r15,%r12;\n"

        #ifndef NO_FMADD
        "vfmadd231ps	%zmm2, %zmm1, %zmm5;\n"
        "vfmadd231ps	%zmm2, %zmm6, %zmm7;\n"
        #else
        "vmulps %zmm2, %zmm1,  %zmm1;\n"
        "vaddps %zmm1, %zmm5, %zmm5; \n"  //values
        "vmulps %zmm2, %zmm6,  %zmm6;\n"
        "vaddps %zmm6, %zmm7, %zmm7; \n"  //values
        #endif

        "addq	$12, %rdx; \n" // headers += 1 * c + 2 * i => 1 + 8 but rounded to 12

        "subq $1,%r11; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]-1...
    "jnz compute_4r2_Spmv512_avx_asm_float_inrow_exp; \n"
    "compute_4r2_Spmv512_avx_asm_float_inrow_exp_stop:\n"

    "vmovaps %ymm0, %ymm1;\n"
    "vextractf64x4   $0x1, %zmm0, %ymm0;\n"
    "vaddps  %ymm0, %ymm1, %ymm0;\n"

    "vextractf128    $0x1, %ymm0, %xmm1;\n"
    "vaddps  %xmm1, %xmm0, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  (%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, (%r9,%r10,4);\n"

    "vmovaps %ymm3, %ymm1;\n"
    "vextractf64x4   $0x1, %zmm3, %ymm3;\n"
    "vaddps  %ymm3, %ymm1, %ymm3;\n"

    "vextractf128    $0x1, %ymm3, %xmm1;\n"
    "vaddps  %xmm1, %xmm3, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  4(%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, 4(%r9,%r10,4);\n"

    "vmovaps %ymm5, %ymm1;\n"
    "vextractf64x4   $0x1, %zmm5, %ymm5;\n"
    "vaddps  %ymm5, %ymm1, %ymm5;\n"

    "vextractf128    $0x1, %ymm5, %xmm1;\n"
    "vaddps  %xmm1, %xmm5, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  8(%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, 8(%r9,%r10,4);\n"

    "vmovaps %ymm7, %ymm1;\n"
    "vextractf64x4   $0x1, %zmm7, %ymm7;\n"
    "vaddps  %ymm7, %ymm1, %ymm7;\n"

    "vextractf128    $0x1, %ymm7, %xmm1;\n"
    "vaddps  %xmm1, %xmm7, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  12(%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, 12(%r9,%r10,4);\n"

    "compute_4r2_Spmv512_avx_asm_float_outrow_exp:\n"

    "addq $4, %r10;\n"   // idxRow += 1

    "cmp %rdi, %r10;\n" // idxRow == nbRows
    "jl compute_4r2_Spmv512_avx_asm_float_loop_exp;\n" // if nbBlocks != 0 go to beginning

    "NOP;\n" // To avoid the presence of ret just after the jne

"compute_4r2_Spmv512_avx_asm_float_out_exp:\n"

"popq    %rbx;\n"
"popq    %rax;\n"
"popq    %r10;\n"
"popq    %r11;\n"
"popq    %r12;\n"
"popq    %r13;\n"
"popq    %r14;\n"
"popq    %r15;\n"

"ret;\n"
);

//////////////////////////////////////////////////////////////////////////

extern "C" void core_SPC5_8rV2c_Spmv_asm_double(const long int nbRows, const int* rowsSizes,
                                   const unsigned char* headers,
                                                   const double* values,
                                                   const double* x, double* y);

/* (nbRows rdi , rowsSizes rsi , headers rdx , values rcx, x r8, y r9 ) */
/* r10. must be saved r11/r12/r13/r14/r15 */
__asm__(
".global core_SPC5_8rV2c_Spmv_asm_double \n"
"core_SPC5_8rV2c_Spmv_asm_double: \n"

"pushq   %r15;\n"
"pushq   %r14;\n"
"pushq   %r13;\n"
"pushq   %r12;\n"
"pushq   %r11;\n"
"pushq   %r10;\n"
"pushq   %rax;\n"
"pushq   %rbx;\n"

"xorq %r12,%r12; \n"    // valIdx

// if no rows, jump to end
"test %rdi, %rdi;\n"
"jz   compute_8r_Spmv512_avx_asm_double_out_exp;\n"
    "xorq %r10, %r10; \n" // r10 = 0 idxRow
    "mov %rcx,%rbx;\n"

    "compute_8r_Spmv512_avx_asm_double_loop_exp:\n"

    "movslq 4(%rsi,%r10,4),%r11; \n"  // rowsSizes[idxRow+1]
    "subl 0(%rsi,%r10,4),%r11d; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]
    // if no blocks for this row, jumpb
    "jz compute_8r_Spmv512_avx_asm_double_outrow_exp; \n"

    "vpxor %ymm0,%ymm0,%ymm0; \n" // %zmm0 sum
    "vpxor %ymm3,%ymm3,%ymm3; \n" // %zmm0 sum

    "vpxor %ymm5,%ymm5,%ymm5; \n" // %zmm0 sum
    "vpxor %ymm7,%ymm7,%ymm7; \n" // %zmm0 sum

    "vpxor %ymm9,%ymm9,%ymm9; \n" // %zmm0 sum
    "vpxor %ymm11,%ymm11,%ymm11; \n" // %zmm0 sum

    "vpxor %ymm13,%ymm13,%ymm13; \n" // %zmm0 sum
    "vpxor %ymm15,%ymm15,%ymm15; \n" // %zmm0 sum
/*#ifndef NO1TEST
    "vpxorq %zmm10,%zmm10,%zmm10; \n" // %zmm0 sum

    "compute_8r_Spmv512_avx_asm_double_inrow_exp_nomorethan1_loop:"
        "mov  4(%rdx), %rcx;\n"     // mask
        "popcnt %rcx,%rax;\n"
        "cmp $1,%eax;\n"
        "jne compute_8r_Spmv512_avx_asm_double_inrow_exp_morethan1;\n"
        "compute_8r_Spmv512_avx_asm_double_inrow_exp_nomorethan1:\n"

        "movslq 0(%rdx),%r13; \n"    // colIdx

        "vmovsd (%rbx,%r12,8), %xmm8; \n"  // values
        "vmulsd (%r8,%r13,8), %xmm8,  %xmm8;\n"

        "bsf %rcx,%rcx;\n"
        "shrl $2,%ecx;\n" // /4
        "movl $1,%eax;\n"
        "shll %cl,%eax;\n"
        "vbroadcastsd %xmm8, %zmm8;\n"
        "kmovw   %eax, %k1;\n"
        "vaddpd %zmm8,%zmm10,%zmm10{%k1};\n"

        "addq $1,%r12;\n"

        "addq	$8, %rdx; \n" // headers += 1 * c + 2 * i => 1 + 8 but rounded to 12

        "subq $1,%r11; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]-1...
        "jnz compute_8r_Spmv512_avx_asm_double_inrow_exp_nomorethan1_loop; \n"
        "jmp compute_8r_Spmv512_avx_asm_double_inrow_exp_stop;\n"
#endif*/
    "compute_8r_Spmv512_avx_asm_double_inrow_exp:"
/*#ifndef NO1TEST
        "mov  4(%rdx), %rcx;\n"     // mask
        "popcnt %rcx,%rax;\n"
        "cmp $1,%eax;\n"
        "je compute_8r_Spmv512_avx_asm_double_inrow_exp_nomorethan1;\n"
        "compute_8r_Spmv512_avx_asm_double_inrow_exp_morethan1:\n"
#endif*/
        "movslq 0(%rdx),%r13; \n"    // colIdx
        "movzbl  4(%rdx), %r14d;\n"  // mask
        "kmovw   %r14d, %k1;\n"      // mask

        "movzbl  5(%rdx), %r15d;\n"  // mask
        "kmovw   %r15d, %k2;\n"      // mask

        "vexpandpd (%rbx,%r12,8), %zmm1{%k1}{z}; \n"  // values
        "vmovupd (%r8,%r13,8), %ymm2; \n"           // x
        "vextractf64x4   $0x1, %zmm1, %ymm4;\n"

        "popcntw %r14w, %r14w;\n"
        "addq %r14,%r12;\n"

        "vexpandpd (%rbx,%r12,8), %zmm6{%k2}{z}; \n"  // values
        "popcntw %r15w, %r15w;\n"
        "addq %r15,%r12;\n"
        "vextractf64x4   $0x1, %zmm6, %ymm8;\n"


        #ifndef NO_FMADD
        "vfmadd231pd	%ymm2, %ymm1, %ymm0;\n"
        "vfmadd231pd	%ymm2, %ymm4, %ymm3;\n"
        "vfmadd231pd	%ymm2, %ymm6, %ymm5;\n"
        "vfmadd231pd	%ymm2, %ymm8, %ymm7;\n"
        #else
        "vmulpd %ymm2, %ymm1,  %ymm1;\n"
        "vaddpd %ymm1, %ymm0, %ymm0; \n"  //values
        "vmulpd %ymm2, %ymm4,  %ymm4;\n"
        "vaddpd %ymm4, %ymm3, %ymm3; \n"  //values
        "vmulpd %ymm2, %ymm6,  %ymm6;\n"
        "vaddpd %ymm6, %ymm5, %ymm5; \n"  //values
        "vmulpd %ymm2, %ymm8,  %ymm8;\n"
        "vaddpd %ymm8, %ymm7, %ymm7; \n"  //values
        #endif

        "movzbl  6(%rdx), %r14d;\n"  // mask
        "movzbl  7(%rdx), %r15d;\n"  // mask

        "kmovw   %r14d, %k1;\n"      // mask
        "kmovw   %r15d, %k2;\n"      // mask

        "vexpandpd (%rbx,%r12,8), %zmm1{%k1}{z}; \n"  // values
        "vextractf64x4   $0x1, %zmm1, %ymm4;\n"

        "popcntw %r14w, %r14w;\n"
        "addq %r14,%r12;\n"

        "vexpandpd (%rbx,%r12,8), %zmm6{%k2}{z}; \n"  // values
        "popcntw %r15w, %r15w;\n"
        "addq %r15,%r12;\n"
        "vextractf64x4   $0x1, %zmm6, %ymm8;\n"

        #ifndef NO_FMADD
        "vfmadd231pd	%ymm2, %ymm1, %ymm9;\n"
        "vfmadd231pd	%ymm2, %ymm4, %ymm11;\n"
        "vfmadd231pd	%ymm2, %ymm6, %ymm13;\n"
        "vfmadd231pd	%ymm2, %ymm8, %ymm15;\n"
        #else
        "vmulpd %ymm2, %ymm1,  %ymm1;\n"
        "vaddpd %ymm1, %ymm9, %ymm9; \n"  //values
        "vmulpd %ymm2, %ymm4,  %ymm4;\n"
        "vaddpd %ymm4, %ymm11, %ymm11; \n"  //values
        "vmulpd %ymm2, %ymm6,  %ymm6;\n"
        "vaddpd %ymm6, %ymm13, %ymm13; \n"  //values
        "vmulpd %ymm2, %ymm8,  %ymm8;\n"
        "vaddpd %ymm8, %ymm15, %ymm15; \n"  //values
        #endif

        "addq	$8, %rdx; \n" // headers += 1 * c + 2 * i => 1 + 8 but rounded to 12

        "subq $1,%r11; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]-1...
    "jnz compute_8r_Spmv512_avx_asm_double_inrow_exp; \n"
    "compute_8r_Spmv512_avx_asm_double_inrow_exp_stop:\n"

    "vextractf64x4   $0x1, %zmm0, %ymm1;\n"
    "vextractf64x4   $0x1, %zmm3, %ymm4;\n"
    "vextractf64x4   $0x1, %zmm5, %ymm6;\n"
    "vextractf64x4   $0x1, %zmm7, %ymm8;\n"

    "vaddpd  %ymm1, %ymm0, %ymm0;\n"
    "vaddpd  %ymm4, %ymm3, %ymm3;\n"
    "vaddpd  %ymm6, %ymm5, %ymm5;\n"
    "vaddpd  %ymm8, %ymm7, %ymm7;\n"

    "vhaddpd %ymm5, %ymm0, %ymm0;\n"
    "vhaddpd %ymm7, %ymm3, %ymm3;\n"

    "vpermpd $216, %ymm0, %ymm0;\n"
    "vpermpd $216, %ymm3, %ymm3;\n"

    "vhaddpd %ymm3, %ymm0, %ymm0;\n"

/*#ifndef NO1TEST
    "vaddpd  %ymm10, %ymm0, %ymm0;\n"
#endif*/

    "vaddpd  (%r9,%r10,8), %ymm0, %ymm0;\n" // Y[idxRow] += sum
    "vmovupd  %ymm0, (%r9,%r10,8);\n"

    "vextractf64x4   $0x1, %zmm9, %ymm2;\n"
    "vextractf64x4   $0x1, %zmm11, %ymm12;\n"
    "vextractf64x4   $0x1, %zmm13, %ymm14;\n"
    "vextractf64x4   $0x1, %zmm15, %ymm6;\n"

    "vaddpd  %ymm2, %ymm9, %ymm9;\n"
    "vaddpd  %ymm12, %ymm11, %ymm11;\n"
    "vaddpd  %ymm14, %ymm13, %ymm13;\n"
    "vaddpd  %ymm6, %ymm15, %ymm15;\n"

    "vhaddpd %ymm13, %ymm9, %ymm0;\n"
    "vhaddpd %ymm15, %ymm11, %ymm3;\n"

    "vpermpd $216, %ymm0, %ymm0;\n"
    "vpermpd $216, %ymm3, %ymm3;\n"

    "vhaddpd %ymm3, %ymm0, %ymm0;\n"

/*#ifndef NO1TEST
    "vextractf64x4   $0x1, %zmm10, %ymm10;\n"
    "vaddpd  %ymm10, %ymm0, %ymm0;\n"
#endif*/

    "vaddpd  32(%r9,%r10,8), %ymm0, %ymm0;\n" // Y[idxRow] += sum
    "vmovupd  %ymm0, 32(%r9,%r10,8);\n"

    "compute_8r_Spmv512_avx_asm_double_outrow_exp:\n"

    "addq $8, %r10;\n"   // idxRow += 8

    "cmp %rdi, %r10;\n" // idxRow == nbRows
    "jl compute_8r_Spmv512_avx_asm_double_loop_exp;\n" // if nbBlocks != 0 go to beginning

    "NOP;\n" // To avoid the presence of ret just after the jne

"compute_8r_Spmv512_avx_asm_double_out_exp:\n"

"popq    %rbx;\n"
"popq    %rax;\n"
"popq    %r10;\n"
"popq    %r11;\n"
"popq    %r12;\n"
"popq    %r13;\n"
"popq    %r14;\n"
"popq    %r15;\n"

"ret;\n"
);

extern "C" void core_SPC5_8rV2c_Spmv_asm_float(const long int nbRows, const int* rowsSizes,
                                   const unsigned char* headers,
                                                   const float* values,
                                                   const float* x, float* y);

/* (nbRows rdi , rowsSizes rsi , headers rdx , values rcx, x r8, y r9 ) */
/* r10. must be saved r11/r12/r13/r14/r15 */
__asm__(
".global core_SPC5_8rV2c_Spmv_asm_float \n"
"core_SPC5_8rV2c_Spmv_asm_float: \n"

"pushq   %r15;\n"
"pushq   %r14;\n"
"pushq   %r13;\n"
"pushq   %r12;\n"
"pushq   %r11;\n"
"pushq   %r10;\n"
"pushq   %rax;\n"
"pushq   %rbx;\n"

"xorq %r12,%r12; \n"    // valIdx

// if no rows, jump to end
"test %rdi, %rdi;\n"
"jz   compute_8r_Spmv512_avx_asm_float_out_exp;\n"
    "xorq %r10, %r10; \n" // r10 = 0 idxRow

    "compute_8r_Spmv512_avx_asm_float_loop_exp:\n"

    "movslq 4(%rsi,%r10,4),%r11; \n"  // rowsSizes[idxRow+1]
    "subl 0(%rsi,%r10,4),%r11d; \n"   // rowsSizes[idxRow+1]-rowsSizes[idxRow]
    // if no blocks for this row, jumpb
    "jz compute_8r_Spmv512_avx_asm_float_outrow_exp; \n"

    "vpxor %ymm0,%ymm0,%ymm0; \n" // %zmm0 sum
    "vpxor %ymm3,%ymm3,%ymm3; \n" // %zmm0 sum

    "vpxor %ymm5,%ymm5,%ymm5; \n" // %zmm0 sum
    "vpxor %ymm7,%ymm7,%ymm7; \n" // %zmm0 sum

    "vpxor %ymm9,%ymm9,%ymm9; \n" // %zmm0 sum
    "vpxor %ymm11,%ymm11,%ymm11; \n" // %zmm0 sum

    "vpxor %ymm13,%ymm13,%ymm13; \n" // %zmm0 sum
    "vpxor %ymm15,%ymm15,%ymm15; \n" // %zmm0 sum

    "compute_8r_Spmv512_avx_asm_float_inrow_exp:"
        "movslq 0(%rdx),%r13; \n"    // colIdx
        "movzwl  4(%rdx), %r14d;\n"  // mask
        "kmovw   %r14d, %k1;\n"      // mask

        "movzwl  6(%rdx), %r15d;\n"  // mask
        "kmovw   %r15d, %k2;\n"      // mask

        "vexpandps (%rcx,%r12,4), %zmm1{%k1}{z}; \n"  // values
        "vmovups (%r8,%r13,4), %ymm2; \n"           // x
        "vextractf64x4   $0x1, %zmm1, %ymm4;\n"

        "popcntw %r14w, %r14w;\n"
        "addq %r14,%r12;\n"

        "vexpandps (%rcx,%r12,4), %zmm6{%k2}{z}; \n"  // values
        "popcntw %r15w, %r15w;\n"
        "addq %r15,%r12;\n"
        "vextractf64x4   $0x1, %zmm6, %ymm8;\n"

        #ifndef NO_FMADD
        "vfmadd231ps	%ymm2, %ymm1, %ymm0;\n"
        "vfmadd231ps	%ymm2, %ymm4, %ymm3;\n"
        "vfmadd231ps	%ymm2, %ymm6, %ymm5;\n"
        "vfmadd231ps	%ymm2, %ymm8, %ymm7;\n"
        #else
        "vmulps %ymm2, %ymm1,  %ymm1;\n"
        "vaddps %ymm1, %ymm0, %ymm0; \n"  //values
        "vmulps %ymm2, %ymm4,  %ymm4;\n"
        "vaddps %ymm4, %ymm3, %ymm3; \n"  //values
        "vmulps %ymm2, %ymm6,  %ymm6;\n"
        "vaddps %ymm6, %ymm5, %ymm5; \n"  //values
        "vmulps %ymm2, %ymm8,  %ymm8;\n"
        "vaddps %ymm8, %ymm7, %ymm7; \n"  //values
        #endif

        "movzwl  8(%rdx), %r14d;\n"  // mask
        "kmovw   %r14d, %k1;\n"      // mask

        "movzwl  10(%rdx), %r15d;\n"  // mask
        "kmovw   %r15d, %k2;\n"      // mask

        "vexpandps (%rcx,%r12,4), %zmm1{%k1}{z}; \n"  // values
        "vextractf64x4   $0x1, %zmm1, %ymm4;\n"

        "popcntw %r14w, %r14w;\n"
        "addq %r14,%r12;\n"

        "vexpandps (%rcx,%r12,4), %zmm6{%k2}{z}; \n"  // values
        "popcntw %r15w, %r15w;\n"
        "addq %r15,%r12;\n"
        "vextractf64x4   $0x1, %zmm6, %ymm8;\n"

        #ifndef NO_FMADD
        "vfmadd231ps	%ymm2, %ymm1, %ymm9;\n"
        "vfmadd231ps	%ymm2, %ymm4, %ymm11;\n"
        "vfmadd231ps	%ymm2, %ymm6, %ymm13;\n"
        "vfmadd231ps	%ymm2, %ymm8, %ymm15;\n"
        #else
        "vmulps %ymm2, %ymm1,  %ymm1;\n"
        "vaddps %ymm1, %ymm9, %ymm9; \n"  //values
        "vmulps %ymm2, %ymm4,  %ymm4;\n"
        "vaddps %ymm4, %ymm11, %ymm11; \n"  //values
        "vmulps %ymm2, %ymm6,  %ymm6;\n"
        "vaddps %ymm6, %ymm13, %ymm13; \n"  //values
        "vmulps %ymm2, %ymm8,  %ymm8;\n"
        "vaddps %ymm8, %ymm15, %ymm15; \n"  //values
        #endif

        "addq	$12, %rdx; \n" // headers += 1 * c + 2 * i => 1 + 8 but rounded to 12

        "subq $1,%r11; \n"    // rowsSizes[idxRow+1]-rowsSizes[idxRow]-1...
    "jnz compute_8r_Spmv512_avx_asm_float_inrow_exp; \n"
    "compute_8r_Spmv512_avx_asm_float_inrow_exp_stop:\n"

    "vextractf128    $0x1, %ymm0, %xmm1;\n"
    "vaddps  %xmm1, %xmm0, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  (%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, (%r9,%r10,4);\n"

    "vextractf128    $0x1, %ymm3, %xmm1;\n"
    "vaddps  %xmm1, %xmm3, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  4(%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, 4(%r9,%r10,4);\n"

    "vextractf128    $0x1, %ymm5, %xmm1;\n"
    "vaddps  %xmm1, %xmm5, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  8(%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, 8(%r9,%r10,4);\n"

    "vextractf128    $0x1, %ymm7, %xmm1;\n"
    "vaddps  %xmm1, %xmm7, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  12(%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, 12(%r9,%r10,4);\n"

    "vextractf128    $0x1, %ymm9, %xmm1;\n"
    "vaddps  %xmm1, %xmm9, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  16(%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, 16(%r9,%r10,4);\n"

    "vextractf128    $0x1, %ymm11, %xmm1;\n"
    "vaddps  %xmm1, %xmm11, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  20(%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, 20(%r9,%r10,4);\n"

    "vextractf128    $0x1, %ymm13, %xmm1;\n"
    "vaddps  %xmm1, %xmm13, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  24(%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, 24(%r9,%r10,4);\n"

    "vextractf128    $0x1, %ymm15, %xmm1;\n"
    "vaddps  %xmm1, %xmm15, %xmm1;\n"

    "vpermilps       $27, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm1;\n"
    "vpermilps       $177, %xmm1, %xmm0;\n"
    "vaddps  %xmm0, %xmm1, %xmm0;\n"
    "vaddss  28(%r9,%r10,4), %xmm0, %xmm0;\n"
    "vmovss  %xmm0, 28(%r9,%r10,4);\n"

    "compute_8r_Spmv512_avx_asm_float_outrow_exp:\n"

    "addq $8, %r10;\n"   // idxRow += 1

    "cmp %rdi, %r10;\n" // idxRow == nbRows
    "jl compute_8r_Spmv512_avx_asm_float_loop_exp;\n" // if nbBlocks != 0 go to beginning

    "NOP;\n" // To avoid the presence of ret just after the jne

"compute_8r_Spmv512_avx_asm_float_out_exp:\n"

"popq    %rbx;\n"
"popq    %rax;\n"
"popq    %r10;\n"
"popq    %r11;\n"
"popq    %r12;\n"
"popq    %r13;\n"
"popq    %r14;\n"
"popq    %r15;\n"

"ret;\n"
);
