# SPC5

The official repository is https://gitlab.inria.fr/bramas/spc5

The original work (only for X86 AVX512 in assembly) is available at the same address https://gitlab.inria.fr/bramas/spc5 but on branch avx512

## Overview

This package contains functions to compute the SpMV on ARM-SVE-based architectures (a64fx) and X86 AVX-512 architectures.
It comes with different storages and kernels each being more or less adapted to the matrices shapes.

We refer to the related paper *Computing the Sparse Matrix Vector Product using Block-Based Kernels Without Zero Padding on Processors with AVX-512 Instructions* (published in PeerJ CS https://peerj.com/articles/cs-151/ ), and we will appreciate that any use or reference to our work points to this paper for citation.

The SVE work has been submitted and a preprint is availabe at TODO.

## Licence

SPC5 is under MIT licence.

The library does not provide a strong abstraction, but it uses a compact code that is easy to modify or copy/paste to your project.

## Compilation

If you compile on ARM architecture, SVE will be enabled.
If you compile on x86 architecture, SVE using emulation (FARM SVE) will be enabled.
Therefore, one has to explicitely activate AVX512 option to disable SVE and use AVX512.

Here is the list of CMake flags which are also macro that should defined/undefined to select the proper implementation:
- USE_AVX512: Off by default, turning it on will disable sve and enable avx-512

options related to AVX512:
- MKLROOT: is an env variable to select the location where mkl is installed
- USE_MKL: to use mkl in the comparison, is on if MKLROOT exist, off otherwise

If USE_AVX512 is off (default) sve is used:
- USE_FARM: on if USE_AVX512 is off and not on a arm CPU
- FARM_NB_BITS_IN_VEC=512 used if USE_FARM is on
- ARMPLROOT: is an environement variable that will be used to find the location of the ARM PL (no tested)
- USE_ARMPL: to use ARM PL, is on if ARMPLROOT is defined (no tested)
- FACTOLOAD: off by default, will activate the loading of x enterely (see related paper)

Both SVE and AVX512 has this option:
- MHSUM: off by default, will activate the manual multi-reduction (see related paper)

CMake will also look at OpenMP to enable parallel kernels.
The code is protected with the standard _OPENMP macro.

### Using SPC5 in another project

Simply add the spc5.hpp file in your project.
You will also need the flag for C++17 and SVE or AVX512.
You can also use the following macro: USE_AVX512, USE_FARM, FARM_NB_BITS_IN_VEC, FACTOLOAD, MHSUM.

### Compiling the examples

Create and build directory and run our cmake file:
```bash
mkdir build
cd build
# possible cmake commands:
## SVE
CXX=g++ cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_AVX512=OFF -DUSE_MKL=OFF -DMHSUM=OFF -DFACTOLOAD=OFF

CXX=g++ cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_AVX512=OFF -DUSE_MKL=OFF -DMHSUM=ON -DFACTOLOAD=OFF

CXX=g++ cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_AVX512=OFF -DUSE_MKL=OFF -DMHSUM=OFF -DFACTOLOAD=ON

CXX=g++ cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_AVX512=OFF -DUSE_MKL=OFF -DMHSUM=ON -DFACTOLOAD=ON
## AVX-512
CXX=g++ cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_AVX512=ON -DCPU=CNL -DUSE_MKL=ON -DMHSUM=OFF

CXX=g++ cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_AVX512=ON -DCPU=CNL -DUSE_MKL=ON -DMHSUM=ON

# Then build
make
```

Arm Performance Libraries is not mandatory but if the environment variable `ARMPLROOT` is defined then armpl will be used in the examples.
Same for MKL.


### Using SPC5

We strongly suggest users to have a look to the examples.

### Other files
- jobs-afx64.sh : batch job for the paper
- jobs-plafrim.sh : batch job for the paper

- results/gentex.py  : convert csv to figures for the paper
- results/plot.py  : useless file
- results/results-arm-7_stats.csv  : stats of the matrices used in the paper
- results/results-arm-8  : all the output files for the paper & arm sve
- results/results-arm-8.csv  : results for the paper & arm sve
- results/results-avx-2  : all output files for the paper & avx512
- results/results-avx-2.csv  : results for the paper & avx512
- results/script_csv.sh : generate a csv from res files
- results/script-stats.sh : generate a stats csv from res files

