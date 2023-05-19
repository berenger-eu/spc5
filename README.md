# SPC5-ARM-SVE

The official repository is https://gitlab.inria.fr/bramas/spc5-arm-sve
It is similar to the X86 AVX512 version available at https://gitlab.inria.fr/bramas/spc5

## Overview

This package contains function to compute the SpMV on ARM-SVE-based architectures (a64fx).
It comes with different storages and kernels each being more or less adapted to the matrices shapes.

We refer to the related paper *Computing the Sparse Matrix Vector Product using Block-Based Kernels Without Zero Padding on Processors with AVX-512 Instructions* (published in PeerJ CS https://peerj.com/articles/cs-151/ ), and we will appreciate that any use or reference to our work points to this paper for citation.

SPC5-ARM-SVE is under MIT licence.

The library does not provide a strong abstraction, but it uses a compact code that is easy to modify or copy/paste to your project.

## Compilation

### Using SPC5-ARM-SVE on x86

For testing and debugging purposes it is possible to run the lib on X86 cpu,
in this case the farm sve emulation library will be used instead of real SIMD SVE instructions.

### Using SPC5-ARM-SVE in another project

Simply add the content of the src directory to your own project.
You will also need the flag for C++11 and SVE.
You can ensure to reduce NUMA effects by using the macro `SPLIT_NUMA` (you have to bind the threads to ensure correct allocation placement) by passing `-DSPLIT_NUMA`.

### Compiling the examples

Create and build directory and run our cmake file:
```bash
mkdir build
cd build
cmake ..
```

Arm Performance Libraries is not mandatory but if the environment variable `ARMPLROOT` is defined then armpl will be used in the examples.


### Using SPC5-ARM-SVE

We strongly suggest users to have a look to the examples.
We provide the same features for each of our matrix formats, but also generic methods that manage the selection of the appropriate function.


