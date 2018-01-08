# SPC5

## Overview

This package contains function to compute the SpMV on AVX-512-based architectures (KNL, SKL, CNL).
It comes with different storages and kernels each being more or less adapted to the matrices shapes.

We refer to the related paper *Computing the Sparse Matrix Vector Product using Block-Based Kernels Without Zero Padding on Processors with AVX-512 Instructions* (currently under review but available on https://arxiv.org/abs/1801.01134 ), and we will appreciate that any use or reference to our work points to this paper for citation.

SPC5 is under MIT licence.

The library does not provide a strong abstraction, but it uses a compact code that is easy to modify or copy/paste to your project.

## Compilation

### Using SCP5 in another project

Simply add the content of the src directory to your own project.
You will also need the flag for C++11 and AVX-512.
You can avoid the usage of FMADD (and replace each of them by mul/add) by passing `-DNO_FADD`.
You can ensure to reduce NUMA effects by using the macro `SPLIT_NUMA` (you have to bind the threads to ensure correct allocation placement) by passing `-DSPLIT_NUMA`.

### Compiling the examples

Create and build directory and run our cmake file:
```bash
mkdir build
cd build
cmake ..
```

MKL is not mandatory but if the environment variable `MKLROOT` is defined then MKL will be used in the examples.


### Using SCP5

We strongly suggest users to have a look to the examples.
We provide the same features for each of our matrix formats, but also generic methods that manage the selection of the appropriate function.


