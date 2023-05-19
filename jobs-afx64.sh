#!/bin/bash
#PBS -q a64fx
#PBS -l select=1:ncpus=48,place=scatter
#PBS -l walltime=24:00:00

cd /home/ri-bbramas/spc5-arm-sve/build/

module load gcc/10.3.0

make clean

# Gen double version
CXX=g++ cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSEFLOAT=OFF -DUSEDENSE=OFF
make

cp ./load_mm_and_compare ./load_mm_and_compare-double

# Gen float version
CXX=g++ cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSEFLOAT=ON -DUSEDENSE=OFF
make

cp ./load_mm_and_compare ./load_mm_and_compare-float

# Iterate over the matrices

urls=(
    "https://suitesparse-collection-website.herokuapp.com/MM/Fluorem/HV15R.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/POLYFLOW/mixtank_new.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/Si41Ge41H72.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/vanHeukelum/cage15.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/LAW/in-2004.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/ND/nd6k.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/Si87H76.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Freescale/circuit5M.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/LAW/indochina-2004.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/FEMLAB/ns3Da.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/CO.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn21.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Williams/pdb1HYS.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Norris/torso1.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/crankseg_2.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/ldoor.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Boeing/pwtk.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Mazaheri/bundle_adj.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Janna/Cube_Coup_dt0.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Dziekonski/dielFilterV2real.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Janna/Emilia_923.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Freescale/FullChip.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Janna/Hook_1498.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Fluorem/RM07R.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Janna/Serena.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Mittelmann/spal_004.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/TSOPF/TSOPF_RS_b2383_c1.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Gleich/wikipedia-20060925.tar.gz"
)

for url in "${urls[@]}"; do
    echo " ============================================== "
    echo " Work on $url"
    # Download the file
    filename=$(basename "$url" | cut -f 1 -d '.')
    # Check if the file already exists
    if [ -e "$filename" ]; then
        echo "File $filename already exists. Skipping download."
        mtx_file=$(find "$filename" -type f -name '*.mtx' -print -quit)
    else
        wget "$url"

        # Extract the tar.gz file
        tar -xzf "$filename"

        # Get the name of the extracted file
        extracted_files=$(tar -tf "$filename")
        mtx_file=$(echo "$extracted_files" | grep -m 1 '\.mtx$')
    fi
        
    if [[ -n "$mtx_file" ]]; then
        echo "Compute : $mtx_file"
        ./load_mm_and_compare-double "$mtx_file" > res_"$filename"_double.txt
        ./load_mm_and_compare-float "$mtx_file" > res_"$filename"_float.txt
    else
        echo "No .mtx file found in $filename"
    fi
done


#################################

# Dense
# Gen double version
CXX=g++ cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSEFLOAT=OFF -DUSEDENSE=ON
make

cp ./load_mm_and_compare ./load_mm_and_compare-double

# Gen float version
CXX=g++ cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSEFLOAT=ON -DUSEDENSE=ON
make

cp ./load_mm_and_compare ./load_mm_and_compare-float

./load_mm_and_compare-double > res_dense_double.txt
./load_mm_and_compare-float > res_dense_float.txt
