#!/bin/bash
#PBS -q a64fx
#PBS -l select=1:ncpus=48,place=scatter
#PBS -l walltime=24:00:00
#PBS -l naccesspolicy=SINGLEJOB -n

set -x

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

urls_normal=(
    "https://suitesparse-collection-website.herokuapp.com/MM/POLYFLOW/mixtank_new.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/Si41Ge41H72.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/LAW/in-2004.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/ND/nd6k.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/Si87H76.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/FEMLAB/ns3Da.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/CO.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Williams/pdb1HYS.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Norris/torso1.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/crankseg_2.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/ldoor.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Boeing/pwtk.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Mazaheri/bundle_adj.tar.gz"
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

urls_big=(
    "https://suitesparse-collection-website.herokuapp.com/MM/Freescale/circuit5M.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/vanHeukelum/cage15.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn21.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/LAW/indochina-2004.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Fluorem/HV15R.tar.gz"
)

urls=("${urls_normal[@]}")
# urls=("${urls_big[@]}")

working_dir="./matrices/"

for url in "${urls[@]}"; do
    echo " ============================================== "
    echo " Work on $url"
    # Download the file
    filename=$(basename "$url" | cut -f 1 -d '.')
    
    if [[ -n "res_"$filename"_float.txt" ]]; then
        echo "Do it $filename"
        # Check if the file already exists
        if [ -e "$working_dir/$filename" ]; then
            echo "File $working_dir/$filename already exists. Skipping download."
            mtx_file=$(find "$working_dir/$filename" -type f -name '*.mtx' -print -quit)
        else
            wget "$url"

            # Extract the tar.gz file
            tar -xzf "$filename.tar.gz" -C "$working_dir/"

            # Get the name of the extracted file
            extracted_files=$(tar -tf "$filename.tar.gz")
            mtx_file=$(echo "$working_dir/$extracted_files" | grep -m 1 '\.mtx$')
            rm "$filename.tar.gz"
        fi
            
        if [[ -n "$mtx_file" ]]; then
            echo "Compute : $mtx_file"
            taskset -c 0 ./load_mm_and_compare-double "$mtx_file" >> res_"$filename"_double.txt
            taskset -c 0 ./load_mm_and_compare-float "$mtx_file" >> res_"$filename"_float.txt
            rm -r "$working_dir/$filename"
        else
            echo "No .mtx file found in $filename"
        fi
    else
        echo "Already done, skip it $filename"
    fi
done


#################################

use_dense=true

if $use_dense ; then
    echo ==== Dense ===

    # Dense
    # Gen double version
    CXX=g++ cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSEFLOAT=OFF -DUSEDENSE=ON
    make

    cp ./load_mm_and_compare ./load_mm_and_compare-double

    # Gen float version
    CXX=g++ cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSEFLOAT=ON -DUSEDENSE=ON
    make

    cp ./load_mm_and_compare ./load_mm_and_compare-float

    taskset -c 0 ./load_mm_and_compare-double >> res_dense_double.txt
    taskset -c 0 ./load_mm_and_compare-float >> res_dense_float.txt
fi
