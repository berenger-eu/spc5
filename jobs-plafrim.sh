#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=spmv
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH -C "bora&intel&cascadelake"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 36


set -x

##############################################
cd /projets/schnaps/spc5-arm-sve/build/
module load build/cmake/3.15.3 compiler/gcc/11.2.0 linalg/mkl/2022.0.2
use_dense=true
remove_matrix=false
##############################################

make clean

CXX=g++ cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_AVX512=ON -DCPU=CNL -DUSE_MKL=ON -DMHSUM=OFF
make
exec_nohsum=./load_mm_and_compare_no_hsum
mv ./load_mm_and_compare $exec_nohsum

CXX=g++ cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_AVX512=ON -DCPU=CNL -DUSE_MKL=ON -DMHSUM=ON
make
exec_withhsum=./load_mm_and_compare_with_hsum
mv ./load_mm_and_compare $exec_withhsum

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
            $exec_withhsum --mx "$mtx_file" --real=double >> res_"$filename"_withhsum_double.txt
            $exec_withhsum --mx "$mtx_file" --real=float >> res_"$filename"_withhsum_float.txt
            $exec_nohsum --mx "$mtx_file" --real=double >> res_"$filename"_nohsum_double.txt
            $exec_nohsum --mx "$mtx_file" --real=float >> res_"$filename"_nohsum_float.txt
            if $remove_matrix ; then
                rm -r "$working_dir/$filename"
            fi
        else
            echo "No .mtx file found in $filename"
        fi
    else
        echo "Already done, skip it $filename"
    fi
done


#################################

if $use_dense ; then
    echo ==== Dense ===

    $exec_withhsum --dense=2048 --real=double >> res_dense_withhsum_double.txt
    $exec_withhsum --dense=2048 --real=float >> res_dense_withhsum_float.txt
    $exec_nohsum --dense=2048 --real=double >> res_dense_nohsum_double.txt
    $exec_nohsum --dense=2048 --real=float >> res_dense_nohsum_float.txt
fi
