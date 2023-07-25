#!/bin/bash


echo "Proceed $1"
csvfile=$(echo $(basename $1).csv)


if [[ $1 == *-avx* ]] ; then
    echo "matrixname,type,dim,nbnnz,nbnnzperrow,hsum,factox,scalar,MKL,1rVc,2rVc,4rVc,8rVc,1rVcpar,2rVcpar,4rVcpar,8rVcpar,valvec1vs,ratio1vs,valvec2vs,ratio2vs,valvec4vs,ratio4vs,valvec8vs,ratio8vs" > "$csvfile"
    nb=10
elif [[ $1 == *-arm* ]] ; then
    echo "matrixname,type,dim,nbnnz,nbnnzperrow,hsum,factox,scalar,1rVc,2rVc,4rVc,8rVc,1rVcpar,2rVcpar,4rVcpar,8rVcpar,valvec1vs,ratio1vs,valvec2vs,ratio2vs,valvec4vs,ratio4vs,valvec8vs,ratio8vs" > "$csvfile"
    nb=9
else
    echo "Error bad folder name"
fi

for fl in $1/res_*.txt ; do
    substring=${fl#*_}
    matrix_name=$(echo $(basename $substring) | cut -d'_' -f 1)
    dim=$(cat $fl | grep '> number of rows =' | cut -d'=' -f 2 | xargs)
    nbnnz=$(cat $fl | grep '> number of values =' | cut -d'=' -f 2 | xargs)
    nbnnzperrow=$(cat $fl | grep '> number of values per row =' | cut -d'=' -f 2 | xargs)
    
    if [[ $substring == *_float.txt ]] ; then
        float_type="float"
    else
        float_type="double"
    fi
    
    if [[ $substring == *_withhsum_* ]] ; then
        hsum="yes"
    else
        hsum="no"
    fi
    
    if [[ $substring == *_facto_* ]] ; then
        factox="yes"
    else
        factox="no"
    fi
    
    echo "# fl $fl"
    echo " - matrix_name $matrix_name"
    echo " - float_type $float_type"
    echo " - dim $dim"
    echo " - nbnnz $nbnnz"
    echo " - nbnnzperrow $nbnnzperrow"

    pattern="-> GFlops ([0-9]+(\.[0-9]+)?)s"

    csvline="$matrix_name,$float_type,$dim,$nbnnz,$nbnnzperrow,$hsum,$factox"

    count=0

    while read -r line; do
      if [[ $line =~ $pattern ]]; then
        number=${BASH_REMATCH[1]}
        echo " - $number"
        csvline="$csvline,$number"
        count=$((count + 1))
      fi
      
      if (( $count >= $nb )); then
        break
      fi
    done < "$fl"
    
    if (( $count != $nb )); then
      echo "Error: Expected $nb matches, but found $count matches."
      rm $fl
    else
        pattern="-> Number of blocks ([0-9]+)\( avg. ([0-9]+(\.[0-9]+)?) values per block\)"
        nbratio=4
        count=0
        while read -r line; do
            if [[ $line =~ $pattern ]]; then
                numberblocks=${BASH_REMATCH[1]}
                valuesperblock=${BASH_REMATCH[2]}
                echo " - numberblocks-$count $numberblocks"
                echo " - valuesperblock-$count $valuesperblock"
                if [[ $substring == *_float.txt ]] ; then
                    ratio=$(echo "scale=2; 100 * $valuesperblock / ( 2^( $count ) * 16)" | bc)
                else
                    ratio=$(echo "scale=2; 100 * $valuesperblock / ( 2^( $count ) * 8)" | bc)
                fi
                csvline="$csvline,$valuesperblock,$ratio"
                echo " - ratio $ratio"
                count=$((count + 1))
            fi

            if (( $count >= $nbratio )); then
                break
            fi
        done < "$fl"
        
        if (( $count != $nbratio )); then
          echo "Error: Expected $nbratio matches, but found $count matches."
          rm $fl
        else
            echo " - $csvline"
            echo "$csvline" >> "$csvfile"
        fi
    fi
done


