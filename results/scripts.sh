#!/bin/bash


echo "Proceed $1"
csvfile=$(echo $(basename $1).csv)

echo "matrixname,type,hsum,factox,scalar,1rVc,2rVc,4rVc,8rVc,1rVcpar,2rVcpar,4rVcpar,8rVcpar" > "$csvfile"
nb=9

for fl in $1/res_*.txt ; do
    substring=${fl#*_}
    matrix_name=$(echo $(basename $substring) | cut -d'_' -f 1)
    
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

    pattern="-> GFlops ([0-9]+(\.[0-9]+)?)s"

    csvline="$matrix_name,$float_type,$hsum,$factox"

    count=0

    while read -r line; do
      if [[ $line =~ $pattern ]]; then
        number=${BASH_REMATCH[1]}
        echo " - $number"
        csvline="$csvline,$number"
        count=$((count + 1))
      fi
      
      if ((count >= $nb)); then
        break
      fi
    done < "$fl"
    
    if ((count != $nb)); then
      echo "Error: Expected 4 matches, but found $count matches."
      rm $fl
    else
        echo " - $csvline"
        echo "$csvline" >> "$csvfile"
    fi
done


