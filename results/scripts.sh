#!/bin/bash


echo "Proceed $1"
csvfile=$(echo $(basename $1).csv)

use_v2=true

if $use_v2 ; then
    echo "matrixname,type,scalar,1rVc,1rVc_v2,2rVc,2rVc_v2,4rVc,4rVc_v2" > "$csvfile"
    nb=7
else
    echo "matrixname,type,scalar,1rVc,2rVc,4rVc" > "$csvfile"
    nb=4
fi

for fl in $1/res_*.txt ; do
    substring=${fl#*_}
    matrix_name=$(basename ${substring%_*})
    float_type=$(echo $substring | rev | cut -d'_' -f 1 | rev | cut -d'.' -f 1)
    
    echo "# fl $fl"
    echo " - matrix_name $matrix_name"
    echo " - float_type $float_type"

    pattern="-> GFlops ([0-9]+\.[0-9]+)s"

    csvline="$matrix_name,$float_type"

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


