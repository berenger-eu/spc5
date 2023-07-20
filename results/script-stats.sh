#!/bin/bash


echo "Proceed $1"
csvfile=$(echo $(basename $1)_stats.csv)

nb=4

echo "Name & Dim & NNZ & $\frac{NNZ}{N_{rows}}$ & $\beta(1,VS)$ & $\beta(2,VS)$ & $\beta(4,VS)$ & $\beta(8,VS)$  \\\\" > "$csvfile"
echo "\\hline \\hline" >> "$csvfile"

for fl in $1/res_*_nohsum_facto_float.txt ; do
    substring=${fl#*_}
    matrix_name=$(echo $(basename $substring) | cut -d'_' -f 1)
       
    dim=$(cat $fl | grep '> number of rows =' | cut -d'=' -f 2 | xargs)
    nbnnz=$(cat $fl | grep '> number of values =' | cut -d'=' -f 2 | xargs)
    nbnnzperrow=$(cat $fl | grep '> number of values per row =' | cut -d'=' -f 2 | xargs)
       
    echo "# fl $fl"
    echo " - matrix_name $matrix_name"
    echo " - dim $dim"
    echo " - nbnnz $nbnnz"
    echo " - nbnnzperrow $nbnnzperrow"

    pattern="-> Number of blocks ([0-9]+)\( avg. ([0-9]+(\.[0-9]+)?) values per block\)"

    csvline="$matrix_name & $dim & $nbnnz & $nbnnzperrow"

    floatval=(0 0 0 0)
    doubleval=(0 0 0 0)

    count=0

    while read -r line; do
      if [[ $line =~ $pattern ]]; then
        numberblocks=${BASH_REMATCH[1]}
        valuesperblock=${BASH_REMATCH[2]}
        echo " - numberblocks-$count $numberblocks"
        echo " - valuesperblock-$count $valuesperblock"
        floatval[$count]=$(echo "scale=0; 100 * $valuesperblock / ( 2^( $count ) * 16)" | bc)
        echo " - floatval[$count]-$count ${floatval[$count]}"
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
        fl="${fl//float/double}"
    
        count=0

        while read -r line; do
          if [[ $line =~ $pattern ]]; then
            numberblocks=${BASH_REMATCH[1]}
            valuesperblock=${BASH_REMATCH[2]}
            echo " - numberblocks-$count $numberblocks"
            echo " - valuesperblock-$count $valuesperblock"
            doubleval[$count]=$(echo "scale=0; 100 * $valuesperblock / ( 2^( $count ) * 8)" | bc)
            echo " - doubleval[$count]-$count ${doubleval[$count]}"
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
            for (( i = 0 ; i < $nb ; i++ )) ; do
                csvline="$csvline & ${doubleval[$i]}\% $|$ ${floatval[$i]}\% "
            done
            echo " - $csvline"
            echo "$csvline \\\\" >> "$csvfile"
        fi
    fi
done


