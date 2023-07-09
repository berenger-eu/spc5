import sys
import pandas as pd

# Check if the CSV file is provided as a command-line argument
if len(sys.argv) < 2:
    print("Please provide the path to the CSV file as a command-line argument.")
    sys.exit(1)

# Read the CSV file
filename = sys.argv[1]
df = pd.read_csv(filename)

# Get the number of rows and calculate the number of subplots needed
num_rows = len(df)

nb_mat_per_row=8

labels=['scalar', '1rVc', '2rVc', '4rVc', '8rVc']
res=['', '', '', '', '']
avgf64=[0,0,0,0,0]
avgf32=[0,0,0,0,0]

ticks=''

# Iterate over each row in the DataFrame
for i, row in df.iterrows():
    # Get the data for the current row
    matrixname = row['matrixname']
    matrixname=matrixname.replace('_', ' ')
    types = row['type']
    
    matrixnamewithtype = matrixname + ' (f64)' if types == 'double' else matrixname + ' (f32)'

    values = row[['scalar', '1rVc', '2rVc', '4rVc', '8rVc']].astype(float)
        
    # print(matrixname)
    # print(str(values))
    # print(str(values.index))
    
    ticks += matrixnamewithtype if ticks == '' else ',' + matrixnamewithtype
    
    for idx,lb in enumerate(labels):
        val=values[idx]
        if types == 'double':
            avgf64[idx] += val
        else:
            avgf32[idx] += val
        if idx == 0:
            res[idx] += f' ({matrixnamewithtype}, {val}) '
        else:
            speedup=values[idx]/values[0]
            res[idx]+= f' ({matrixnamewithtype}, {val})[{speedup:.1f}] '
            
    if (i+1)%nb_mat_per_row == 0:
        print('===================================')
        print(ticks)
        ticks=''
        for idx,lb in enumerate(labels):
            print('\\addplot[draw=color_' + str(idx) +'!65!black, fill=color_' + str(idx) +'!30!white] coordinates { ' + res[idx] + ' };')
            res[idx] = ''
     
#############################   
ticks+= ',average (f64), average (f32)'
for idx,lb in enumerate(labels):
    val0=avgf64[0]/(num_rows/2)
    val=avgf64[idx]/(num_rows/2)
    if idx == 0:
        res[idx] += f' (average (f64), {val}) '
    else:
        speedup=val/val0
        res[idx]+= f' (average (f64), {val})[{speedup:.1f}] '
for idx,lb in enumerate(labels):
    val0=avgf32[0]/(num_rows/2)
    val=avgf32[idx]/(num_rows/2)
    if idx == 0:
        res[idx] += f' (average (f32), {val}) '
    else:
        speedup=val/val0
        res[idx]+= f' (average (f32), {val})[{speedup:.1f}] '        

print('===================================')

print(ticks)
ticks=''
for idx,lb in enumerate(labels):
    print('\\addplot[draw=color_' + str(idx) +'!65!black, fill=color_' + str(idx) +'!30!white] coordinates { ' + res[idx] + ' };')
    res[idx] = ''
