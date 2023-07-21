import sys
import pandas as pd

# Check if the CSV file is provided as a command-line argument
if len(sys.argv) < 2:
    print("Please provide the path to the CSV file as a command-line argument.")
    sys.exit(1)

# Read the CSV file
filename = sys.argv[1]
df = pd.read_csv(filename)

hsumtypes=['yes','no']
factoxtypes=['yes','no']
nb_mat_per_row=12

allvalues = {}
allvaluespar = {}

if 'MKL' in df.columns:
    labels=['scalar', 'MKL', '1rVc', '2rVc', '4rVc', '8rVc']
    colors=['0', 'mkl', '1', '2', '3', '4']
    labelsparref=2
else:
    labels=['scalar', '1rVc', '2rVc', '4rVc', '8rVc']
    colors=['0', '1', '2', '3', '4']
    labelsparref=1

labelspar=['1rVcpar', '2rVcpar', '4rVcpar', '8rVcpar']
colorspar=['1', '2', '3', '4']

for factox in factoxtypes:
    for hsum in hsumtypes:
        print('% ====================================================')
        print('% ====================================================')
        print('% hsum = ' + hsum)
        print('% factox = ' + factox)        
        print('% ====================================================')
        print('% ====================================================')
        # Get the number of rows and calculate the number of subplots needed
        num_rows = 0


        res=['', '', '', '', '', '']
        avgf64=[0,0,0,0,0,0]
        avgf32=[0,0,0,0,0,0]
        avgf64par=[0,0,0,0,0,0]
        avgf32par=[0,0,0,0,0,0]

        ticks=''

        # Iterate over each row in the DataFrame
        for i, row in df.iterrows():
            if row['hsum'] == hsum and row['factox'] == factox:
                # print(str(i) + " " + str(row))
                # Get the data for the current row
                matrixname = row['matrixname']
                matrixname=matrixname.replace('_', ' ')
                types = row['type']
                
                matrixnamewithtype = matrixname + ' (f64)' if types == 'double' else matrixname + ' (f32)'

                values = row[labels].astype(float)
                valuespar = row[labelspar].astype(float)
                
                if not factox in allvalues:
                    allvalues[factox] = {}
                    allvaluespar[factox] = {}
                if not hsum in allvalues[factox]:
                    allvalues[factox][hsum] = {}
                    allvaluespar[factox][hsum] = {}
                if not matrixname in allvalues[factox][hsum]:
                    allvalues[factox][hsum][matrixname] = {}
                    allvaluespar[factox][hsum][matrixname] = {}
                allvalues[factox][hsum][matrixname][types] = values
                allvaluespar[factox][hsum][matrixname][types] = valuespar
                
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
                        
                for idx,lb in enumerate(labelspar):
                    val=values[idx]
                    if types == 'double':
                        avgf64par[idx] += valuespar[idx]
                    else:
                        avgf32par[idx] += valuespar[idx]
                        
                if (num_rows+1)%nb_mat_per_row == 0:
                    print('% ===================================')
                    print('% ' + ticks)
                    ticks=''
                    for idx,lb in enumerate(labels):
                        print('\\addplot[draw=color_' + colors[idx] +'!65!black, fill=color_' + colors[idx] +'!30!white] coordinates { ' + res[idx] + ' };')
                        res[idx] = ''
                num_rows = num_rows + 1

        if num_rows is not 0:
            print('% ===================================')
            print('% ' + ticks)
            ticks=''
            for idx,lb in enumerate(labels):
                print('\\addplot[draw=color_' + colors[idx] +'!65!black, fill=color_' + colors[idx] +'!30!white] coordinates { ' + res[idx] + ' };')
                res[idx] = ''

            for idx,lb in enumerate(labels):
                avgf64[idx] /= (num_rows)
                avgf32[idx] /= (num_rows)
                avgf64par[idx] /= (num_rows)
                avgf32par[idx] /= (num_rows)
            #############################   
            ticks+= ',average (f64), average (f32)'
            for idx,lb in enumerate(labels):
                val0=avgf64[0]
                val=avgf64[idx]
                if idx == 0:
                    res[idx] += f' (average (f64), {val}) '
                else:
                    speedup=val/val0
                    res[idx]+= f' (average (f64), {val})[{speedup:.1f}] '
            for idx,lb in enumerate(labels):
                val0=avgf32[0]
                val=avgf32[idx]
                if idx == 0:
                    res[idx] += f' (average (f32), {val}) '
                else:
                    speedup=val/val0
                    res[idx]+= f' (average (f32), {val})[{speedup:.1f}] '        

            print('% ===================================')

            print('% ' + ticks)
            ticks=''
            for idx,lb in enumerate(labels):
                print('\\addplot[draw=color_' + colors[idx] +'!65!black, fill=color_' + colors[idx] +'!30!white] coordinates { ' + res[idx] + ' };')
                res[idx] = ''
            
            
            if not factox in allvalues:
                allvalues[factox] = {}
                allvaluespar[factox] = {}
            if not hsum in allvalues[factox]:
                allvalues[factox][hsum] = {}
                allvaluespar[factox][hsum] = {}
            if not 'average' in allvalues[factox][hsum]:
                allvalues[factox][hsum]['average'] = {}
                allvaluespar[factox][hsum]['average'] = {}
            allvalues[factox][hsum]['average']['double'] = avgf64
            allvalues[factox][hsum]['average']['float'] = avgf32
            allvaluespar[factox][hsum]['average']['double'] = avgf64par
            allvaluespar[factox][hsum]['average']['float'] = avgf32par
        
##############################################################
print('#################################################')
for factox in factoxtypes:
    for hsum in hsumtypes:
        if factox in allvalues and hsum in allvalues[factox]:
            print(f'% factox {factox} hsum {hsum}')
            ticks=''    
            for idx,lb in enumerate(labels):
                res=''
                for idxmat,matrixname in enumerate(['CO','dense','nd6k','average']):
                    
                    val0=allvalues[factox][hsum][matrixname]['double'][0]
                    val=allvalues[factox][hsum][matrixname]['double'][idx]
                    if idx == 0:
                        res += f' ({matrixname} (f64), {val}) '
                    else:
                        speedup=val/val0
                        res+= f' ({matrixname} (f64), {val})[{speedup:.1f}] '
                    if idx == 0:
                        ticks += f'{matrixname} (f64)' if ticks == '' else f',{matrixname} (f64)'
                        
                    val0=allvalues[factox][hsum][matrixname]['float'][0]
                    val=allvalues[factox][hsum][matrixname]['float'][idx]
                    if idx == 0:
                        res += f' ({matrixname} (f32), {val}) '
                    else:
                        speedup=val/val0
                        res+= f' ({matrixname} (f32), {val})[{speedup:.1f}] ' 
                    if idx == 0:
                        ticks += f'{matrixname} (f32)' if ticks == '' else f',{matrixname} (f32)'
                
                print('\\addplot[draw=color_' + colors[idx] +'!65!black, fill=color_' + colors[idx] +'!30!white] coordinates { ' + res + ' };')
            
            print('% ' + ticks)
    
        
##############################################################
print('#################################################')
for factox in factoxtypes:
    for hsum in hsumtypes:
        if factox in allvaluespar and hsum in allvaluespar[factox]:
            print(f'% factox {factox} hsum {hsum}')
            ticks=''    
            for idx,lb in enumerate(labelspar):
                res=''
                for idxmat,matrixname in enumerate(['CO','dense','nd6k','average']):
                    
                    val0=allvalues[factox][hsum][matrixname]['double'][labelsparref+idx]
                    val=allvaluespar[factox][hsum][matrixname]['double'][idx]
                    if idx == None:
                        res += f' ({matrixname} (f64), {val}) '
                    else:
                        speedup=val/val0
                        res+= f' ({matrixname} (f64), {val})[{speedup:.1f}] '
                    if idx == 0:
                        ticks += f'{matrixname} (f64)' if ticks == '' else f',{matrixname} (f64)'
                        
                    val0=allvalues[factox][hsum][matrixname]['float'][labelsparref+idx]
                    val=allvaluespar[factox][hsum][matrixname]['float'][idx]
                    if idx == None:
                        res += f' ({matrixname} (f32), {val}) '
                    else:
                        speedup=val/val0
                        res+= f' ({matrixname} (f32), {val})[{speedup:.1f}] ' 
                    if idx == 0:
                        ticks += f'{matrixname} (f32)' if ticks == '' else f',{matrixname} (f32)'
                
                print('\\addplot[draw=color_' + colorspar[idx] +'!65!black, fill=color_' + colorspar[idx] +'!30!white] coordinates { ' + res + ' };')
            
            print('% ' + ticks)
    

