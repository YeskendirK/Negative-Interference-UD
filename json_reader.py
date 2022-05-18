import os 
import json

#open a text file writer 
textf = open('output.txt', 'w')

#read json file from json_files directory 

json_dir = 'json_files\metavaljson'
data = {}
textf.write('Filename || Accuracy"\n')
for f1 in os.listdir(json_dir):
    #print(f1)
    d1 ={}
    f2 = os.path.join(json_dir, f1)
    with open(f2, 'r') as f:
        d1 = json.load(f)
        #print(data)
    f1 = f1.split('.json')[0]
    data[f1] = d1
    #writer 
    text1 = f1
    text2 = str(d1['LAS']['aligned_accuracy'])

    #write to text file 
    
    textf.write(text1 + '\t'+ '||' + text2 + '\n')

textf.close()
    