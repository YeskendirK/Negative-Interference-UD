import os 
import json

import re

#read json file from json_files directory 
#open a text file writer 
textf = open('output.txt', 'w')
json_dir = '/home/lcur1322/github/clone/3/Negative-Interference-UD/metavalidation_0.0001_0.0001_20_20_sgd_ONLY'
data = {}
textf.write('Filename || Accuracy"\n')
for f1 in os.listdir(json_dir):
    #print(f1)
    d1 ={}
    print(f1)
    if f1.endswith('.json'):
        lang_name = f1.split('-')[0]
        f2 = os.path.join(json_dir, f1)
        with open(f2, 'r',  encoding="utf8") as f:
            d1 = json.load(f)
            #print(data)
        f1 = f1.split('.json')[0]
        data[f1] = d1
        #writer 

        text1 = f1
        text2 = str(d1['LAS']['aligned_accuracy'])

        #write to text file 
        dig = re.findall(r'\d+', f1) 
        print(int(dig[0]))
        textf.write(text1 + '\t'+ '||' + text2 + '\n')
    

textf.close()
    