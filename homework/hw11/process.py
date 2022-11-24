
not_used = [52,53,5469,70,71,72,73,74,75,76]
data = [
    [
        'id','ccf','age','sex','painloc','painexer','relrest','pncaden','cp','trestbps','htn','hol','smoke','cigs','years','fbs','dm','famhist','restecg','ekgmo','ekgday','ekgyr','dig','prop','nitr','pro','diuretic','proto','thaldur','thaltime','met','thalach','thalrest','tpeakbps','tpeakbpd','dummy','trestbpd','exang','xhypo','oldpeak','slope','rldv5','rldv5e','ca','restckm','exerckm','restef','restwm','exeref','exerwm','thal','thalsev','thalpul','earlobe','cmo','cday','cyr','num','lmt','ladprox','laddist','diag','cxmain','ramus','om1','om2','rcaprox','rcadist','lvx1','lvx2','lvx3','lvx4','lvf','cathef','junk','name'
    ]
]
for f in ['cleveland.data','hungarian.data', 'long-beach-va.data']:
    with open('cleveland.data', 'r') as f:
        raw = f.readlines()
        sample = []
        for line in raw:
            l = line.strip().split(' ')
            
            sample += l
            if l[-1] == 'name':
                if len(sample) != 76:
                    print('wrong size')
                else:
                    data.append(sample)
                
                sample = []

# print(data)
import csv
with open('heart_disease.csv', 'w') as f:
    w = csv.writer(f)
    w.writerows(data)