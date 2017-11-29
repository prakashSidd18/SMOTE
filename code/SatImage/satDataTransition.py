f1=open("sat.trn","r")
f2=open("sat.tst","r")

out=open("sat.csv","w")


for i in f1.readlines():
  line=i.rstrip().split(" ")
  if line[-1]=='4':
    line[-1]='1'
  else:
    line[-1]='0'
  out.write(",".join(line)+"\n")

for i in f2.readlines():
  line=i.rstrip().split(" ")
  if line[-1]=='4':
    line[-1]='1'
  else:
    line[-1]='0'
  out.write(",".join(line)+"\n")
f1.close()
f2.close()
out.close()

f=open("sat.csv","r")
gb=f.readlines()
classList2=[x.rstrip().split(",")[-1] for x in gb]
from collections import Counter
classCount=dict(Counter(classList2))
print "minorityClass="+ str(classCount['1'])
print "majorityClass="+ str(classCount['0'])

