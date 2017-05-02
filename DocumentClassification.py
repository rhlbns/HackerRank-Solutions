from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import collections
import numpy as np

with open('trainingdata.txt', 'r') as f:
    N= f.readline()
    docs= f.readlines()

x_data, y_train= [],[]
for lines in docs:
    y_train.append(int(lines[0]))
    temp= lines.split(' ')
    x_data.append(temp[1:len(lines)-1])
    
laxicon= []
for x in x_data:
    for word in x:
        laxicon.append(word)

laxicon.sort()
count= collections.Counter(laxicon)
count= count.most_common(15000)

laxicon= []
for i,word in enumerate(count):
    laxicon.append(word[0])
laxicon= laxicon[0:3000]

x_train= []
for x in x_data:
    x.sort()
    count= collections.Counter(x)
    
    vec= np.zeros(len(laxicon), dtype= float)
    
    for i,words in enumerate(laxicon):
        if (words in count):
            vec[i]= 1
            
    x_train.append(vec)
  
    
clf= LogisticRegression()
clf.fit(x_train, y_train)

N= int(input())
doc_test= []
for _ in range(N):
    temp= input().split(' ')
    doc_test.append(temp)

x_test= []
for doc in doc_test:
    count= collections.Counter(doc)
    #print (count)
    temp= [0 for _ in range(len(laxicon))]
    for i,words in enumerate(laxicon):
        #print (words)
        if (words in count):
            temp[i]= 1
          
    x_test.append(temp)
    
#print (x_test)
y_predict= clf.predict(x_test)
for y in y_predict:
    print (y)
