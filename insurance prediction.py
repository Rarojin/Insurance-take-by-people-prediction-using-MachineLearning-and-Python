import pandas as pd
data = pd.DataFrame({'age':[5,10,45,60,35,8,55,19,67,20],
                     'salary':[0,0,30000,55000,28000,23000,49000,8000,60000,12000],
                     'insurance':[0,0,1,1,1,0,1,0,1,0]})
from  sklearn.neighbors import KNeighborsClassifier
x=data.drop('insurance',axis=1)
y=data['insurance']
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(x,y)
new=pd.DataFrame({'age':[6,15,41,63,25],'salary':[0,0,29000,52000,72000]})
new['prediction']=knn.predict(new)
print(new)