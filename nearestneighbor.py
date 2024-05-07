import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score, recall_score, f1_score
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib.lines import Line2D
import seaborn as sns

df = pd.read_csv('/Applications/Projects/Python/Datasets/bpm.csv')
new_df = pd.read_csv('/Applications/Projects/Python/Datasets/test.csv')


df=pd.DataFrame(df)
new_df=pd.DataFrame(new_df)

def convertToSeconds(time):
    return( pd.to_datetime(time,format="%H:%M:%S").dt.second + (pd.to_datetime(time ,format="%H:%M:%S").dt.minute)*60 + (pd.to_datetime(time,format="%H:%M:%S").dt.hour)*60*60)


df['time'] = pd.to_datetime(df['time']).dt.strftime("%H:%M:%S")
new_df['time'] = pd.to_datetime(new_df['time']).dt.strftime("%H:%M:%S")

# convert time to seconds
new_df['time'] = convertToSeconds(new_df['time'])
df['time'] = convertToSeconds(df['time'])



X_train = df[['time','bpm']]

knn = NearestNeighbors(n_neighbors=1)
knn.fit(X_train[['time','bpm']])

distances, indices = knn.kneighbors(new_df[['time','bpm']])

threshold_normal = 10 
threshhold_normal_negative = -5
threshold_slight_deviation = 15  
threshold_slight_deviation_negative=-10

classifications = []
predictions = []
predictionsForaccuracy = []
for i in range(len(new_df)):
    nearest_entry = X_train.iloc[indices[i][0]]
    predictions.append([X_train.iloc[indices[i][0]].time])
    predictionsForaccuracy.append([X_train.iloc[indices[i][0]].time])
    diff_bpm = abs(new_df.iloc[i]['bpm'] - nearest_entry['bpm'])
    diff_time = abs(new_df.iloc[i][['time']] - nearest_entry[['time']]).sum()
    if threshhold_normal_negative <=diff_bpm <= threshold_normal :  
        classifications.append("Normal")
        
    elif threshold_slight_deviation_negative <= diff_bpm <= threshold_slight_deviation :  
        classifications.append("Slight Deviation")
        
    else:
        classifications.append("Emergency")
        


colors = {'Normal': '#3EEA72', 'Slight Deviation': '#F97C06', 'Emergency': 'red'}
classificationColors = [colors[classification] for classification in classifications]
print("nearest neighbor identified=> ", predictions[1])


accuracy = accuracy_score(new_df['time'], predictions)
print("Accuracy:", accuracy)

precision = precision_score(new_df['time'], predictions, average='weighted')
print('precision= ',precision)
recall = recall_score(new_df['time'], predictions, average='weighted')
print('recall= ',recall)

f1 = f1_score(new_df['time'], predictions, average='weighted')
print('f1_score= ',f1)




class_report = classification_report(new_df['time'], predictions,target_names=new_df['time'], output_dict=True)
print("Classification Report:\n", class_report)



legend_labels = ['Normal', 'Slight Deviation', 'Emergency']

legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) 
                  for label, color in zip(classifications,classificationColors)]

plt.figure(figsize=(10, 6))
plt.scatter(new_df['time'], new_df['bpm'], c=classificationColors, label='Test Inputs',zorder=10)
plt.plot(df['time'], df['bpm'], label='Training Data', color='blue')
plt.xlabel('Time')
plt.ylabel('BPM')
plt.title('BPM Over Time with Test Input Classifications')
plt.legend(handles=legend_handles, labels=legend_labels)
plt.show()

plotLabels =["Accuracy","Precision","F-Score","Recall"]
plotValues = [accuracy*100,precision*100,f1*100,recall*100]
plt.figure(figsize=(6, 6))
bar=plt.bar(plotLabels,plotValues , color=['skyblue','#8DD116','#EBF11A','#FA753F'])
for bar, value in zip(bar, plotValues):
    plt.text(bar.get_x() + bar.get_width() / 2, value - 5, f'{value:.2f}', ha='center', va='bottom')
plt.xlabel("Metrics")
plt.ylabel("Value")
plt.title("Accuracy")
plt.ylim(0, 100)  
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

