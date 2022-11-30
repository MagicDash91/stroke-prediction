import pandas as pd
import numpy as np
import streamlit as st

st.header('Stroke Prediction Using Random Forest Classifier')

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df['bmi'].fillna(df['bmi'].median(), inplace = True)

#Change value in ever_married column
df['ever_married'] = df['ever_married'].replace(['No'],'0')
df['ever_married'] = df['ever_married'].replace(['Yes'],'1')
#Change value in work_type column
df['work_type'] = df['work_type'].replace(['children'],'0')
df['work_type'] = df['work_type'].replace(['Govt_job'],'1')
df['work_type'] = df['work_type'].replace(['Never_worked'],'2')
df['work_type'] = df['work_type'].replace(['Private'],'3')
df['work_type'] = df['work_type'].replace(['Self-employed'],'4')
#Change value in Residence_type column
df['Residence_type'] = df['Residence_type'].replace(['Rural'],'0')
df['Residence_type'] = df['Residence_type'].replace(['Urban'],'1')
#Change value in smoking_status column
df['smoking_status'] = df['smoking_status'].replace(['never smoked'],'0')
df['smoking_status'] = df['smoking_status'].replace(['formerly smoked'],'1')
df['smoking_status'] = df['smoking_status'].replace(['smokes'],'2')
df['smoking_status'] = df['smoking_status'].replace(['Unknown'],'3')
#Change value in gender column
df['gender'] = df['gender'].replace(['Female'],'0')
df['gender'] = df['gender'].replace(['Male'],'1')
df['gender'] = df['gender'].replace(['Other'],'2')

df['gender'] = pd.to_numeric(df['gender'])
df['ever_married'] = pd.to_numeric(df['ever_married'])
df['work_type'] = pd.to_numeric(df['work_type'])
df['Residence_type'] = pd.to_numeric(df['Residence_type'])
df['smoking_status'] = pd.to_numeric(df['smoking_status'])

from sklearn.utils import resample
#create two different dataframe of majority and minority class 
df_majority = df[(df['stroke']==0)] 
df_minority = df[(df['stroke']==1)] 
# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples= 4861, # to match majority class
                                 random_state=0)  # reproducible results
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_minority_upsampled, df_majority])

import scipy.stats as stats
z = np.abs(stats.zscore(df_upsampled))
data_clean = df_upsampled[(z<3).all(axis = 1)]

data_clean2 = data_clean.drop('id', axis=1)

X = data_clean2.drop('stroke', axis=1)
y = data_clean2['stroke']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0, max_depth=3)
rfc.fit(X_train, y_train)


gender = st.radio("What's your Gender ?",('Female', 'Male', 'Other'))
if gender == 'Female':
  gender2 = gender.replace('Female', '0')
  gender3 = float(gender2)
elif gender == 'Male':
  gender2 = gender.replace('Male', '1')
  gender3 = float(gender2)
elif gender == 'Other':
  gender2 = gender.replace('Other','2')
  gender3 = float(gender2)

#gender = st.number_input('Gender')

age = st.number_input('Age')

hypertension = st.radio("Do You have hypertension ?",('No', 'Yes'))
if hypertension == 'No':
  hypertension2 = hypertension.replace('No', '0')
  hypertension3 = float(hypertension2)
elif hypertension == 'Yes':
  hypertension2 = hypertension.replace('Yes', '1')
  hypertension3 = float(hypertension2)


heart = st.radio("Do You have heart Disease ?",('No', 'Yes'))
if heart == 'No':
  heart2 = heart.replace('No', '0')
  heart3 = float(heart2)
elif heart == 'Yes':
  heart2 = heart.replace('Yes', '1')
  heart3 = float(heart2)

#heart = st.number_input('Do You Have heart Disease ?')

marry = st.radio("Are You Married ?",('No', 'Yes'))
if marry == 'No':
  marry2 = marry.replace('No', '0')
  marry3 = float(marry2)
elif marry == 'Yes':
  marry2 = marry.replace('Yes', '1')
  marry3 = float(marry2)

#marry = st.number_input('Are you Married ?')

work = st.radio("Your Worktype ?",('Children','Government Job','Never Worked','Private','Self Employed'))
if work == 'Children':
  work2 = work.replace('Children', '0')
  work3 = float(work2)
elif work == 'Government Job':
  work2 = work.replace('Government Job', '1')
  work3 = float(work2)
elif work == 'Never Worked':
  work2 = work.replace('Never Worked', '2')
  work3 = float(work2)
elif work == 'Private':
  work2 = work.replace('Private', '3')
  work3 = float(work2)
elif work == 'Self Employed':
  work2 = work.replace('Self Employed', '4')
  work3 = float(work2)

#work = st.number_input('Your Worktype ?')

res = st.radio("Your Residence Type ?",('Rural', 'Urban'))
if res == 'Rural':
  res2 = res.replace('Rural', '0')
  res3 = float(res2)
elif res == 'Urban':
  res2 = res.replace('Urban', '1')
  res3 = float(res2)

#residence = st.number_input('Your Residence Type ? (0 = Rural, 1 = Urban)')

avg = st.number_input('Average Glucose Level ?')

bmi = st.number_input('You BMI ?')

smoke = st.radio("Your Smoking Status ?",('Never Smoked','Formerly Smoked','Smokes','Unknown'))
if smoke == 'Never Smoked':
  smoke2 = smoke.replace('Never Smoked', '0')
  smoke3 = float(smoke2)
elif smoke == 'Formerly Smoked':
  smoke2 = smoke.replace('Formerly Smoked', '1')
  smoke3 = float(smoke2)
elif smoke == 'Smokes':
  smoke2 = smoke.replace('Smokes', '2')
  smoke3 = float(smoke2)
elif smoke == 'Unknown':
  smoke2 = smoke.replace('Unknown', '3')
  smoke3 = float(smoke2)

#smoke = st.number_input('Your smoking status ?')

Xnew3 = [[gender3, age, hypertension3, heart3, marry3, work3, res3, avg, bmi, smoke3]]

y_pred_prob4 = rfc.predict_proba(Xnew3)
y_pred_prob_df4 = pd.DataFrame(data=y_pred_prob4, columns=['Prob of dont have stroke', 'Prob of have stroke'])
hasil = (y_pred_prob_df4)*100
hasil2 = hasil.astype(int)
st.write("Result (Probability in Percantage)")
st.dataframe(hasil2)
st.write("Bar Chart Result (Probability in Percantage)")
st.bar_chart(hasil2)
