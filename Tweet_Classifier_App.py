##import streamlit
import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import resample

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC  
from sklearn.model_selection import GridSearchCV




header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title('Tweet classifier')



with dataset:
    st.header('Dataset')
    st.text('Data collected')

    test_data = pd.read_csv('data/test.csv')
    st.write(test_data.head())

    train_data = pd.read_csv('data/train.csv')
    st.write(train_data.head())

    st.subheader('Number of Tweets v/s Sentiments')
    tweets_vs_sentiments_dist= pd.DataFrame(train_data["sentiment"].value_counts()).head()
    st.bar_chart(tweets_vs_sentiments_dist)


    
with features:
    st.header('Features we created')
    st.text('')
    

with model_training: 
    st.header('Time to train the model')
    st.text('')

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('Max depth of the model?',min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('How many trees should there be?', options=[100,200,300,'No limit'], index = 0)

    sel_col.text('Here is a list of features in the data:')
    sel_col.write(train_data.columns)

    input_feature =sel_col.text_input('which feature should be used as the input feature?',"sentiment")
    
    regr=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)

    X= train_data[[input_feature]]
    y= train_data[["tweetid"]]

    regr.fit(X, y) 

    prediction = regr.predict(y)

    disp_col.subheader('Mean absolute error of the model is;')
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader("mean squared error of the model is:")
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader('R squared score of the model is:')
    disp_col.write(mean_absolute_error(y, prediction))

    # Initialize model
    X_test= test_data[[input_feature]]
    y_pred_1= test_data[["tweetid"]]

    X_test= train_data[[input_feature]]
    y_test_1= train_data[["tweetid"]]



    ##MNB_model_1 = MultinomialNB()
    # Fit the model
    #MNB_model_1.fit(X, y)

    #Create Prediction
    #y_pred_1 = MNB_model_1.predict(X)

    #Get f1_score

    #f1score_1 = f1_score(y_test_1, y_pred_1, average= 'weighted')
    #Disp_col.write('macro f1 score:', f1score_1)


    





