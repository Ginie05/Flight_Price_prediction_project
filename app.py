import pandas as pd
import numpy as np
import xgboost as xgb
import os
import pickle
import joblib
import sklearn
from sklearn.metrics import r2_score
from sklearn import set_config
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PowerTransformer, FunctionTransformer,StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.ensemble import RandomForestRegressor

from feature_engine.encoding import RareLabelEncoder, MeanEncoder, CountFrequencyEncoder
from feature_engine.selection import SelectBySingleFeaturePerformance
from feature_engine.datetime import DatetimeFeatures
from feature_engine.outliers import Winsorizer

import warnings
import matplotlib.pyplot as plt

import streamlit as st 

pd.set_option("display.max_columns",None)
sklearn.set_config(transform_output="pandas")
# convenience function
#airline
airline_pipe=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("grouper", RareLabelEncoder(tol=0.1,replace_with="Other", n_categories=2)),
    ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

#doj
features_to_ex=["month","week","day_of_week","day_of_year"]

doj_pipe=Pipeline(steps=[
    ("dt",DatetimeFeatures(features_to_extract=features_to_ex, yearfirst=True,format="mixed")),
    ("scalar", MinMaxScaler())
])

#source & destination
location_pipe=Pipeline(steps=[
    ("grouper",RareLabelEncoder(tol=0.1,replace_with="Other",n_categories=2)),
    ("encoder",MeanEncoder()),
    ("scaler",PowerTransformer())
])

def is_north(x):
    columns=x.columns.to_list()
    north_cities=["Delhi","kokata","Mumbai","New Delhi"]
    return(
        x
        .assign(**{
            f"{col}_is_north":x.loc[:,col].isin(north_cities).astype(int)
            for col in columns
        })
        .drop(columns=columns)
    )  

location_transformer=FeatureUnion(transformer_list=[
    ("part-1",location_pipe),
    ("part-2",FunctionTransformer(func=is_north))
])

#time
time_pipe1=Pipeline(steps=[
    ("dt", DatetimeFeatures(features_to_extract=["hour","minute"],format="mixed")),
    ("scaler",MinMaxScaler())
])

def part_of_day(x,morning=4,noon=12,eve=16,night=20):
    columns=x.columns.to_list()
    x_temp=x.assign(**{
        col:pd.to_datetime(x.loc[:,col],format="mixed").dt.hour
        for col in columns
    })
    return (
        x_temp
        .assign(**{
            f"{col}_part_of_day":np.select(
                [x_temp.loc[:,col].between(morning,noon,inclusive="left"),
                 x_temp.loc[:,col].between(noon,eve,inclusive="left"),
                 x_temp.loc[:,col].between(eve,night,inclusive="left")],
                ["morning","afternoon","evening"],
                default="night"
            )
            for col in columns
        })
        .drop(columns=columns)
    )

time_pipe2=Pipeline(steps=[
    ("part",FunctionTransformer(func=part_of_day)),
    ("encoder", CountFrequencyEncoder()),
    ("scaler",MinMaxScaler())
])

time_transformer=FeatureUnion(transformer_list=[
    ("part1",time_pipe1),
    ("part2", time_pipe2)
])

#duration

class RBFpercentileSimi(BaseEstimator,TransformerMixin):
    def __init__(self, variables=None,percentiles=[0.2,0.5,0.75],gamma=0.1):
        self.variables=variables
        self.percentiles=percentiles
        self.gamma=gamma

    def fit(self,x,y=None):
        if not self.variables:
            self.variables=x.select_dtypes(include="number").columns.to_list()

        self.reference_values_={
            col:(
                x
                .loc[:,col]
                .quantile(self.percentiles)
                .values
                .reshape(-1,1)
            )
            for col in self.variables
        }
        return self    

    def transform(self,x):
        objects=[]
        for col in self.variables:
            columns=[f"{col}_rbf_{int(percentile*100)}" for percentile in self.percentiles]
            obj=pd.DataFrame(
                data=rbf_kernel(x.loc[:,[col]],Y=self.reference_values_[col],gamma=self.gamma),
                columns=columns
            )
            objects.append(obj)
        return pd.concat(objects,axis=1)
    

def duration_category(x,short=180,med=400):
    return(
        x
        .assign(duration_cat=np.select([x.duration.lt(short),
									    x.duration.between(short, med, inclusive="left")],
									   ["short", "medium"],
									   default="long"))
        .drop(columns="duration")
    )

def is_over(x,value=1000):
    return (
        x
        .assign(**{
            f"duration_over_{value}":x.duration.ge(value).astype(int)
        })
        .drop(columns="duration")
    )

duration_pipe1=Pipeline(steps=[
    ("rbf",RBFpercentileSimi()),
    ("scaler",PowerTransformer())
])

duration_pipe2=Pipeline(steps=[
    ("cat",FunctionTransformer(func=duration_category)),
    ("encoder",OrdinalEncoder(categories=[["short","medium","long"]])),
])

duration_union=FeatureUnion(transformer_list=[
    ("part1",duration_pipe1),
    ("part2",duration_pipe2),
    ("part3",FunctionTransformer(func=is_over)),
    ("part4",StandardScaler())
])

duration_transformer=Pipeline(steps=[
    ("outlier",Winsorizer(capping_method="iqr",fold=1.5)),
    ("imputer", SimpleImputer(strategy="median")),
    ("union",duration_union)
])

#total_stops
def is_direct(x):
    return x.assign(direct_flight=x.total_stops.eq(0).astype(int))

total_stops_transformer=Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("",FunctionTransformer(func=is_direct))
])

#additional_info
info_pipe1=Pipeline(steps=[
    ("group",RareLabelEncoder(tol=0.1,n_categories=2,replace_with="Other")),
    ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
])

def have_info(x):
	return x.assign(additional_info=x.additional_info.ne("No Info").astype(int))

info_union = FeatureUnion(transformer_list=[
	("part1", info_pipe1),
	("part2", FunctionTransformer(func=have_info))
])

info_transformer = Pipeline(steps=[
	("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
	("union", info_union)
])

#Column_transformer
column_transformer=ColumnTransformer(transformers=[
    ("air_pipeline", airline_pipe, ["airline"]),
    ("doj_pipeline", doj_pipe,["date_of_journey"]),
    ("location_transformer", location_transformer,["source","destination"]),
    ("time_transformer", time_transformer,["dep_time","arrival_time"]),
    ("duration_transformer", duration_transformer,["duration"]),
    ("stops",total_stops_transformer,["total_stops"]),
    ("info",info_transformer,["additional_info"])
], remainder="passthrough")

#Selector
estimator = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)

selector = SelectBySingleFeaturePerformance(
	estimator=estimator,
	scoring="r2",
	threshold=0.1
) 

#complete pipeline
preprocessor = Pipeline(steps=[
	("ct", column_transformer),
	("selector", selector)
])

# read the training data
path=r"C:\Users\LENOVO\OneDrive\Desktop\Flight_price_predict_project\DataSets\train1_data_new.csv"
train1_data=pd.read_csv(path)
x_train=train1_data.drop(columns="price")
y_train=train1_data.price.copy()

#fit and save preprocessor

preprocessor.fit(x_train,y_train)
joblib.dump(preprocessor,"preprocessor.joblib")

#web application
st.set_page_config(
    page_title="Flights Prices Prediction using ML model",
    page_icon="✈️",
    layout="wide"
)

st.title("Flights Prices Prediction using Streamlit")

 #user inputs
airline=st.selectbox(
    "Airline:",
     options=x_train["airline"].unique()
)

date_of_journey=st.date_input("Date of Journey:")

source=st.selectbox(
    "Source",
     options=x_train["source"].unique()
)

destination=st.selectbox(
    "Destination",
     options=x_train["destination"].unique()
)

dep_time=st.time_input("Departure Time:")

arrival_time=st.time_input("Arrival Time:")

duration=st.number_input(
    "Duration (mins):",
     step=1,
     min_value=0
)

total_stops=st.number_input(
    "Total_Stops:",
     step=1,
     min_value=0 
)

additional_info=st.selectbox(
    "Additional Info::",
     options=x_train["additional_info"].unique()
)

x_new=pd.DataFrame(dict(
    airline=[airline],
    date_of_journey=[date_of_journey],
    source=[source],
    destination=[destination],
    dep_time=[dep_time],
    arrival_time=[arrival_time],
    duration=[duration],
    total_stops=[total_stops],
    additional_info=[additional_info]  
)).astype({
    col:"str"
    for col in ["date_of_journey","dep_time","arrival_time"]
})

if st.button("predict"):
    saved_preprocessor=joblib.load("preprocessor.joblib")
    x_new_pre=saved_preprocessor.transform(x_new)

    with open("xgboost-model", "rb") as f:
        model=pickle.load(f)
    x_new_xgb=xgb.DMatrix(x_new_pre)
    pred=model.predict(x_new_xgb)[0]
        
    st.info(f"The predicted price is {pred:,.0f} INR")