# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 19:16:57 2019

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


def turntonum(tobeturned,ctobestripped="",ctobereplaced="",cnew=""):
    #takes the list tobturned strips it from ctobestripped and makes the replacemets and int   
    tobeturned=[x.strip(ctobestripped)for x in tobeturned]
    tobeturned=pd.to_numeric(pd.Series(tobeturned).apply(lambda x: x.replace(ctobereplaced,cnew)))
    return tobeturned

def dummify(df,tobedummy):
    m = pd.get_dummies(tobedummy)
    df = pd.concat([df, m], axis=1)
    return df

listings=pd.read_csv ("listings.csv")
listings.drop_duplicates(subset=["id"], inplace=True)
listings.info()

#quality criteria of the listings I will use.
#price cannot be 0
listings.drop(listings[listings.price == "0"].index, inplace=True)

#invalid
listings.drop(listings[listings.accommodates <= 0].index, inplace=True)

#Only market tested data will be used 
listings.drop(listings[listings.review_scores_rating <= 70].index, inplace=True)

#throw away data without reviews, with last calendar update older than 2 months, as useless
listings.drop(listings[listings.first_review.isna()& ~listings["calendar_updated"].str.contains("day|week|month ago")].index, inplace=True)

#throw away old data (last review > than 2 years old)
listings.drop(listings[listings.last_review < "2017-08-07"].index, inplace=True)



#the following ups r2 from 0.51 to 0.54
#deletes 18 values
listings.drop(listings[(listings.cancellation_policy == "super_strict_30") | (listings.cancellation_policy == "super_strict_60")].index, inplace=True)
#------------------------------------------------------------

#--------------------------------------------------------------
listings = listings.reset_index(drop=True)
listings.extra_people=turntonum(listings.extra_people,"$",",","")
listings.price=turntonum(listings.price,"$",",","")


#if the next don't exist they are 0
#and then they are turned to num
listings["security_deposit"].fillna("0", inplace=True)
listings.security_deposit=turntonum(listings.security_deposit,"$",",","")

listings["cleaning_fee"].fillna("0", inplace=True)
listings.cleaning_fee=turntonum(listings.cleaning_fee,"$",",","")

#everybody pays for cleaning.some just say it
listings["real_price"]=listings["price"]+listings["cleaning_fee"]/listings["minimum_nights"]
listings.info()
#listings["max_price"]=(listings["accommodates"]-listings["guests_included"])*listings["extra_people"]+listings["real_price"]
#people give price according to accommodated not the minimum included.
#chose not to go with the above


listings["price_per_guest"]=listings["real_price"]/listings["guests_included"]

#drop them as outliers deletes 15.
#listings.drop(listings[listings.price_per_guest >= 500].index, inplace=True)
#listings = listings.reset_index(drop=True)



#done with the drops---------------------------------------------------------------------------------------------
#guests included cannot be more than the accomodated.
#if the extra people price is 0, then guests included are the maximum of people accomodated

listings['guests_included'] = np.where((listings['extra_people'] == 0)\
                             ,listings['accommodates'], listings['guests_included'])

listings['guests_included'] = np.where((listings['guests_included'] > listings["accommodates"])\
                             ,listings['accommodates'], listings['guests_included'])

#fill na with 0 
listings["bathrooms"].fillna(0, inplace=True)
listings["bedrooms"].fillna(0, inplace=True)
listings["beds"].fillna(0, inplace=True)
listings["reviews_per_month"].fillna(0, inplace=True)
listings["summary"].fillna("", inplace=True)
listings["name"].fillna("", inplace=True)

#done with the transformations
descriptions=listings[["name","summary",'space','description','neighborhood_overview','notes','transit',\
                  'access','interaction','house_rules','amenities']]


listings=listings[['listing_url','neighbourhood_cleansed',\
                  'is_location_exact','property_type',\
                  'room_type','accommodates','bathrooms','bedrooms','beds','bed_type',\
                  'price','security_deposit','cleaning_fee',\
                  'guests_included','last_review',\
                  'number_of_reviews',"minimum_nights","extra_people",\
                  'instant_bookable','cancellation_policy',\
                  'reviews_per_month',"review_scores_rating",\
                  "real_price","price_per_guest"]]


 

#turning dates into ordeals for regression but they seem unrelated
#listings['first_review'] = pd.to_datetime(listings['first_review'], format='%Y-%m-%d')
#listings['first_review']=listings['first_review'].map(dt.datetime.toordinal)

#-----------------------------------new columns-----------------------------------------------------


listings[listings.room_type=="Shared room"]




#gives the average price of each cleansed neighbourhood. the other expected_by were highly correlated
listings['expected_by_neigh'] = listings["guests_included"]*listings["price_per_guest"].groupby(listings['neighbourhood_cleansed']).transform('mean')
#listings['expected_by_neigh2']= listings.groupby('neighbourhood_cleansed')["price_per_guest"].transform(lambda x : x.mean())
#listings['expected_by_prop'] = listings["guests_included"]*listings["price_per_acom"].groupby(listings['property_type']).transform('mean')
#listings['expected_by_room'] = listings["guests_included"]*listings["price_per_acom"].groupby(listings['room_type']).transform('mean')
#listings['expected_by_bed'] =listings["guests_included"]*listings["price_per_acom"].groupby(listings['bed_type']).transform('mean')
#listings['expected_by_exact'] = listings["guests_included"]*listings["price_per_acom"].groupby(listings['is_location_exact']).transform('mean')
#listings['expected_by_cancel'] = listings["guests_included"]*listings["price_per_acom"].groupby(listings['cancellation_policy']).transform('mean')

#dropper=["price","cleaning_fee",'neighbourhood_cleansed','property_type','room_type',\
#         'bed_type','is_location_exact','cancellation_policy','latitude','longitude']
#listings.drop(dropper,inplace=True, axis=1)

#listings.info()

#df = sns.load_dataset(listings)


#checking key_words-------------------------------------------------------------------------------------------
listings[listings["summary"].str.contains("central station|walking distance|public transport|city center|city centre")].real_price.mean()

listings[listings["name"].str.contains("luxurious")].real_price.mean()

listings.room_type

listings.real_price.mean()  

listings.groupby(['room_type'])[["real_price"]].count()

# Change line width
#getting an idea---------------------------------------------------------------
listings.hist(bins=50, figsize=(20,15))
plt.savefig("attribute_histogram_plots")
plt.show()
plt.savefig('general.png')

#longitude latitude plot (with color)
listings.plot(kind="scatter", x="longitude", y="latitude", alpha=0.2,\
figsize=(10,7),c="price", cmap=plt.get_cmap("jet"), colorbar=True,\
    sharex=False)
plt.savefig('map1.png')

#-------------------------------------------------------------------------------



#correlation matrix+plot-------------------------------------------------------
corr_matrix = listings.corr()
corr_matrix["real_price"].sort_values(ascending=False)

sns.set(rc={'figure.figsize':(30.7,14.27)})
hel=sns.heatmap(data=corr_matrix, annot=True)
fig2=hel.get_figure()
fig2.savefig('heatmap.png')
#------------------------------------------------------------------------------

#how many neighbohoods
len(listings['neighbourhood'].value_counts())


#cancellation policy,neighbourhood cleansed,property_type,room_type,"bed_type"
#
###city is contained in neighboorhoods cleansed
 
# Give it to the BOXENPLOT--------------------------------------------------
sns.set(rc={'figure.figsize':(30.7,14.27)})
my_order = listings.groupby(["neighbourhood_cleansed"],sort = True)["price_per_guest"].mean().iloc[::-1].index

boom = sns.boxenplot(x="neighbourhood_cleansed", y="price_per_guest",\
                   data=listings, palette="Set2",outlier_prop=0.01,order=sorted(my_order))
boom.set_xticklabels(boom.get_xticklabels(), rotation=30, ha="right")
boom.set_title("Prices by Neighbourhood")
fig = boom.get_figure()
fig.savefig('Prices by Neighbourhood.png') 

#------------------------------------------------------------------------------

#testing bolleans
#is_location_exact
listings.groupby(['cancellation_policy'])[["real_price"]].mean()
#m=listings.groupby(['neighbourhood_cleansed'])[["price_per_acc"]].mean()
#trials=pd.concat([trials,m],axis=1)



#preparing the learning
x=listings[['real_price','bedrooms',"bathrooms","room_type","accommodates","number_of_reviews",\
            'beds','expected_by_neigh','reviews_per_month',"cancellation_policy"]]#"cancellation_policy","neighbourhood_cleansed",\
           # "property_type","room_type","bed_type",'bathrooms',\
           # 'bedrooms','beds','security_deposit','guests_included',\
           # 'reviews_per_month',"accommodates",
#            'accommodates','bedrooms','beds','bathrooms']]
#'expected_by_room','expected_by_cancel',\
#            'expected_by_prop','expected_by_exact','expected_by_bed'
x.info()

#--------------------------------------------------------------------------
x=dummify(x,x.room_type)
x=dummify(x,x.cancellation_policy)

drops = ["cancellation_policy","room_type"]
#         "room_type","bed_type"]
x.drop(drops, inplace=True, axis=1)
#---------------------------------------------------------------
#try accomodat/sqft and then ifna->acom *num

#x = pd.concat([x, np.log(listings['accommodates'])], axis=1)

y=np.log(x["real_price"])
x=x.drop(["real_price"],axis=1)
x.info()
#from sklearn.metrics import mean_squared_error
from sklearn.model_selection  import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

#--------------------------------gradient boosting-------------------------------------
from sklearn import ensemble
#from sklearn.metrics import mean_squared_error
clf = ensemble.GradientBoostingRegressor(n_estimators =100, max_depth = 7,
          learning_rate = 0.05, loss = 'ls')
clf.fit(x_train, y_train)
#clfpred = clf.predict(x_test)
#r2_score(y_test,clfpred)

clf.score(x_test,y_test)


#linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train,y_train)
#lmpred = lm.predict(x_test)
#r2_score(y_test,lmpred)
lm.score(x_test,y_test).round(2)
#0.398934438504138
#0.45058655931133307
#--------------------------------------------------------------------------------



#0.40744782476506913
#0.35

#-------------------------------------------------------------------------------


#--------------------------random forest-----------------------------------------------------
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 2000, random_state = 42)
# Train the model on training data
rf.fit(x_train, y_train)
#rfpred=rf.predict(x_test)
#r2_score(y_test,rfpred)

rf.score(x_test,y_test)
#0.3046711480564963
#0.39325714884103563

#polynomial regression-------------------------------------------------------------------------
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.pipeline import make_pipeline
  # transforms the existing features to higher degree features.
poly_features = PolynomialFeatures(degree=2)
x_train_poly = poly_features.fit_transform(x_train)
  
  # fit the transformed features to Linear Regression
#poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression(fit_intercept = False))
poly_model = LinearRegression()
poly_model.fit(x_train_poly, y_train)
poly_model.score(poly_features.fit_transform(x_test),y_test)  
  # predicting on training data-set

