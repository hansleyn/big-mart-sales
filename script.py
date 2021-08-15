# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np

# Read files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#add a column with the title "source" to the train data and all entries
#have been made to be the str 'train'
train["source"]="train"

#add a column with the title "source" to the test data and all entries
#have been made to be the str 'test'
test["source"]="test"

#concatenate the "test" and "train" data together
data = pd.concat([train, test], ignore_index=True)

print (train.shape, test.shape, data.shape)

# Checking which columns contain missing values
data.apply(lambda x: sum(x.isnull()))

# Some basic statistics for the numerical variables
data.describe()

# Looking at the number of unique values in each column
data.apply(lambda x: len(x.unique()))


#Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=="object"]
#Exclude ID cols and source:
categorical_columns = [x for x in categorical_columns if x not in ["Item_Identifier",
                                                                   "Outlet_Identifier","source"]]
#Print frequency of categories
for col in categorical_columns:
    print ("\nFrequency of Categories for variable %s" %col)
    print (data[col].value_counts())
#%%
#Determine the average weight per item:
item_avg_weight = data.pivot_table(values=["Item_Weight"], index = ["Item_Identifier"], aggfunc = np.mean)

#Get a boolean variable specifying missing Item_Weight values
miss_weight = data["Item_Weight"].isnull()

#Impute data and check missing values before and after imputation to confirm
print ("Original # missing: %d" %sum(miss_weight))

data.loc[miss_weight,"Item_Weight"] = data.loc[miss_weight,"Item_Identifier"].apply(lambda x: item_avg_weight.Item_Weight[x])

miss_weight = data["Item_Weight"].isnull()

print ("Final # missing: %d" %sum(miss_weight))

#%%

#Determing the mode for each
outlet_size_mode = data.pivot_table(values=["Outlet_Size"], index=["Outlet_Type"], aggfunc=(lambda x: x.mode().iat[0]))
print ("Mode for each Outlet_Type:")
print (outlet_size_mode)    

#Get a boolean variable specifying missing Outlet_Size values
miss_size = data["Outlet_Size"].isnull()

#Impute data and check missing values before and after imputation to confirm
print ("Original # missing: %d" %sum(miss_size))
       
data.loc[miss_size,"Outlet_Size"] = data.loc[miss_size, "Outlet_Type"].apply(lambda x: outlet_size_mode.Outlet_Size[x])

miss_size = data["Outlet_Size"].isnull()

print ("Final # missing: %d" %sum(miss_size))
#%%
    
print (data.pivot_table(values=["Item_Outlet_Sales"],index=["Outlet_Type"], aggfunc = np.mean))

#%%

#Determine average visibility of a product
visibility_avg = data.pivot_table(values=["Item_Visibility"], index=["Item_Identifier"], aggfunc = np.mean)

#Impute 0 values with mean visibility of that product:
miss_visibility = (data["Item_Visibility"] == 0)

print ("Number of 0 values initially: %d"% sum(miss_visibility))
data.loc[miss_visibility,"Item_Visibility"] = data.loc[miss_visibility,"Item_Identifier"].apply(lambda x: visibility_avg.Item_Visibility[x])

miss_visibility = (data["Item_Visibility"] == 0)

print ("Number of 0 values after modification: %d" %sum(miss_visibility))

#%%
#Determine another variable with means ratio
data["Item_Visibility_MeanRatio"] = data.apply(lambda x: x["Item_Visibility"]/(visibility_avg.Item_Visibility[x["Item_Identifier"]]), axis=1)

print (data["Item_Visibility_MeanRatio"].describe())

#%%

#Get the first two characters of ID:
data["Item_Type_Combined"] = data["Item_Identifier"].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
data["Item_Type_Combined"] = data["Item_Type_Combined"].map({"FD":"Food", "NC":"Non-Consumable", "DR":"Drinks"})
print (data["Item_Type_Combined"].value_counts())

#%%

#Years:
data["Outlet_Years"] = 2013 - data["Outlet_Establishment_Year"]

print (data["Outlet_Years"].describe())

#%%

#Change categories of low fat:
print ("Original Categories:")
print (data["Item_Fat_Content"].value_counts())

print ('\nModified Categories:')
data["Item_Fat_Content"] = data["Item_Fat_Content"].replace({"LF":"Low Fat", "reg":"Regular", "low fat":"Low Fat"})
print (data["Item_Fat_Content"].value_counts())

#%%

#Mark non-consumables as separate category in low_fat:
data.loc[data["Item_Type_Combined"]=="Non-Consumable", "Item_Fat_Content"] = "Non-Edible"

print (data["Item_Fat_Content"].value_counts())

#%%

#Import library:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#New variable for outlet
data["Outlet"] = le.fit_transform(data["Outlet_Identifier"])
var_mod = ["Item_Fat_Content","Outlet_Location_Type","Outlet_Size","Item_Type_Combined","Outlet_Type","Outlet"]

le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])
    
#One Hot Coding:
data = pd.get_dummies(data, columns= var_mod)
    
data[["Item_Fat_Content_0","Item_Fat_Content_1","Item_Fat_Content_2"]].head(12)

#%%

#Drop the columns which have been converted to different types:
data.drop(["Item_Type","Outlet_Establishment_Year"],axis=1,inplace=True)

#Divide into test and train:
train = data.loc[data["source"]=="train"]
test = data.loc[data["source"]=="test"]

#Drop unnecessary columns:
test.drop(["Item_Outlet_Sales","source"],axis=1,inplace=True)
train.drop(["source"],axis=1,inplace=True)
"""
#Export files as modified versions:
train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)
"""