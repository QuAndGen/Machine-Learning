# -*- coding: utf-8 -*-
"""
Created on Tue Feb 07 12:05:36 2017

@author: 310227823
"""

#Data structure

#Lists – Lists are one of the most versatile data structure in Python. 
#A list can simply be defined by writing a list of comma separated values in 
#square brackets. Lists might contain items of different types, but usually the 
#items all have the same type.
#Python lists are mutable and individual elements of a list can be changed.

list1=[0,1,2,3,4]

list1 #Print all data
list1[:] #Print all data
#Get data by index ( position) index starts from 0
list1[0] #first value in the list
list1[1] #second value in the list

list1[1:3] #second and third value in the list (Note:- Given 3 will not give 3rd value. actual position is 1,2)
list1[:3]#first 3 elements
list1[1:10] #second to last values in the list (Note:- However we do not have 10 values in the list but 
#it will give value till max position and rest of the posstion will be ignored )

list1[-1]#Get last value (Not in case of Revers order, index starts with -1 
#. For example here in list1 we have 5 values so the revese possition will go by -1,-2,....-5)

#Modify list

list1[3]=6 #replace fourth possition value which is 3 in list1 with 6
list1
list1.append(10) #Insert new value in list
list1.remove(10) #Remove value from the list.(Note in this case we need to know the actual value, not position)
list1=list1+[201]#Insert new value in list
list1.append(10) #Insert new value in list
list1.pop(5)#This will delete the data and print also
list1.insert(2,200)#This will add 200 at 2nd position and replace the posstion of other values
list1.index(200)#Get the index of actual value. In this case we are just trying to find the position of no 200.

del list1[2]#Delete second value

len(list1)#Length of list
[1, 2, 3] + [4, 5, 6]#Add values to the list
['Hi!'] * 4#Repeat Hi! 4 times in a list
3 in [1, 2, 3]#Check the value existance in the list (Membership)
for x in [1, 2, 3]: print x #Print the value of list (Iteration)

cmp(list1, list2)#Compares elements of both lists.
len(list)#Gives the total length of the list.
max(list)#Returns item from the list with max value.
min(list)#Returns item from the list with min value.
list(seq)#Converts a tuple into list.
list.append(obj)#Appends object obj to list
list.count(obj)#Returns count of how many times obj occurs in list
list.extend(seq)#Appends the contents of seq to list
list.index(obj)#Returns the lowest index in list that obj appears
list.insert(index, obj)#Inserts object obj into list at offset index
list.pop(obj=list[-1])#Removes and returns last object or obj from list
list.remove(obj)#Removes object obj from list
list.reverse()#Reverses objects of list in place
list.sort([func])#Sorts objects of list, use compare func if given

#Create two list and combine them
names = ['Bob','Jessica','Mary','John','Mel']
births = [968, 155, 77, 578, 973]
BabyDataSet = list(zip(names,births))
BabyDataSet
#Count the occurance of each ellement in list
a=[1,2,3,1,3,1,2,3,4,4]
[[x,a.count(x)] for x in set(a)]

#Strings********************************************************************
#Strings can simply be defined by use of single ( ' ), double ( " ) or triple ( "' ) 
#inverted commas. Strings enclosed in tripe quotes ( "' ) can span over multiple lines 
#and are used frequently in docstrings (Python’s way of documenting functions). 
#\ is used as an escape character. Please note that Python strings are immutable, 
#so you can not change part of strings.

msg="Hello"

print msg
a=raw_input("Enter you Name:",) #Ask user to input their name
print msg+" "+a
msg[1] #Get first character from the text.

print "My name is %s and weight is %d kg!" % ('Zara', 21) 

capitalize()#Capitalizes first letter of string
center(width, fillchar)#Returns a space-padded string with the original string centered 
#to a total of width columns.
count(str, beg= 0,end=len(string))#Counts how many times str occurs in string or in a substring of string 
#if starting index beg and ending index end are given.
decode(encoding='UTF-8',errors='strict')#Decodes the string using the codec registered for encoding. 
#encoding defaults to the default string encoding.

encode(encoding='UTF-8',errors='strict')#Returns encoded string version of string; on error, default is to raise a ValueError 
#unless errors is given with 'ignore' or 'replace'.
endswith(suffix, beg=0, end=len(string))#Determines if string or a substring of string (if starting index beg and ending index end are given) 
#ends with suffix; returns true if so and false otherwise.
expandtabs(tabsize=8)#Expands tabs in string to multiple spaces; defaults to 8 spaces per tab if tabsize not provided.
find(str, beg=0 end=len(string))#Determine if str occurs in string or in a substring of string if starting index beg and ending index 
#end are given returns index if found and -1 otherwise.
index(str, beg=0, end=len(string))#Same as find(), but raises an exception if str not found.
isalpha()#Returns true if string has at least 1 character and all characters are alphabetic and false otherwise.
isdigit()#Returns true if string contains only digits and false otherwise.
islower()#Returns true if string has at least 1 cased character and 
#all cased characters are in lowercase and false otherwise.
isnumeric()#Returns true if a unicode string contains only numeric characters and false otherwise.
isspace()#Returns true if string contains only whitespace characters and false otherwise.
istitle()#Returns true if string is properly "titlecased" and false otherwise.
isupper()#Returns true if string has at least one cased character and all cased characters are 
#in uppercase and false otherwise.
seq=['a','b']
seq1='c'
seq1.join(seq)#Merges (concatenates) the string representations of elements in sequence seq into a string, with separator string.


len(string)#Returns the length of the string
ljust(width[, fillchar])#Returns a space-padded string with the original string left-justified to a total of width columns.
lower()#Converts all uppercase letters in string to lowercase.
lstrip()#Removes all leading whitespace in string.
maketrans()#Returns a translation table to be used in translate function.
max(str)#Returns the max alphabetical character from the string str.
min(str)#Returns the min alphabetical character from the string str.
replace(old, new [, max])#Replaces all occurrences of old in string with new or at most max occurrences if max given.
rfind(str, beg=0,end=len(string))#Same as find(), but search backwards in string.
rindex( str, beg=0, end=len(string))#Same as index(), but search backwards in string.
rjust(width,[, fillchar])#Returns a space-padded string with the original string right-justified to a total of width columns.
rstrip()#Removes all trailing whitespace of string.
split(str="", num=string.count(str))#Splits string according to delimiter str (space if not provided) and returns list of substrings; 
#split into at most num substrings if given.
splitlines( num=string.count('\n'))#Splits string at all (or num) NEWLINEs and returns a 
#list of each line with NEWLINEs removed.
startswith(str, beg=0,end=len(string))#Determines if string or a substring of string (if starting 
#index beg and ending index end are given) 
#starts with substring str; returns true if so and false otherwise.
strip([chars])#Performs both lstrip() and rstrip() on string
swapcase()#Inverts case for all letters in string.
title()#Returns "titlecased" version of string, that is, all words begin with uppercase and the rest are lowercase.
translate(table, deletechars="")#Translates string according to translation table str(256 chars), removing those in the del string.
upper()#Converts lowercase letters in string to uppercase.
zfill (width)#Returns original string leftpadded with zeros to a total of 
#width characters; intended for numbers, zfill() retains any sign given (less one zero).
isdecimal()#Returns true if a unicode string contains only decimal characters and false otherwise.

#Tuples – **********************************************************************
#A tuple is represented by a number of values separated by commas. Tuples are immutable 
#and the output is surrounded by parentheses so that nested tuples are processed correctly. Additionally, 
#even though tuples are immutable, they can hold mutable data if needed.
#Since Tuples are immutable and can not change, they are faster in processing as compared to lists. 
#Hence, if your list is unlikely to change, you should use tuples, instead of lists.

tup=10,20,2,1,4
tup=(10,20,2,1,4)
tup#print all values
tup[0]#print 1st value
tup[3]#print 3rd value
tup[-1]#print last value
tup[1:3]#print 2nd and 3rd value
tup[:3]#print 1st value to 3rd value


#Dictionary – *************************************************************
#Dictionary is an unordered set of key: value pairs, with the requirement 
#that the keys are unique (within one dictionary). A pair of braces creates an empty dictionary: {}. 

dicton={'A':2300,'B':345,'C':'abc','D':'xyz','E':12345,'F':'tttt'}
dicton#Print all values
dicton['F']#Print only selected field

dicton.keys()#GEt column names
dicton.values()#GEt only values
dicton['A'] = 8; # update existing entry
dicton['G'] = "DPS School"; # Add new entry
del dicton['G']#Delete keys G
del dicton #Delete dictionary
dicton.clear()#Make empty dictionary

dicton1={'H':34,'I':23,'C':23,'D':'xyz','E':12345,'T':'aaaa'}
dic2={'ID':[1,2],'Namw':['A','B']}
cmp(dicton,dicton1)#Compare two dictionary

len(dicton)#Get the number of items of dictionary

str(dicton)#Produces a printable string representation of a dictionary

type(dicton)#Returns the type of the passed variable. If passed variable is dictionary, 
#then it would return a dictionary type.

dict.clear()#Removes all elements of dictionary dict
dict.copy()Returns a shallow copy of dictionary dict
dict.fromkeys()#Create a new dictionary with keys from seq and values set to value.
dict.get(key, default=None)#For key key, returns value or default if key not in dictionary
dict.has_key(key)#Returns true if key in dictionary dict, false otherwise
dict.items()#Returns a list of dict's (key, value) tuple pairs
dict.keys()#Returns list of dictionary dict's keys
dict.setdefault(key, default=None)#Similar to get(), but will set dict[key]=default if key is not already in dict
dict.update(dict2)#Adds dictionary dict2's key-values pairs to dict
dict.values()#Returns list of dictionary dict's values


#*************************************************************************************
#Connect with Database
#*************************************************************************************

import MySQLdb

# Open database connection
db = MySQLdb.connect("localhost","testuser","test123","TESTDB" )

# prepare a cursor object using cursor() method
cursor = db.cursor()

# Prepare SQL query to INSERT a record into the database.
sql = """INSERT INTO EMPLOYEE(FIRST_NAME,
         LAST_NAME, AGE, SEX, INCOME)
         VALUES ('Mac', 'Mohan', 20, 'M', 2000)"""
try:
   # Execute the SQL command
   cursor.execute(sql)
   # Commit your changes in the database
   db.commit()
except:
   # Rollback in case there is any error
   db.rollback()

# disconnect from server
db.close()

#*******************Read the data
#!/usr/bin/python

import MySQLdb

# Open database connection
db = MySQLdb.connect("localhost","testuser","test123","TESTDB" )

# prepare a cursor object using cursor() method
cursor = db.cursor()

# Prepare SQL query to INSERT a record into the database.
sql = "SELECT * FROM EMPLOYEE \
       WHERE INCOME > '%d'" % (1000)
try:
   # Execute the SQL command
   cursor.execute(sql)
   # Fetch all the rows in a list of lists.
   results = cursor.fetchall()
   for row in results:
      fname = row[0]
      lname = row[1]
      age = row[2]
      sex = row[3]
      income = row[4]
      # Now print fetched result
      print "fname=%s,lname=%s,age=%d,sex=%s,income=%d" % \
             (fname, lname, age, sex, income )
except:
   print "Error: unable to fecth data"

# disconnect from server
db.close()

#Connect sql with the help of pandas

# Parameters
TableName = "data"

DB = {
    'drivername': 'mssql+pyodbc',
    'servername': 'DAVID-THINK',
    #'port': '5432',
    #'username': 'lynn',
    #'password': '',
    'database': 'BizIntel',
    'driver': 'SQL Server Native Client 11.0',
    'trusted_connection': 'yes',  
    'legacy_schema_aliasing': False
}
#**********************************************************************************************************
#Database connection
#**********************************************************************************************************
# Import libraries
import pandas as pd
import sys
from sqlalchemy import create_engine, MetaData, Table, select, engine
# Create the connection
engine = create_engine(DB['drivername'] + '://' + DB['servername'] + '/' + DB['database'] + '?' + 'driver=' + DB['driver'] + ';' + 'trusted_connection=' + DB['trusted_connection'], legacy_schema_aliasing=DB['legacy_schema_aliasing'])
conn = engine.connect()

# Required for querying tables
metadata = MetaData(conn)

# Table to query
tbl = Table(TableName, metadata, autoload=True, schema="dbo")
#tbl.create(checkfirst=True)

# Select all
sql = tbl.select()

# run sql code
result = conn.execute(sql)

# Insert to a dataframe
df = pd.DataFrame(data=list(result), columns=result.keys())

# Close connection
conn.close()

print('Done')




#****************************Data Frame******************************************
import pandas as pd
#Create data frame with list


#Create two list and combine them
names = ['Bob','Jessica','Mary','John','Mel']
births = [968, 155, 77, 578, 973]
BabyDataSet = list(zip(names,births))
BabyDataSet
df = pd.DataFrame(data = BabyDataSet, columns=['Names', 'Births'])
df
#****************************************************************************************************
#Reading or sending data from other sources
#Send this data to csv
df.to_csv('births1880.csv',index=False,header=False)
#REad data from csv file
Location = r'births1880.csv'
df = pd.read_csv(Location,header=None,names=['Names','Births'])
df
import os
os.remove(Location)#REmove file from directory


# Location of file
Location = r'C:\Users\david\notebooks\Lesson3.xlsx'

# Parse a specific sheet
df = pd.read_excel(Location, 0, index_col='StatusDate')# StatusDate is a column name in excel 0 = first sheet
df.dtypes

# Export to Excel

df.to_excel('test1.xlsx', sheet_name = 'test1', index = False)
df.to_excel('test2.xlsx', sheet_name = 'test2', index = False)
df.to_excel('test3.xlsx', sheet_name = 'test3', index = False)

# List to hold file names
FileNames = []

# Your path will be different, please modify the path below.
os.chdir(r"C:\Users\david\notebooks")

# Find any file that ends with ".xlsx"
for files in os.listdir("."):
    if files.endswith(".xlsx"):
        FileNames.append(files)
        
FileNames
#****************************************************************************************************
#Prepare Data

# Check data type of the columns
df.dtypes

# Check data type of Births column
df.Births.dtype

# The inital set of baby names

df[:]#Print all data
df[2:]#Print data from 3rd row to end of data

df[2:]#Print data from 3rd row to end of data

df.info()#Get all information about the data frame

df.head()#GEt default top of rows
df.head(2)#GEt default 2 number of rows

df.tail()#Get defalt botom rows
df.tail(2)#Get defalt 2 botom rows
#Unique information from the specific column
df['Names'].unique()#Get unique name
# If you actually want to print the unique values:
for x in df['Names'].unique():
    print(x)

print(df['Names'].describe())#Get the number of unique count

Output.extend(zip(random_states, random_status, data, rng))#combine multiple list together

# Save results to excel
df.to_excel('Lesson3.xlsx', index=False)

df.index#Get index of data frame


df[df['Births']>155]#Print data frame which are after 155 birth

df.index


# Our small data set
d = [0,1,2,3,4,5,6,7,8,9]

# Create dataframe
df = pd.DataFrame(d)
df
# Lets change the name of the column
df.columns = ['Rev']
df
# Lets add a column
df['NewCol'] = 5
df
# Lets modify our new column
df['NewCol'] = df['NewCol'] + 1
df
# We can delete columns
del df['NewCol']
df
# Lets add a couple of columns
df['test'] = 3
df['col'] = df['Rev']
df
# If we wanted, we could change the name of the index
i = ['a','b','c','d','e','f','g','h','i','j']
df.index = i
df
df.loc['a']#Getting row with index 'a'

# df.loc[inclusive:inclusive]
df.loc['a':'d']#Getting row with index a to d

# Note: .iloc is strictly integer position based.
# df.iloc[inclusive:exclusive]
df.iloc[0:3]#Get first 3 rows index is starting from 0

df['Rev']#Get one column
df[['Rev', 'test']]#Get data from two columns

# df.ix[rows,columns]
df.ix[0:3,'Rev']#Get data from Rev column for only 3 rows


df.ix[5:,'col']#Get data from col column of rows satrting from 5th to end

df.ix[:3,['col', 'test']]#Get 3 rows data of two columns

#brief look at the stack and unstack functions.******************************************************
# Our small data set
d = {'one':[1,1],'two':[2,2]}#Create a dictionary
i = ['a','b']
# Create dataframe
df = pd.DataFrame(data = d, index = i)
df

df.index
# Bring the columns and place them in the index
stack = df.stack()
stack
# The index now includes the column names
stack.index
unstack = df.unstack()
unstack
unstack.index
#Transpose the data frame
transpose = df.T
transpose


 #groupby function.************************************************************************************
# Our small data set
d = {'one':[1,1,1,1,1],
     'two':[2,2,2,2,2],
     'letter':['a','a','b','b','c']}

# Create dataframe
df = pd.DataFrame(d)
df

# Create group object
one = df.groupby('letter')#Group the column letter
one
one.sum()#Get sum of all numeric columns
one.mean()#Get average of all numeric columns
#Aggregate the data based on two columns
letterone = df.groupby(['letter','one']).sum()#sum 
letterone
letterone.index#Get the index of group by column
#You may want to not have the columns you are grouping by become your index, this can be easily achieved as shown below.
letterone = df.groupby(['letter','one'], as_index=False).sum()
letterone
letterone.index


#***************************sorting the data
# Method 1:
Sorted = df.sort_values(['Births'], ascending=False)
Sorted.head(1)
# Method 2:
df['Births'].max()

#***************************Outliear

import pandas as pd
import sys

# Create a dataframe with dates as your index
States = ['NY', 'NY', 'NY', 'NY', 'FL', 'FL', 'GA', 'GA', 'FL', 'FL'] 
data = [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
idx = pd.date_range('1/1/2012', periods=10, freq='MS')
df1 = pd.DataFrame(data, index=idx, columns=['Revenue'])
df1['State'] = States

# Create a second dataframe
data2 = [10.0, 10.0, 9, 9, 8, 8, 7, 7, 6, 6]
idx2 = pd.date_range('1/1/2013', periods=10, freq='MS')
df2 = pd.DataFrame(data2, index=idx2, columns=['Revenue'])
df2['State'] = States

# Combine dataframes
df = pd.concat([df1,df2])
df

# Method 1

# make a copy of original df
newdf = df.copy()

newdf['x-Mean'] = abs(newdf['Revenue'] - newdf['Revenue'].mean())
newdf['1.96*std'] = 1.96*newdf['Revenue'].std()  
newdf['Outlier'] = abs(newdf['Revenue'] - newdf['Revenue'].mean()) > 1.96*newdf['Revenue'].std()
newdf

# Method 2
# Group by item

# make a copy of original df
newdf = df.copy()

State = newdf.groupby('State')

newdf['Outlier'] = State.transform( lambda x: abs(x-x.mean()) > 1.96*x.std() )
newdf['x-Mean'] = State.transform( lambda x: abs(x-x.mean()) )
newdf['1.96*std'] = State.transform( lambda x: 1.96*x.std() )
newdf

# Method 2
# Group by multiple items

# make a copy of original df
newdf = df.copy()

StateMonth = newdf.groupby(['State', lambda x: x.month])

newdf['Outlier'] = StateMonth.transform( lambda x: abs(x-x.mean()) > 1.96*x.std() )
newdf['x-Mean'] = StateMonth.transform( lambda x: abs(x-x.mean()) )
newdf['1.96*std'] = StateMonth.transform( lambda x: 1.96*x.std() )
newdf


# Method 3
# Group by item

# make a copy of original df
newdf = df.copy()

State = newdf.groupby('State')

def s(group):
    group['x-Mean'] = abs(group['Revenue'] - group['Revenue'].mean())
    group['1.96*std'] = 1.96*group['Revenue'].std()  
    group['Outlier'] = abs(group['Revenue'] - group['Revenue'].mean()) > 1.96*group['Revenue'].std()
    return group

Newdf2 = State.apply(s)
Newdf2

# Method 3
# Group by multiple items

# make a copy of original df
newdf = df.copy()

StateMonth = newdf.groupby(['State', lambda x: x.month])

def s(group):
    group['x-Mean'] = abs(group['Revenue'] - group['Revenue'].mean())
    group['1.96*std'] = 1.96*group['Revenue'].std()  
    group['Outlier'] = abs(group['Revenue'] - group['Revenue'].mean()) > 1.96*group['Revenue'].std()
    return group

Newdf2 = StateMonth.apply(s)
Newdf2

# make a copy of original df
newdf = df.copy()

State = newdf.groupby('State')

newdf['Lower'] = State['Revenue'].transform( lambda x: x.quantile(q=.25) - (1.5*(x.quantile(q=.75)-x.quantile(q=.25))) )
newdf['Upper'] = State['Revenue'].transform( lambda x: x.quantile(q=.75) + (1.5*(x.quantile(q=.75)-x.quantile(q=.25))) )
newdf['Outlier'] = (newdf['Revenue'] < newdf['Lower']) | (newdf['Revenue'] > newdf['Upper']) 
newdf

#***************************************************************************************************************
#Using Pandas for Analyzing Data - Data Munging

#%matplotlib
import numpy as np
import pandas as pd
#Read the csv file of your choice using Pandas
ver=pd.read_csv("ver.csv")
#View the first few rows of the file and set the number of columns displayed. 
#Note: Using ver.head() will display the first five rows.
pd.set_option('display.max_columns', 80) 
ver.head(3)

#Determine the number of rows and columns in the dataset
ver.shape
#Find the number of rows in the dataset
len(ver)
#Get the names of the columns
ver.columns
#Get the first five rows of a column by name
ver['action_taken'][:5]

#reate categorical ranges for numerical data. Note that that you can specifiy the number of ranges you wish.
incomeranges = pd.cut(ver['applicant_income_000s'], 14)#WE have divided data in 14 bucket
incomeranges[:5]
#Look at the value counts in the ranges created above
pd.value_counts(incomeranges)
#Index into the first six columns of the first row
ver.ix[0,0:6]
#Order the data by specified column
ver['loan_amount_000s'].order()[:5]
#Sort by a column and that obtain a cross-section of that data
sorteddata = ver.sort(['loan_amount_000s'])
sorteddata.ix[:,0:6].head(3)
#Obtain the first three rows and first three columns of the sorted data
sorteddata.iloc[0:3,0:3]
#Obtain value counts of specifiec column
ver['action_taken_name'].value_counts()
#A way to obtain the datatype for every column
zip(ver.columns, [type(x) for x in ver.ix[0,:]])

#The Pandas way to obtain datatypes for every column
ver.dtypes
#Get the unique values for a column by name.
ver['county_name'].unique()
#Get a count of the unique values of a column
len(ver['county_name'].unique())
#Index into a column and get the first four rows
ver.ix[0:3,'preapproval_name']
#Obtain binary values, Get the true or false which are not matching with given criteria
ver.ix[0:3,'preapproval_name'] == "Preapproval was requested"

#**********************************************Grouping and Aggregating**************************************
#Using Pandas for Analyzing Data - Grouping and Aggregating

ver=pd.read_csv("ver.csv")
#Melt data
melt = pd.melt(ver, id_vars = 'loan_purpose_name')
#Obtain the first five rows of melted data
melt.iloc[0:5,:]
#Return descriptive statistics of the dataset
ver.describe()
#Crosstab of the data by specified columns
pd.crosstab(ver['county_name'],ver['action_taken_name'])

#Return a subset of the data
incomesubset = ver[(ver['applicant_income_000s'] > 0 ) & (ver['applicant_income_000s'] < 1000)]
incomesubset

#Query the data
qry1 = ver.query('(applicant_income_000s > 0) & (applicant_income_000s < 1000)') 
qry1.head(10)
#Group data and obtain the mean
grouped1 = ver.groupby(['applicant_race_name_1','loan_purpose_name']).mean()
grouped1

#Check a boolean condition
(ver.ix[:,'applicant_income_000s'] > 9000).any()#get true or false if any observation falling between criteria
#Get descriptive statistics for a specified column
ver.applicant_income_000s.describe()  
#Group data and obtain the mean values
grpagg = ver.groupby('purchaser_type_name').aggregate(np.mean)
grpagg

#Return boolean values for a specified criteria
criterion = ver['applicant_race_name_1'].map(lambda x: x.startswith('W'))
criterion.head()

#**************************************Visualization with Python**************************************
#Using Pandas for Analyzing Data - Visualization

%matplotlib inline
import numpy as np
import pandas as pd

#Read the csv file of your choice
ver=pd.read_csv("ver.csv")

#Plot counts of a specified column using Pandas
ver.loan_purpose_name.value_counts().plot(kind='barh')

#Bar plot of median values
ver.groupby('agency_abbr')['applicant_income_000s'].agg(np.median).plot(kind = 'bar')

#Box plot example

g = sns.factorplot("loan_purpose_name", "loan_amount_000s", "agency_abbr", ver, kind="box",                        
                   palette="PRGn",aspect=2.25)
g.set(ylim=(0, 600))

#Bar plot example
sns.factorplot("loan_purpose_name", data=ver, hue="agency_abbr",size=3,aspect=2)
#Another bar plot example
sns.factorplot("loan_purpose_name", "loan_amount_000s", data=ver, palette="BuPu_d")

#Violin plot example
sns.violinplot(ver["loan_amount_000s"], ver["loan_purpose_name"], color="BuPu_d").set_ylim(0, 800)
sns.despine(left=True);
#Regression plot
sns.regplot("logloanamt", "logincome", data=ver, robust=True, ci=95, color="seagreen")
sns.despine();
#Bar plot of median values
sns.factorplot("agency_abbr", "loan_amount_000s", data=ver, palette="PuBu_d", estimator=np.median);
#Bar plot example
sns.factorplot("loan_purpose_name", data=ver, hue="action_taken_name");


#***********************************time seriese********************************************************
#Analyzing Data with Pandas - Time Series
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from pandas_datareader import data, wb# You need to install if you do not have this package

#Read the data
yhoo = pd.DataReader("yhoo", "yahoo", datetime.datetime(2007, 1, 1),datetime.datetime(2012,1,1))

#Plot stock price and volume

top = plt.subplot2grid((4,4), (0, 0), rowspan=3, colspan=4)
top.plot(yhoo.index, yhoo["Close"])
plt.title('Yahoo Price from 2007 - 2012')

bottom = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4)
bottom.bar(yhoo.index, yhoo['Volume'])
plt.title('Yahoo Trading Volume')

plt.gcf().set_size_inches(15,8)

#Calculate moving averages

mavg = yhoo['30_MA_Open'] = pd.stats.moments.rolling_mean(yhoo['Open'], 30)
yhoo['30_MA_Open'].tail() 

#Look at selected rows
yhoo[160:165]
#Index into a particular date
yhoo.ix['2010-01-04']

#Look at volume for the time period
yhoo.Volume.plot()

#More plots
yhoo.plot(subplots = True, figsize = (8, 8));
plt.legend(loc = 'best')
plt.show()

close_px = yhoo['Adj Close']
mavg = pd.rolling_mean(close_px, 30)

#Moving average plot
yhoo.Close.plot(label='Yahoo')
mavg.plot(label='mavg')
plt.legend()
plt.gcf().set_size_inches(15,8)

#KDE plot

yhoo.Close.plot(kind='kde')


#*******************************************************************************************
#Generating data
#*******************************************************************************************
# Create a weekly (mondays) date range
        rng = pd.date_range(start='1/1/2009', end='12/31/2012', freq='W-MON')
        
# Create random data
        data = np.randint(low=25,high=1000,size=len(rng))
# Status pool
        status = [1,2,3]
# Make a random list of statuses
        random_status = [status[np.randint(low=0,high=len(status))] for i in range(len(rng))]
#matrix of random number
weights_h = [[random.random() for e in inputs[0]] for e in range(hiden_neurons)]

np.random.rand(2,3)
#get a random array of floats between 0 and 1 as Pavel mentioned 
W = np.random.random((L_out, L_in +1))

#normalize so that it spans a range of twice epsilon
W = W * 2 * .12

#shift so that mean is at zero
W = W - .12 
#For random numbers out of 10. For out of 20 we have to multiply by 20.
x = np.int_(np.random.rand(10) * 10)
#Generate random number by numpy
import numpy as np
matrix = np.random.choice([x for x in xrange(low,high,step)],rows*cols)
matrix.resize(rows,cols)
np.matrix(np.random.randint(22,37, size=(rows, cols)))



#Following are a list of libraries, you will need for any scientific computations and data analysis:
#
#NumPy 
#stands for Numerical Python. The most powerful feature of NumPy is n-dimensional array. This library 
#also contains basic linear algebra functions, Fourier 
#transforms,  advanced random number capabilities and tools for integration with other low level languages 
#like Fortran, C and C++

#SciPy 
#stands for Scientific Python. SciPy is built on NumPy. It is one of the most useful library for variety 
#of high level science and engineering modules like discrete Fourier transform, Linear Algebra, Optimization 
#and Sparse matrices.

#Matplotlib for plotting vast variety of graphs, starting from histograms to line plots to heat plots.. 
#You can use Pylab feature in ipython notebook (ipython notebook –pylab = inline) to use these plotting 
#features inline. If you ignore the inline option, then pylab converts ipython environment to an 
#environment, very similar to Matlab. You can also use Latex commands to add math to your plot.

#Pandas 
#for structured data operations and manipulations. It is extensively used for data munging and 
#preparation. Pandas were added relatively recently to Python and have been instrumental in 
#boosting Python’s usage in data scientist community.

#Scikit 
#Learn for machine learning. Built on NumPy, SciPy and matplotlib, this library 
#contains a lot of effiecient tools for machine learning and statistical modeling including 
#classification, regression, clustering and dimensionality reduction.

#Statsmodels 
#for statistical modeling. Statsmodels is a Python module that allows users to explore data, 
#estimate statistical models, and perform statistical tests. An extensive list of descriptive 
#statistics, statistical tests, plotting functions, and result statistics are available for different 
#types of data and each estimator.

#Seaborn
#for statistical data visualization. Seaborn is a library for making attractive and informative 
#statistical graphics in Python. It is based on matplotlib. Seaborn aims to make visualization a 
#central part of exploring and understanding data.

#Bokeh 
#for creating interactive plots, dashboards and data applications on modern web-browsers. 
#It empowers the user to generate elegant and concise graphics in the style of D3.js. 
#Moreover, it has the capability of high-performance interactivity over very large or streaming datasets.

#Blaze 
#for extending the capability of Numpy and Pandas to distributed and streaming datasets. 
#It can be used to access data from a multitude of sources including Bcolz, MongoDB, SQLAlchemy, 
#Apache Spark, PyTables, etc. Together with Bokeh, Blaze can act as a very powerful tool for creating 
#effective visualizations and dashboards on huge chunks of data.

#Scrapy 
#for web crawling. It is a very useful framework for getting specific patterns of data. 
#It has the capability to start at a website home url and then dig through web-pages within the 
#website to gather information.

#SymPy for symbolic computation. It has wide-ranging capabilities from basic symbolic arithmetic 
#to calculus, algebra, discrete mathematics and quantum physics. Another useful feature is the 
#capability of formatting the result of the computations as LaTeX code.
#Requests for accessing the web. It works similar to the the standard python library 

#urllib2 
#but is much easier to code. You will find subtle differences with urllib2 but for beginners, 
#Requests might be more convenient.
#Additional libraries, you might need:

#os for Operating system and file operations
#networkx and igraph for graph based data manipulations
#regular expressions for finding patterns in text data
#BeautifulSoup for scrapping web. It is inferior to Scrapy as it will extract 
#information from just a single webpage in a run.



