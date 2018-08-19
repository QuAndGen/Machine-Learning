# -*- coding: utf-8 -*-
"""
Created on Mon Jul 03 21:55:17 2017

@author: ram
"""

#Ask the user for a number. Depending on whether the number is even or odd, 
#print out an appropriate message to the user. Hint: how does an even / odd number 
#react differently when divided by 2?
#
#Extras:
#
#If the number is a multiple of 4, print out a different message.
#Ask the user for two numbers: one number to check (call it num) and one number to 
#divide by (check). If check divides evenly into num, tell that to the user. 
#If not, print a different appropriate message.


num=input('Enter you number:')
if float(num)%2==0:
    print("You have provided Even number")
else:
    print("This is odd number")


num=input('Enter you number:')
check=input('Enter you division:')
if float(num)%float(check)==0:
    print("Both numbers are divisible")
else:
    print("Both numbers are not divisible in first place.")

#****************************************************Ex3
#a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
#and write a program that prints out all the elements of the list that are less than 5.
#
#Extras:
#
#Instead of printing the elements one by one, make a new list that has all the elements 
#less than 5 from this list in it and print out this new list.
#Write this in one line of Python.
#Ask the user for a number and return a list that contains only elements from the original 
#list a that are smaller than that number given by the user.

a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

for i in range(0,len(a),1):
    if i<5:
        print(i)

[x for x in a if x<5]

num=input('Enter your number:')

[x for x in a if x<int(num)]

#****************************************************Ex4
#Create a program that asks the user for a number and then prints out a list of all 
#the divisors of that number. (If you don’t know what a divisor is, it is a number that 
#divides evenly into another number. For example, 13 is a divisor of 26 because 26 / 13 has 
#no remainder.)

num=input('Enter you number:')
check=input('Enter you division:')
if float(num)%float(check)==0:
    print("Both numbers are divisible")
else:
    print("Both numbers are not divisible in first place.")
    
#***********************************************Ex5
#Take two lists, say for example these two:
#
#  a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
#  b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
#and write a program that returns a list that contains only the elements that are 
#common between the lists (without duplicates). Make sure your program works on two 
#lists of different sizes.
#
#Extras:
#
#Randomly generate two lists to test this
#Write this in one line of Python (don’t worry if you can’t figure this out at this 
#point - we’ll get to it soon)

a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

set([x for x in a if x in b])

import random

a=[random.randint(1,100) for i in range(100)]
b=[random.randint(1,100) for i in range(100)]

set([x for x in a if x in b])

#**************************************Ex6
#Ask the user for a string and print out whether this string is a palindrome or not. 
#(A palindrome is a string that reads the same forwards and backwards.)

ch=input('Enter your text:')

ch='malyalam'
ch=list(ch)
j=1
for i in range(0,len(ch),1):
    if ch[i]==ch[-i-1]:
        j+=1

if j==len(ch)-1:
    print('equi')

#Let’s say I give you a list saved in a variable: a = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]. 
#Write one line of Python that takes this list a and makes a new list that has only the even 
#elements of this list in it.

a = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

[x for x in a if x%2==0]

years_of_birth = [1990, 1991, 1990, 1990, 1992, 1991]

[2017-age for age in years_of_birth]

print('Type Enter to quit')
user1=''
user2=''
    
while user1 !='enter' or user2 !='enter':
    user1=input('Enter your number:')
    user2=input('Enter you number:')
    if int(user1)==int(user2):
        print('You both have same idea')


#write a short python function is_multiple(n,m), that takes two integer values and returns True
#if n is a multiple of m, that is n=mi for some integer i, and False otherwise.:

def is_multiple(n=None,m=None):
    if n%m==0 or m%n==0:
        print('Both numbers are multiple')
    else:
        print('Both numbers are not multiple')


is_multiple(2,4)
#function to check even    
def is_even(n=None):
    if n%2==0:
        print('Number is Even!')
    else:
        print('Number is odd!')

is_even(4)
is_even()
is_even(3)

def minmax(lst):
    x,y=lst[1],lst[1]
    for i in range(len(lst)):
        if y<=lst[i]:
            y=lst[i]
        if x>=lst[i]:
            x=lst[i]
    return (x,y)

minmax([3,4,5,2,1])
minmax([-3,-4,-5,-2,-1])

#write a short python function that takes a postive integer n and returns the sum of the
#squares of all the positive integers smaller than n.
s=0
def n_sqr(x):
    s=x**2
    if x==0:
        return s
    else:
        s+=n_sqr(x-1)
    return s 

n_sqr(3)

#or 
def n_sqr(x):
    total=0
    for i in range(x+1):
        total+=i**2
    return total

n_sqr(3)
n_sqr(-7)

#write a short python function that takes a postive integer n and returns the sum of the
#squares of all the positive ODD integers smaller than n.

def n_sqr(x):
    total=0
    for i in range(x+1):
        if i%2!=0:
            total+=i**2
    return total

n_sqr(5)

x='ram singh'

#Python allows negative integers to be used as indices into a sequence, such as a sting.
#if string s has length n, and expression s[k] is used for index -n<=k<0, what is the equivalent:
#    index j>=0 such that s[j] references the same element.

#Ans:
#    Indexing works in both positive and negative direction. if pass any negative value, it
#    starts index from the end of array or string or list. Unlike postive index, negative
#    index start with -1 and end with negative sign of length of object
#    possive index starts with 0 to n-1. so for every k<0, there will be 0=<j<n
#    

#Ques
#generate list of [50,60,70,80]
#Ans
range(50,90,10)
#what parameter should be sent to range constructor , to produce a range with 
#values 8 6 4 2 0 -2 -4 -6 -8

range(-8,10,2)

range(-8,10,2)
generate a list [1,2,4,8,16,32,64,128,256]
[2**x for x in range(9)]


rd.choice([1,10])

rd.choice
x=[2,3,4,6,7]

Generate your own choice function
def my_choice(seq):
    import random as rd
    i=rd.randrange(0,len(x),1)
    return seq[i]

my_choice(x)

#reverse the list of element
lst=non empty lst
lst1=[]
    i from 0 to length(lst) 
        lst1[i]=[lst[(length(list)-i)*(-1)]

x1=[0]*len(x)
for i in range(0,len(x)):
    x1[i]=x[-i-1]
x1

#distinct number of pair from the list which product is odd
y=[1,2,3,5,7]
output=[]
for i in set(y):
    if len(output)==0:
        output.append(i)
    for j in output:
        if (i*j)%2!=0:
            if i not in output:
                output.append(i)

output
#another simple way is if you thing, mathamaticaly product of odd numbers only can be odd.
#so from the list if we just get list of odd elements, then result will always be product.

[x for x in y if x%2!=0]

#get distinct value from the list
output=[]
for i in y:
    if i not in output:
        output.append(i)
output

#C1.16----Page 52
def scale(lst,scl):
    for i in range(len(lst)):
        lst[i]*=scl
    return lst

scale([2,3,5,2],2)

#C1.17 

def scale(lst,scl):
    list1=[]
    for val in lst:
        list1.append(val*scl)
    return list1


#Generate the series  [0, 2, 6, 12, 20, 30, 42, 56, 72, 90]
x=[0]*10
y=[]
z=[0]*10
for i in range(1,len(x)):
    x[i]=x[i-1]+2
    y.append(x[i])
    for j in range(len(y)):
        z[i]+=y[j]

#Generate alphabet
list(map(chr, range(97, 123))) 
#or 
list(map(chr, range(ord('a'), ord('z')+1)))

#C1.20 shufle the element of a list
import random as rd

y=['a','b','c','d','e']

[rd.randint(0,len(y)) for i in range(len(y))]

output=[]
while len(output)<len(y):
    r=rd.randint(0,len(y)-1)
    if y[r] not in output:
        output.append(y[r])

output

#Read text file
import os
os.getcwd()
os.chdir("D:\\Ram\\Python\\Practice")
next=[]
f = open("month.txt")
ls=f.read()
a=ls.split('\n')

#dot product of two array
def dot_prod(a,b):
    c=[]
#    if len(a) !=len(b):
#        print('Length of the both array must be equal')
#        return
    for a1,b1 in zip(a,b):
        c.append(a1*b1)
    return c

dot_prod([3,4,5],[1,2])


#C1.23 error handing in out of range    
x1=[0]*5

for i in range(6):
    try:
        x1[i]=rd.randint(1,5)
    except(IndexError):
        print("IndexError:Don't try buffer overflow attacks in Python!")

#C1.24 Count of vowel
i=0
for x in 'ram singh':
    if x in  ['a','e','i','o','u']:
        i+=1

#REmove stop words
stop_word=", - _ + - ' ? % $ # @ ! ` ~ ^ * ( ) < > : ; | \ { } [ ]"
strings="Let's try, Mike."

for i in strings:
    if i in stop_word.split():
        print(i)
        strings.replace(str(i),'',1)

"'" in stop_word.split()

def equation(a,b,c):
    if a==b-c:
        print("a=b-c")
    elif a+b==c:
        print("a+b=c")
    else:
        print('No equation')
    return

equation(1,2,3)


#Recursion
#Draw a ruler


def draw_line(tick_length,tick_label=''):
    """draw one line with given tick length (followed by optional lablel)"""
    line='-'*tick_length
    if tick_label:
        line+=" "+tick_label
    print(line)

def draw_interval(center_length):
    """Draw tick interval based upon a central tick length"""
    if center_length>0:             #stop when length drops to 0
        draw_interval(center_length-1)
        print(center_length)
        draw_line(center_length)
        draw_interval(center_length-1)

def draw_ruler(num_inches,major_length):
    """Draw english ruler with given number of inches , major tick length"""
    draw_line(major_length,'0')         #draw inch 0 line
    for j in range(1,1+num_inches):
        draw_interval(major_length-1)
        draw_line(major_length,str(j))

draw_ruler(2,3)

draw_line(5,'0')

draw_interval(3)

#Binery search
array=sorted(array)
def binery_search(array,value):
    if len(array)==0:
        print('Value is not in the array')
    
    high=len(array)-1
    low=0
    mid=(low+high)//2
    if value>array[high] or value<array[low] or mid==low==high:
        print('Value is not in the array')
        return
    if value==array[mid]:
        print('Value is there in the data')
        return
    elif value<array[mid]:
        array=[i for i in array if i<=array[mid]]
    elif value>array[mid]:
        array=[i for i in array if i>=array[mid]]
    else:
        print('Value is not in the data set')
        return
    binery_search(array,value)
    return

x=[3,4,5,6,6,3,3.5]

binery_search(x,3.5)


array=sorted(x)
len(x)

mid

        
value=6
high=len(array)-1
low=0
if value>array[high]:
    print('Value is not in the array')
    return
mid=(low+high)//2
if value==array[mid]:
    print('Value is there in the data')
    return
elif value<array[mid]:
    array=[i for i in array if i<=array[mid]]
elif value>array[mid]:
    array=[i for i in array if i>=array[mid]]
else:
    print('Value is not in the data set')
    return
    
binery_search(array,value)
    

#Theano scan function

import numpy as np
import theano
import theano.tensor as T

X=T.vector('X')

def square(x):
    return x*x

outputs,updates=theano.scan(
        fn=square,
        sequences=X,
        n_steps=X.shape[0]
        )

square_op=theano.function(
        inputs=[X],
        outputs=outputs
        )
o_val=square_op(np.array([1,2,3,4,5]))

#create fibnonacci function

N=T.scalar('N')

def recurrence(n,fn_1,fn_2):
    return fn_1+fn_2,fn_1

outputs,updates=theano.scan(
        fn=recurrence,
        sequences=T.arange(N),
        outputs_info=[1.,1.]
        )

fib_fun=theano.function(
        inputs=[N],
        outputs=outputs
        )

oval=fib_fun(8)

