# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 21:44:18 2017

@author: ram
"""

import numpy
import theano.tensor as T
from theano import function


x=T.dscalar('x')

y=T.dscalar('y')

x

z=x+y

f=function([x,y],z)

f(2,3)


numpy.allclose(f(16.3, 12.1), 28.4)


z.eval({x:5,y:10})









z.eval({x:[[1,2],[3,4]],y:[[4,3],[6,4]]})




x = T.dmatrix('x')
x=T.dmatrix('x')

y = T.dmatrix('y')
y=T.dmatrix('y')
z = x + y
z=x+y

f = function([x, y], z)
f=function([x,y],z)


f([[1, 2], [3, 4]], [[10, 20], [30, 40]])

f([[1,2],[3,4]],[[4,3],[6,4]])


import theano

a=theano.tensor.vector()

out=a+a**10

f=theano.function([a],out)

f([1,2,3])
print(f([1,2,3]))


#generate logistic function with theano

x=T.matrix('x')

s=1/(1+T.exp(-x))
out=s

f=theano.function([x],out)



f([[3,2],[4,2]])

th=(1+T.tanh(x/2))/2
out=th
f([[3,2],[4,2]])


a,b=T.matrices('a','b')


m=T.mean([a,b],axis=1)
s=T.std(a,axis=1)
mx=T.max([a,b],axis=1)


f=theano.function([a,b],[m,s,mx])

a1=[[1,2,3],[4,2,4],[6,7,3]]

b1=[[3,4,4],[5,2,3],[3,1,2]]

f(a1,b1)


#passing default value to paramerter

x,y=T.dscalars('x','y')

z=x*y

f=theano.function([x,theano.In(y,value=2)],z)

f(2,4)


#passing default value and give a name to parameters
x,y,w=T.dscalars('x','y','w')

z=(x+y)*w


f=theano.function([x,theano.In(y,value=2),theano.In(w,value=2,name='name_w')],z)

f(2,4) #only x and y

f(2,4,5) #all


f(2,name_w=5) #only x and w

f(2,y=1) #only x and y with actual name of parameter


#use shared variable to 

st=theano.shared(0)
inc=T.iscalar('inc')

acum=theano.function([inc],st,updates=[(st,st+inc)]) #this will add st value with parameters
#to decrease it we can use - instead of +

acum(5) #whenever we will run this code , it will save all value in st by increasing it
acum(2) #whenever we will run this code , it will save all value in st by increasing it
acum(1) #whenever we will run this code , it will save all value in st by increasing it

st.get_value() # to get final value we can use it 5+2+1

st.set_value(-1) #reset the value at -1

acum(3) # add 3 to st which has value of -1 
st.get_value() # so the final value will be 3-1=2









































































