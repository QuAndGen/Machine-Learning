#Set

#A set is an unordered collection of zero or more immutable Python data objects. 
#Sets do not allow duplicates and are written as comma-delimited values enclosed 
#in curly braces. The empty set is represented by set(). Sets are heterogeneous, 
#and the collection can be assigned to a variable as below.

mySet1 = {3,6,"cat",4.5,False}
mySet2 = {3,'Hello',54,True,4.53}
#Membership
'cat' in mySet1

#length
len(mySet1)

#Returns a new set with all elements from both sets
mySet1 | mySet2
mySet1.union(mySet2)


#Returns a new set with only those elements common to both sets
mySet1 & mySet2
mySet1.intersection(mySet2)
#Returns a new set with all items from the first set not in second

mySet1 - mySet2
mySet1.difference(mySet2)

#Asks whether all elements of the first set are in the second
mySet3 = {3,6,"cat",4.5,False,344,6,3,4,34}
mySet1 = {3,6,"cat",4.5,False}
mySet1<= mySet3
mySet1.issubset(mySet3)

mySet1.add("house")
mySet1.remove(4.5)
mySet1.pop() #remove one element from left side
mySet1.clear()

#Dictionary
#           keys:value
phoneext={'david':1410,'brad':1137}

#get the keys of the dictionary
phoneext.keys()

#keys converted into list
list(phoneext.keys())
#get value
phoneext.values() #in python 2.6, the output is by default list
#get value converted into list
list(phoneext.values())
#get the item of diction
phoneext.items() #in python 2.6, the output is by default list
#get item of dictionary converted into list and element will be in tupple
list(phoneext.items())

#get the value associated with key name. If key is present, then value will be reurned
#otherwise 'None'
phoneext.get("kent")

#get the value associated with key name. If key is present, then value will be reurned
#otherwise 'No Entry'
phoneext.get("kent","NO ENTRY")




#Error Handling
import math

#anumber = int(input("Please enter an integer "))

anumber = input("Please enter an integer ")

try:
    print(math.sqrt(anumber))
except:
    print("Bad Value for square root")
    print("Using absolute numeric value instead")
    #print(math.sqrt(abs(anumber)))


#
anumber = input("Please enter an integer ")
if type(anumber)==str: #int, float
    raise RuntimeError("Texts are not allowed.")
elif anumber < 0:
    raise RuntimeError("You can't use a negative number")
else:
    print(math.sqrt(anumber))



class Fraction:

    def __init__(self,top,bottom):

        self.num = top
        self.den = bottom
    #def show(self):
    #    print (str(self.num)+"/"+str(self.den))
    def __str__(self):
        return str(self.num)+"/"+str(self.den)
    def __add__(self,otherfraction):
        newnum = self.num*otherfraction.den + self.den*otherfraction.num
        newden = self.den * otherfraction.den
        common = gcd(newnum,newden)
        return Fraction(newnum//common,newden//common)


f1=Fraction(4,5)

f1.show()

print(f1)

str(f1)

f1 = Fraction(1,4)
f2 = Fraction(1,2)

print(f1.__add__(f2))


#Overall summary of class

def gcd(m,n):
    while m%n != 0:
        oldm = m
        oldn = n

        m = oldn
        n = oldm%oldn
    return n

class Fraction:
     def __init__(self,top,bottom):
         self.num = top
         self.den = bottom

     def __str__(self):
         return str(self.num)+"/"+str(self.den)

     def show(self):
         print(self.num,"/",self.den)

     def __add__(self,otherfraction):
         newnum = self.num*otherfraction.den + \
                      self.den*otherfraction.num
         newden = self.den * otherfraction.den
         common = gcd(newnum,newden)
         return Fraction(newnum//common,newden//common)

     def __eq__(self, other):
         firstnum = self.num * other.den
         secondnum = other.num * self.den

         return firstnum == secondnum

x = Fraction(1,2)
y = Fraction(2,3)






#Gate

class LogicGate:

    def __init__(self,n):
        self.name = n
        self.output = None

    def getName(self):
        return self.name

    def getOutput(self):
        self.output = self.performGateLogic()
        return self.output


class BinaryGate(LogicGate):

    def __init__(self,n):
        LogicGate.__init__(self,n)

        self.pinA = None
        self.pinB = None

    def getPinA(self):
        if self.pinA == None:
            return int(input("Enter Pin A input for gate "+self.getName()+"-->"))
        else:
            return self.pinA.getFrom().getOutput()

    def getPinB(self):
        if self.pinB == None:
            return int(input("Enter Pin B input for gate "+self.getName()+"-->"))
        else:
            return self.pinB.getFrom().getOutput()

    def setNextPin(self,source):
        if self.pinA == None:
            self.pinA = source
        else:
            if self.pinB == None:
                self.pinB = source
            else:
                print("Cannot Connect: NO EMPTY PINS on this gate")


class AndGate(BinaryGate):

    def __init__(self,n):
        BinaryGate.__init__(self,n)

    def performGateLogic(self):

        a = self.getPinA()
        b = self.getPinB()
        if a==1 and b==1:
            return 1
        else:
            return 0

class OrGate(BinaryGate):

    def __init__(self,n):
        BinaryGate.__init__(self,n)

    def performGateLogic(self):

        a = self.getPinA()
        b = self.getPinB()
        if a ==1 or b==1:
            return 1
        else:
            return 0

class UnaryGate(LogicGate):

    def __init__(self,n):
        LogicGate.__init__(self,n)

        self.pin = None

    def getPin(self):
        if self.pin == None:
            return int(input("Enter Pin input for gate "+self.getName()+"-->"))
        else:
            return self.pin.getFrom().getOutput()

    def setNextPin(self,source):
        if self.pin == None:
            self.pin = source
        else:
            print("Cannot Connect: NO EMPTY PINS on this gate")


class NotGate(UnaryGate):

    def __init__(self,n):
        UnaryGate.__init__(self,n)

    def performGateLogic(self):
        if self.getPin():
            return 0
        else:
            return 1


class Connector:

    def __init__(self, fgate, tgate):
        self.fromgate = fgate
        self.togate = tgate

        tgate.setNextPin(self)

    def getFrom(self):
        return self.fromgate

    def getTo(self):
        return self.togate


def main():
   g1 = AndGate("G1")
   g2 = AndGate("G2")
   g3 = OrGate("G3")
   g4 = NotGate("G4")
   c1 = Connector(g1,g3)
   c2 = Connector(g2,g3)
   c3 = Connector(g3,g4)
   print(g4.getOutput())

main()


#Excercise

#Create a two new gate classes, one called NorGate the other called NandGate. 
#NandGates work like AndGates that have a Not attached to the output. 
#NorGates work lake OrGates that have a Not attached to the output.
#
#Create a series of gates that prove the following equality 
#NOT (( A and B) or (C and D)) is that same as NOT( A and B ) and NOT (C and D). 
#Make sure to use some of your new gates in the simulation.




#Excercise

#Construct a class hierarchy for people on a college campus. Include faculty, staff, 
#    and students. What do they have in common? What distinguishes them from one another?
#Construct a class hierarchy for bank accounts.
#Construct a class hierarchy for different types of computers.
#Using the classes provided in the chapter, interactively construct a circuit and 
#    test it.


   
    
        
        
#Programming Exercises

#Implement the simple methods getNum and getDen that will return the 
#numerator and denominator of a fraction.

class number:
    def __init__(self,obj):
        self.num=obj.split('/',2)[0]
        self.den=obj.split('/',2)[1]
    
    def getNum(self):
        return(int(self.num))
    
    def getden(self):
        return(int(self.den))



a=number('5/4')


a.getden()
a.getNum()


#Ex 2
#In many ways it would be better if all fractions were maintained in lowest terms 
#right from the start. Modify the constructor for the Fraction class so that GCD is 
#used to reduce fractions immediately. Notice that this means the __add__ function no 
#longer needs to reduce. Make the necessary modifications.




def gcd(m,n):
    while m%n != 0:
        oldm = m
        oldn = n

        m = oldn
        n = oldm%oldn
    return n


class Fraction:

    def __init__(self,top,bottom):
        common = gcd(top,bottom)
        self.num = top//common
        self.den = bottom//common
        
    #def show(self):
    #    print (str(self.num)+"/"+str(self.den))
    def __str__(self):
         return (str(self.num)+"/"+str(self.den))
    def __add__(self,otherfraction):
        newnum = self.num*otherfraction.den + self.den*otherfraction.num
        newden = self.den * otherfraction.den
        #common = gcd(newnum,newden)
        return Fraction(newnum,newden)#Fraction(newnum//common,newden//common)


f1=Fraction(4,10)
f2=Fraction(3,6)
a=f1.__add__(f2)
a.__str__()


#Excercise 3
#Implement the remaining simple arithmetic operators (__sub__, __mul__, and __truediv__)



class Fraction:

    def __init__(self,top,bottom):
        common = gcd(top,bottom)
        self.num = top//common
        self.den = bottom//common
        
    #def show(self):
    #    print (str(self.num)+"/"+str(self.den))
    def __str__(self):
         return (str(self.num)+"/"+str(self.den))
    def __add__(self,otherfraction):
        newnum = self.num*otherfraction.den + self.den*otherfraction.num
        newden = self.den * otherfraction.den
        #common = gcd(newnum,newden)
        return Fraction(newnum,newden)#Fraction(newnum//common,newden//common)

    def __sub__(self,otherfraction):
        newnum = self.num*otherfraction.den - self.den*otherfraction.num
        newden = self.den * otherfraction.den
        #common = gcd(newnum,newden)
        return Fraction(newnum,newden)
    def __mul__(self,otherfraction):
        newnum = self.num*otherfraction.num
        newden = self.den * otherfraction.den
        #common = gcd(newnum,newden)
        return Fraction(newnum,newden)
    def div(self,otherfraction): #we can give any name to function name inside of class
        newnum = self.num*otherfraction.den
        newden = self.den * otherfraction.num
        #common = gcd(newnum,newden)
        return Fraction(newnum,newden)

f1=Fraction(4,10)
f2=Fraction(3,6)

#substraction of two fraction
a=f1.__sub__(f2)
a.__str__()
#multiplication of two fraction
a=f1.__mul__(f2)
a.__str__()
#division of two fraction
a=f1.div(f2)
a.__str__()

#Implement the remaining relational operators (__gt__, __ge__, __lt__, __le__, and __ne__)


class Fraction:

    def __init__(self,top,bottom):
        common = gcd(top,bottom)
        self.num = top//common
        self.den = bottom//common
        
    #def show(self):
    #    print (str(self.num)+"/"+str(self.den))
    def __str__(self):
         return (str(self.num)+"/"+str(self.den))
    def __add__(self,otherfraction):
        newnum = self.num*otherfraction.den + self.den*otherfraction.num
        newden = self.den * otherfraction.den
        #common = gcd(newnum,newden)
        return Fraction(newnum,newden)#Fraction(newnum//common,newden//common)

    def __sub__(self,otherfraction):
        newnum = self.num*otherfraction.den - self.den*otherfraction.num
        newden = self.den * otherfraction.den
        #common = gcd(newnum,newden)
        return Fraction(newnum,newden) #we call this class again so that it will have new value as initial
    def __mul__(self,otherfraction):
        newnum = self.num*otherfraction.num
        newden = self.den * otherfraction.den
        #common = gcd(newnum,newden)
        return Fraction(newnum,newden)
    def div(self,otherfraction): #we can give any name to function name inside of class
        newnum = self.num*otherfraction.den
        newden = self.den * otherfraction.num
        #common = gcd(newnum,newden)
        return Fraction(newnum,newden)
    def __gt__(self,otherfraction): #we can give any name to function name inside of class
        newnum1 = self.num*otherfraction.den
        newden1 = self.den*otherfraction.den
        newnum2 = self.den*otherfraction.num
        newden2 = self.den*otherfraction.den
        if newnum1>newnum2:
            return (str(self.num)+"/"+str(self.den)+ ' is greater than '+str(otherfraction.num)+"/"+str(otherfraction.den))
        else:
            return (str(otherfraction.num)+"/"+str(otherfraction.den)+ ' is greater than '+str(self.num)+"/"+str(self.den))
    def __ge__(self,otherfraction): #we can give any name to function name inside of class
        newnum1 = self.num*otherfraction.den
        newden1 = self.den*otherfraction.den
        newnum2 = self.den*otherfraction.num
        newden2 = self.den*otherfraction.den
        if newnum1>=newnum2:
            return (str(self.num)+"/"+str(self.den)+ ' is greater than or equal to '+str(otherfraction.num)+"/"+str(otherfraction.den))
        else:
            return (str(otherfraction.num)+"/"+str(otherfraction.den)+ ' is greater than or equal to '+str(self.num)+"/"+str(self.den))



f1=Fraction(4,10)
f2=Fraction(6,15)

f2.__ge__(f1)

#Ex 5
#Modify the constructor for the fraction class so that it checks to make sure that 
#the numerator and denominator are both integers. If either is not an integer the 
#constructor should raise an exception.

class Fraction:

    def __init__(self,top,bottom):
        import numpy as np
        if (type(top)!=int or type(bottom)!=int):
            raise RuntimeError("Only integers are allowed.")
        self.common = abs(gcd(top,bottom))
        self.num = (abs(top)//self.common)*(np.sign(top)*np.sign(bottom))
        self.den = abs(bottom//self.common)
        
    #def show(self):
    #    print (str(self.num)+"/"+str(self.den))
    def __str__(self):
         return (str(self.num)+"/"+str(self.den))
    def __add__(self,otherfraction):
        newnum = self.num*otherfraction.den + self.den*otherfraction.num
        newden = self.den * otherfraction.den
        #common = gcd(newnum,newden)
        return Fraction(newnum,newden)#Fraction(newnum//common,newden//common)


f1=Fraction(2,3)
f2=Fraction(4,6)

f3=f2.__add__(f1)

f1.__str__()
f2.__str__()

f1.den
f1.num
f1.common


#Research the __radd__ method. How does it differ from __add__? 
#When is it used? Implement __radd__

class Addition:

    def __init__(self,num1):
        self.num1 = num1
        
    def show(self):
        print (self.num1)
  
    def __add__(self,otherfraction):
        self.num1 = self.num1+otherfraction.num1 
        return self
    def __radd__(self,otherfraction):
        self.num1 = self.num1+otherfraction.num1
        return self #cumulative sum of previous value


f1=Addition(2)
f2=Addition(4)

f3=f2.__add__(f1)

f1.show()
f2.show()
f3.show()


#**************************************************************************
#An Anagram Detection Example by running loop
#**************************************************************************



def string_found(str1,str2):

    list_s1=list(str1)
    list_s2=list(str2)

    pos1=0
    chek=0
    gotit=True
    while pos1<len(list_s1) and gotit:
        pos2=0
        found=False
        while pos2<len(list_s2) and not found:
            if list_s1[pos1]==list_s2[pos2]:
                found=True
            else:
                pos2+=1
                found=False
        if found==True:
            list_s1[pos1]=None
            chek+=1
        else:
            gotit=False
    
        pos1+=1
    if chek==len(str1):
        print('All characters are matching')
    elif chek>=1 and chek<len(str1) :
        print('Few characters are matching')
    else:
        print('None of the characters are matching')

#All characters are matching
s1='heart'
s2='earth'

string_found(s1,s2)

#Few characters are matching
s1='hearts'
s2='earth'

string_found(s1,s2)

#None of the characters are matching
s1='dfg'
s2='earth'

string_found(s1,s2)


#**************************************************************************
#An Anagram Detection Example by sorting method
#**************************************************************************


def string_found(str1,str2):

    list_s1=list(str1)
    list_s2=list(str2)
    
    list_s1.sort()
    list_s2.sort()
    

    pos=0
    chek=0
    
    print(list_s1)
    print(list_s2)
    
    while pos<min(len(list_s1),len(list_s2)):
        if list_s1[pos]==list_s2[pos]:
            chek+=1
        pos+=1
 
    if chek==len(str1):
        print('All characters are matching')
    elif chek>=1 and chek<len(str1) :
        print('Few characters are matching')
    else:
        print('None of the characters are matching')




#All characters are matching
s1='heart'
s2='earth'

string_found(s1,s2)

#Few characters are matching
s1='hearts'
s2='earth'

string_found(s1,s2)

#None of the characters are matching
s1='dfg'
s2='earth'

string_found(s1,s2)




#**************************************************************************
#An Anagram Detection Example by Count and Compare
#**************************************************************************


def string_found(str1,str2):
    c_1=[0]*26
    c_2=[0]*26
    
    
    for i in range(len(str1)):
        pos=ord(str1[i])-ord('a') #give unicode of lower case 'a' as 97
        c_1[pos]=c_1[pos]+1 #increase the count if value is coming multiple times
    
    for i in range(len(str2)):
        pos=ord(str2[i])-ord('a') #give unicode of lower case 'a' as 97
        c_2[pos]=c_2[pos]+1 #increase the count if value is coming multiple times
    
    chek=0
    pos=0
    while pos<len(c_1):
        if c_1[pos]==0 and c_2[pos]==0:
            pos+=1
        elif (c_1[pos]==c_2[pos]) and c_1[pos]!=0:
            chek+=1
            pos+=1
        else:
            pos+=1
        
 
    if chek==len(str1):
        print('All characters are matching')
    elif chek>=1 and chek<len(str1) :
        print('Few characters are matching')
    else:
        print('None of the characters are matching')




#All characters are matching
s1='heart'
s2='earth'

string_found(s1,s2)

#Few characters are matching
s1='hearts'
s2='earth'

string_found(s1,s2)

#None of the characters are matching
s1='dfg'
s2='earth'

string_found(s1,s2)

 

#**************************************************************************
#Big o estimation in python lilst and dictionary process
#**************************************************************************

import timeit
 
    
popzero = timeit.Timer("x.pop(0)",
                       "from __main__ import x")
popend = timeit.Timer("x.pop()",
                      "from __main__ import x")

x = list(range(2000000))
popzero.timeit(number=1000)
#4.8213560581207275

x = list(range(2000000))
popend.timeit(number=1000)
#0.0003161430358886719


popzero = timeit.Timer("x.pop(0)",
                "from __main__ import x")
popend = timeit.Timer("x.pop()",
               "from __main__ import x")
print("pop(0)   pop()")
for i in range(1000000,100000001,1000000):
    x = list(range(i))
    pt = popend.timeit(number=1000)
    x = list(range(i))
    pz = popzero.timeit(number=1000)
    print("%15.5f, %15.5f" %(pz,pt))


import timeit
import random

for i in range(10000,1000001,20000):
    t = timeit.Timer("random.randrange(%d) in x"%i,
                     "from __main__ import random,x")
    x = list(range(i))
    lst_time = t.timeit(number=1000)
    
    x = {j:None for j in range(i)}
    d_time = t.timeit(number=1000)
    print("%d,%10.3f,%10.3f" % (i, lst_time, d_time))
    
 
    
#**************************************************************************
#The Stack Abstract Data Type
#**************************************************************************

#Note, there is no data type as such for Stack so lets create class of it
    
class Stack:
    
    def __init__(self):
        self.item=[]
    
    def isEmpty(self):
        return self.item==[]
    
    def pop(self):
        return self.item.pop()
    
    
    def push(self,item):
        self.item.append(item)

    def peek(self):
        return self.item[len(self.item)-1]
    def size(self):
        return len(self.item)
    def show(self):
        return self.item
    def revstring(self):
        #return self.item.reverse()
        return self.item[::-1]
    

s=Stack()

s.isEmpty()

s.pop()

s.push(2)



s.push('hey')
s.show()

s.push(True)
s.show()

s.push(34534)
a=s.show()

s.peek()


s.revstring()


#**************************************************************************
#Simple Balanced Parentheses
#**************************************************************************

#(5+6)∗(7+8)/(4+3)


def parChecker(symbolString):
    s = Stack()
    balanced = True
    index = 0
    while index < len(symbolString) and balanced:
        symbol = symbolString[index]
        if symbol == "(":
            s.push(symbol)
        else:
            if s.isEmpty():
                balanced = False
            else:
                s.pop()

        index = index + 1

    if balanced and s.isEmpty():
        return True
    else:
        return False

print(parChecker('((()))'))
print(parChecker('(()'))



#**************************************************************************
#Balanced Symbols (A General Case)
#**************************************************************************
#{ { ( [ ] [ ] ) } ( ) }

from pythonds.basic.stack import Stack

def parChecker(symbolString):
    s = Stack()
    balanced = True
    index = 0
    while index < len(symbolString) and balanced:
        symbol = symbolString[index]
        if symbol in "([{":
            s.push(symbol)
        else:
            if s.isEmpty():
                balanced = False
            else:
                top = s.pop()
                if not matches(top,symbol):
                       balanced = False
        index = index + 1
    if balanced and s.isEmpty():
        return True
    else:
        return False

def matches(open,close):
    opens = "([{"
    closers = ")]}"
    return opens.index(open) == closers.index(close)


print(parChecker('{{([][])}()}'))
print(parChecker('[{()]'))



#this code is changing the place of two elements simultaneously
def reverse_in_place(lst):      # Declare a function
    size = len(lst)             # Get the length of the sequence
    hiindex = size - 1          #index will be one less than size, becuase it starts from 0
    its = size/2                # Number of iterations required
    for i in xrange(0, its):    # i is the low index pointer
        temp = lst[hiindex]     # assign last elements to temp variable
        lst[hiindex] = lst[i]   #assign first element to last index
        lst[i] = temp           #assign temp value to first index
        hiindex -= 1            # Decrement the high index pointer
    return(lst)




    
#**************************************************************************
#Converting Decimal Numbers to Binary Numbers
#**************************************************************************
#convert things in binnary, octa or hexa

def divid(val,base):
    digits = "0123456789ABCDEF"
    bins=Stack()
    
    while val>0:
        rem=val%base
        bins.push(rem)
        val=val//base
    bins_code=""
    while not bins.isEmpty():
        bins_code=bins_code+digits[bins.pop()]
    return(bins_code)
    

divid(25,2)
divid(25,8)
divid(256,16)
divid(26,26)



#**************************************************************************
#Infix, Prefix and Postfix Expressions
#**************************************************************************

#Infix Expression    Prefix Expression	Postfix Expression
#A + B * C + D	    + + A * BCD	    A B C * + D +
#(A + B) * (C + D)	* + A B + CD	    A B + C D + *
#A * B + C * D	   + * A B * C D	    A B * C D * +
#A + B + C + D	   + + + A B C D	    A B + C + D +


from pythonds.basic.stack import Stack

def infixToPostfix(infixexpr):
    prec = {}
    prec["*"] = 3
    prec["/"] = 3
    prec["+"] = 2
    prec["-"] = 2
    prec["("] = 1
    opStack = Stack()
    postfixList = []
    tokenList = infixexpr.split()

    for token in tokenList:
        if token in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or token in "0123456789":
            postfixList.append(token)
        elif token == '(':
            opStack.push(token)
        elif token == ')':
            topToken = opStack.pop()
            while topToken != '(':
                postfixList.append(topToken)
                topToken = opStack.pop()
        else:
            while (not opStack.isEmpty()) and \
               (prec[opStack.peek()] >= prec[token]):
                  postfixList.append(opStack.pop())
            opStack.push(token)

    while not opStack.isEmpty():
        postfixList.append(opStack.pop())
    return " ".join(postfixList)

print(infixToPostfix("A * B + C * D"))
print(infixToPostfix("( A + B ) * C - ( D - E ) * ( F + G )"))

      

    
    
#Postfix Evaluation
from pythonds.basic.stack import Stack

def postfixEval(postfixExpr):
    operandStack = Stack()
    tokenList = postfixExpr.split()

    for token in tokenList:
        if token in "0123456789":
            operandStack.push(int(token))
        else:
            operand2 = operandStack.pop()
            operand1 = operandStack.pop()
            result = doMath(token,operand1,operand2)
            operandStack.push(result)
    return operandStack.pop()

def doMath(op, op1, op2):
    if op == "*":
        return op1 * op2
    elif op == "/":
        return op1 / op2
    elif op == "+":
        return op1 + op2
    else:
        return op1 - op2

print(postfixEval('7 8 + 3 2 + /'))




#**************************************************************************
# Implementing a Queue in Python
#**************************************************************************

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
    

q=Queue()

q.enqueue(4)
q.enqueue('dog')
q.enqueue(True)
print(q.size())


#Simulation: Hot Potato
#In this game (see Figure 2) children line up in a circle and pass an item 
#from neighbor to neighbor as fast as they can. At a certain point in the game, 
#the action is stopped and the child who has the item (the potato) is removed from 
#the circle. Play continues until only one child is left.

#from pythonds.basic.queue import Queue

def hotPotato(namelist, num):
    simqueue = Queue() #taken from previous queue
    for name in namelist:
        simqueue.enqueue(name)

    while simqueue.size() > 1:
        for i in range(num):
            simqueue.enqueue(simqueue.dequeue())

        simqueue.dequeue()

    return simqueue.dequeue()

print(hotPotato(["Bill","David","Susan","Jane","Kent","Brad"],7))


#Simulation: Printing Tasks


from pythonds.basic.queue import Queue

import random

class Printer:
    def __init__(self, ppm):
        self.pagerate = ppm #page per minute
        self.currentTask = None #if there is any current task assigned to printer
        self.timeRemaining = 0 #remaining time

    def tick(self):
        if self.currentTask != None: #if there is task
            self.timeRemaining = self.timeRemaining - 1 #reduce the time by 1 second
            if self.timeRemaining <= 0: #if remaining time is less than zero or negative, empty the current task
                self.currentTask = None

    def busy(self):#check wether if task is already assigned to printer or not
        if self.currentTask != None: 
            return True
        else:
            return False

    def startNext(self,newtask): #assign new task
        self.currentTask = newtask
        self.timeRemaining = newtask.getPages() * 60/self.pagerate #get time to print, number of page * 60/page rate

class Task:
    def __init__(self,time):
        self.timestamp = time # total time to be taken
        self.pages = random.randrange(1,21) #number of random page

    def getStamp(self):
        return self.timestamp #print time

    def getPages(self):
        return self.pages #showw the number of page

    def waitTime(self, currenttime):
        return currenttime - self.timestamp #show the remaining time left


def simulation(numSeconds, pagesPerMinute):

    labprinter = Printer(pagesPerMinute) #page per minute
    printQueue = Queue() #assign a new queue
    waitingtimes = []

    for currentSecond in range(numSeconds): #run for each second in one hour

      if newPrintTask():
         task = Task(currentSecond)
         printQueue.enqueue(task)

      if (not labprinter.busy()) and (not printQueue.isEmpty()):
        nexttask = printQueue.dequeue()
        waitingtimes.append( nexttask.waitTime(currentSecond))
        labprinter.startNext(nexttask)

      labprinter.tick()

    averageWait=sum(waitingtimes)/len(waitingtimes)
    print("Average Wait %6.2f secs %3d tasks remaining."%(averageWait,printQueue.size()))

def newPrintTask():
    num = random.randrange(1,181)
    if num == 180:
        return True
    else:
        return False

for i in range(10):
    simulation(3600,5)



#Deque has can accpet entry and exit from both end
#Implementing a Deque in Python
    

class Deque:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def addFront(self, item):
        self.items.append(item)

    def addRear(self, item):
        self.items.insert(0,item)

    def removeFront(self):
        return self.items.pop()

    def removeRear(self):
        return self.items.pop(0)

    def size(self):
        return len(self.items)
    
    
    
 #from pythonds.basic.deque import Deque
# A palindrome is a string that reads the same forward and backward, for example, radar, toot, and madam.
def palchecker(aString):
    chardeque = Deque()

    for ch in aString:
        chardeque.addRear(ch)

    stillEqual = True

    while chardeque.size() > 1 and stillEqual:
        first = chardeque.removeFront()
        last = chardeque.removeRear()
        if first != last:
            stillEqual = False

    return stillEqual

print(palchecker("lsdkjfskf"))
print(palchecker("radar"))
