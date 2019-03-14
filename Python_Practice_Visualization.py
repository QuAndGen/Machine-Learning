# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 19:34:40 2017

@author: ram
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')



train=pd.read_csv("D:/Ram/Python/Edu2code/Edu2code_Module1/linear_models/HOusing price/train.csv",header=0)


train.head()

#********************************Line chart****************************************************

plt.plot(train['YrSold'],train['SalePrice'])
plt.show()

year_by_data=train.groupby(by=['YrSold'],axis=0)['SalePrice','GarageArea','GrLivArea','TotalBsmtSF'].mean()

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(year_by_data.index,year_by_data.SalePrice,linewidth=5,color='k')#use blue=b, k=black, r=red etc
plt.title('Average price by year')
plt.xlabel('Year of price')
plt.ylabel('Average price in $')
plt.grid(True,color='r')#use blue=b, k=black, r=red etc


for xy in zip(year_by_data.index,year_by_data.SalePrice):                                       
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')


#*********************************************************************************
#use another parameters in line chart


plt.plot(year_by_data.index,year_by_data.SalePrice,color='k'
         ,alpha=.5,antialiased=True,linestyle='--',linewidth=1.0
         ,marker='+',markeredgewidth=10,markeredgecolor='g',markerfacecolor='k',markersize=5)
plt.show()

#another way to define parameters
plt.subplot(axisbg=(.3,.30,.30)) #adding background color 
# 0.75 a grayscale intensity (any float in [0,1]

lines , = plt.plot(year_by_data.index,year_by_data.SalePrice)
plt.setp( lines , markersize=15 , marker='d',markerfacecolor ='g' , markeredgecolor ='r')
plt.draw()

#another way to define parameters
line , = plt.plot(year_by_data.index,year_by_data.SalePrice)
line.set_markersize(15)
line. set_marker('d')
line.set_markerfacecolor('g')
line.set_markeredgecolor('r')
plt.draw()


#Property        Value
#alpha           The alpha transparency on 0-1 scale
#antialiased     True or False - use antialised rendering
#color           A matplotlib color arg
#data_clipping   Whether to use numeric to clip data
#label           A string optionally used for legend
#linestyle       One of - : -. -
#linewidth       A float, the line width in points
#marker          One of + , o . s v x > <, etc
#markeredgewidth The line width around the marker symbol
#markeredgecolor The edge color if a marker is used
#markerfacecolor The face color if a marker is used
#markersize      The size of the marker in points
#

#Abbreviation	Color
#b	          blue
#g	          green
#r	          red
#c	          cyan
#m	          magenta
#y	          yellow
#k	          black
#w	          white
#0.75        a grayscale intensity (any float in [0,1]
##2F4F4F     an RGB hex color string, eg, this example is dark slate gray
#(0.18, 0.31, 0.31) an RGB tuple; this is also dark slate gray
#red            any legal html color name





#Symbol	 Description
#-	solid line
#–	dashed line
#-.	dash-dot line
#:	dotted line
#.	points
#,	pixels
#o	circle symbols
#^	triangle up symbols
#v	triangle down symbols
#<	triangle left symbols
#>	triangle right symbols
#s	square symbols
#+	plus symbols
#x	cross symbols
#D	diamond symbols
#d	thin diamond symbols
#1	tripod down symbols
#2	tripod up symbols
#3	tripod left symbols
#4	tripod right symbols
#h	hexagon symbols
#H	rotated hexagon symbols
#p	pentagon symbols
#|	vertical line symbols
#_	horizontal line symbols
#steps	use gnuplot style ‘steps’ # kwarg only


#************************************************************************************
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(year_by_data.index,year_by_data.GrLivArea,label='Living Area')
plt.plot(year_by_data.index,year_by_data.TotalBsmtSF,label='Basment area')
plt.legend(loc='upper left')
plt.grid(True,color='r')#use blue=b, k=black, r=red etc

for xy in zip(year_by_data.index,year_by_data.GrLivArea):                                       
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

plt.show()

#************************************************************************************
plt.subplot(2, 1, 1)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(year_by_data.index,year_by_data.GrLivArea,label='Living Area')
plt.legend(loc='upper left')
plt.grid(True,color='r')#use blue=b, k=black, r=red etc

for xy in zip(year_by_data.index,year_by_data.GrLivArea):                                       
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')


plt.subplot(2, 1, 2)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(year_by_data.index,year_by_data.TotalBsmtSF,label='Basment area')
plt.legend(loc='upper left')
plt.grid(True,color='r')#use blue=b, k=black, r=red etc
for xy in zip(year_by_data.index,year_by_data.TotalBsmtSF):                                       
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

plt.show()

#****************************axis and figure*********************************************

#When plot is called, the pylab interface makes a call to gca() (“get current axes”) to get a 
#reference to the current
#axes. gca in turn, makes a call to gcf to get a reference to the current figure. gcf, finding
# that no figure has been
#created, creates the default figure figure() and returns it. gca will then return the 
#current axes of that figure if it
#exists, or create the default axes subplot(111) if it does not. Thus the code above is 
#equivalent to

plt.figure(num=None, #index of figure, we use number from 1 to n for different chart 
           figsize=None, #it is inches, and parameters are (width,height)
           dpi=None, #dpi is dot per inche, increase numbers to make more resolution
           facecolor=None, #determine the color of background
           edgecolor=None, #determine the color of border
           frameon=True #wether to draw the figure frame
           )
#we can swape the information regarding figure
plt.figure(1)
plt.plot(year_by_data.index,year_by_data.GrLivArea,label='Living Area')
plt.figure(2)
plt.plot(year_by_data.index,year_by_data.TotalBsmtSF,label='Basment area')
plt.title('Living area by year') #title of figure 2
plt.figure(1)
plt.title('basment area by year') #title of figure 1
plt.show()

#closing the chart box
plt.close(1) #use number to close the chart box, if any, this depend on figure number
plt.close('all') #use number to close the all chart box, if any

fig = plt.figure(1)
plt.setp(fig , facecolor ='g' , edgecolor ='r')
plt.plot(year_by_data.index,year_by_data.GrLivArea,label='Living Area')
plt.show()

#*******************************subplot******************************************************
plt.subplot(121)  #1 row block, 2 column, 1st cell
plt.plot([1,2,3],[1,2,3])
#plt.xlabel=[]

plt.subplot(122) #1 row , 2 column, 2nd cell
plt.plot([1,2,3],[1,4,9])
plt.show()

plt.subplot(211)  #2 rows, 1 column, 1st cell
plt.plot([1,2,3],[1,2,3])
#plt.xlabel=[] # we will force this lable so that both will have same

plt.subplot(212) #2 rows , 1 column, 1st cell
plt.plot([1,2,4],[1,4,9])
plt.show()

import numpy as np

dt=.001
t=np.arange(0,10,dt)
r=np.exp(-t[:1000]/0.05)
x=np.random.rand(len(t))
s=np.convolve(x,r,mode=2)[:len(x)]*dt

plt.figure(figsize=(7,5))
plt.plot(t,s,label='trend')
plt.axis([0,10,min(s),2*max(s)]) #[xmin, xmax, ymin, ymax]
#plt.xlim(0,10)
#plt.ylim(min(s),1.1*max(s))
plt.title('Gausian colored')
# this is an inset axes over the main axes
a=plt.axes([.65,.6,.2,.2],axisbg='y')
n,bins,patches=plt.hist(s,400,normed=1)
plt.title('probability')
plt.setp(a,xticks=[],yticks=[])

# this is an inset axes over the main axes

a=plt.axes([.2,0.6,.2,.2],axisbg='y')
plt.plot(t[:len(r)],r)
plt.title('impulse response')
plt.setp(a,xlim=(0,.2),xticks=[],yticks=[])

plt.show

#**********************************Text**************************************
xlabel(s) - add a label s to the x axis
ylabel(s) - add a label s to the y axis
title(s) - add a title s to the axes
text(x, y, s) - add text s to the axes at x, y in data coords
figtext(x, y, s) - add text to the figure at x, y in relative 0-1 figure coords

#As with lines, there are three ways to set text 
#properties: using keyword arguments to a text command, calling set on a text instance or a 
#sequence of text instances, or calling an instance
#method on a text instance. These three are illustrated below

#keywords args
plt.plot([14,25,14],[52,56,14])
plt.xlabel('time',color='r',size=15)
plt.ylabel('sales',color='r',size=15)
plt.title('Nice chart')
plt.show()

#use set method
plt.plot([14,25,14],[52,56,14])
locs,labels=plt.xticks() #setting of lable of axis
plt.setp(labels,color='g',rotation=45,size=10,style='italic')
plt.show()
#
#Property            Value
#alpha               The alpha transparency on 0-1 scale
#color               A matplotlib color arg
#family              set the font family, eg ’sans-serif’, ’cursive’, ’fantasy’
#fontangle           the font slant, one of ’normal’, ’italic’, ’oblique’
#horizontalalignment ’left’, ’right’ or ’center’
#multialignment      ’left’, ’right’ or ’center’ only for multiline strings
#name                the font name, eg, ’Sans’, ’Courier’, ’Helvetica’
#position            the x,y location
#variant             the font variant, eg ’normal’, ’small-caps’
#rotation            the angle in degrees for rotated text
#size                the fontsize in points, eg, 8, 10, 12
#style               the font style, one of ’normal’, ’italic’, ’oblique’
#text                set the text string itself
#verticalalignment   ’top’, ’bottom’ or ’center’
#weight              the font weight, eg ’normal’, ’bold’, ’heavy’, ’light’


#***********************annotations***************************************************

#argument            coordinate system
#'figure points'     points from the lower left corner of the figure
#'figure pixels'     pixels from the lower left corner of the figure
#'figure fraction'   0,0 is lower left of figure and 1,1 is upper, right
#'axes points'       points from lower left corner of axes
#'axes pixels'       pixels from lower left corner of axes
#'axes fraction'     0,1 is lower left of axes and 1,1 is upper right
#'data'              use the axes data coordinate system

import numpy as np

fig=plt.figure()
ax=fig.add_subplot(111,polar=True)
r=np.arange(0,1,0.001)
theta=2*2*np.pi*r

line,=plt.plot(theta,r,color='#ee8d18',lw=3)
ind=800
thisr,thistheta=r[ind],theta[ind]
plt.plot(thistheta,thisr,'o')
plt.annotate('a polar'
             ,xy=(thistheta,thisr) # theta , radius
             ,xytext=(.05,.05) #fraction ,fraction
             ,textcoords='figure fraction'
             ,arrowprops=dict(facecolor='black',shrink=.05)
             ,horizontalalignment='left'
             ,verticalalignment='bottom'
             )
plt.show()

#another example with anotation
plt.plot([4,5,6,3,4],[111,243,345,456,121])
plt.plot(6,345,'o')
plt.annotate('maximum value'
             ,xy=(6,345) # theta , radius
             ,xytext=(1,1) #fraction ,fraction #change this numbers between 0 to 1 for direction of arraow
             ,textcoords='figure fraction'
             ,arrowprops=dict(facecolor='black',shrink=.05)
             ,horizontalalignment='right'
             ,verticalalignment='top'
             )
plt.show()


#******************************Math text**********************************************

plt.plot([4,5,6,3,4],[111,243,345,456,121])
#plt.title('Alpha>beata')#plane text
#plt.title(r'$\alpha>\beta$')#math text
#plt.title(r'$\alpha_i>\beta^i$')#math text with subscript
#plt.text(5,243,r'$\sum_{i=0}^\infty x_i$') #adding summation
plt.text(6,345,r's(t)=$\cal{A}\rm{sin}(2\omega t)$')
plt.show()


s = r'$\cal{R}\prod_{i=\alpha}^\infty a_i\rm{sin}(2 \pi f x_i)$'
ax=plt.axes([.2,.2,1,1],axisbg='y')
plt.plot([4,5,6,3,4],[111,243,345,456,121])
plt.axis([0,10,100,500]) #[xmin, xmax, ymin, ymax]
plt.text(6,400,s) #this is not working becuase s is very large
plt.show()


#*******************************Image plot*****************************************
x=np.random.rand(20,20)
plt.imshow(x)

plt.imshow(X,  #array_like, shape (n, m) or (n, m, 3) or (n, m, 4)
                #Display the image in `X` to current axes.  `X` may be an
                #array or a PIL image. If `X` is an array, it
                #can have the following shapes and types:
                #- MxN -- values to be mapped (float or int)
                #- MxNx3 -- RGB (float or uint8)
                #- MxNx4 -- RGBA (float or uint8)
           cmap=None,  #the matplotlib.colors coloremap instance
           norm=None, #the normalization stat, normalize the pixels between 0 to 1
           aspect=None,  #['auto' | 'equal' | scalar], optional, default: None
                          #If 'auto', changes the image aspect ratio to match that of the
                                  #axes.
           interpolation=None, #the interploation method 'none', 'nearest', 'bilinear', 'bicubic',
                               #'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser',
                               #'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc',
                               #'lanczos'
           alpha=None, #the alpha transperancy value
           vmin=None, #the min for image scaling `vmin` and `vmax` are used in conjunction with norm to normalize
                       #luminance data.  Note if you pass a `norm` instance, your
                       #settings for `vmin` and `vmax` will be ignored.
           vmax=None, #the max for image scaling # use vmin and vmax to normalize the data between
           origin=None, #the origin of image  ['upper' | 'lower'], optional, default: None
                       #Place the [0,0] index of the array in the upper left or lower left
                         #corner of the axes. If None, default to rc `image.origin`.
           extent=None, #scalars (left, right, bottom, top), optional, default: None
                       #The location, in data-coordinates, of the lower-left and
                        #upper-right corners. If `None`, the image is positioned such that
                        #the pixel centers fall on zero-based (row, column) indices.
           shape=None, # scalars (columns, rows), optional, default: None For raw buffer images
           filternorm=1, #convert numbers into percentage, sum of pixels should be equal to 1
           filterrad=4.0, #scalar, optional, default: 4.0
                           #The filter radius for filters that have a radius parameter, i.e.
                           #when interpolation is one of: 'sinc', 'lanczos' or 'blackman'
           imlim=None, 
           resample=None, 
           url=None, 
           hold=None, 
           data=None, 
           **kwargs)



#******************Bar chart******************************************************************

Lot_by_price=train.groupby(by=['LotConfig'],axis=0)['SalePrice','GarageArea','GrLivArea','TotalBsmtSF'].mean()

xl=range(len(Lot_by_price.SalePrice))

fig = plt.figure(figsize=(10,10))
#plt.figure(figsize=(7,5))
ax = fig.add_subplot(211)
plt.bar(xl,Lot_by_price.SalePrice, label="Example two", color='g')
plt.xticks(xl,Lot_by_price.index)
plt.legend()
ax = fig.add_subplot(212)
plt.bar(xl,Lot_by_price.GarageArea, label="Example two", color='g')
plt.xticks(xl,Lot_by_price.index)
#for i, v in enumerate(Lot_by_price.index):
#    ax.text(v, i + .25, str(v), color='blue', fontweight='bold')
plt.legend()
plt.show()


#lets combin these charts in one chart ******************************************
width =.25 #width of chart
pos_1stbar=xl #position of first chart on x axis
pos_2dbar=[p + width for p in xl] #possition of 2nd chart on x axis, 
                                    #similarty for 3rd chart we can use p+width*2
plt.bar(pos_1stbar,Lot_by_price.GrLivArea, label="Example two", color='#F78F1E',width=width)
plt.bar(pos_2dbar,Lot_by_price.GarageArea, label="Example two", color='#FFC222',width=width)
#[p + 1.5 * width for p in xl]

plt.xlim(min(xl)-width, max(xl)+width*4) #where .25 is width of chart we cange it
plt.ylim([0, max(Lot_by_price.GrLivArea + Lot_by_price.GarageArea )])
plt.xticks([p + 1.5 * width for p in xl],Lot_by_price.index)
plt.legend(loc='upper left')
plt.show()

#lets combin these charts  in one chart and add error bar************************************


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

width =.25 #width of chart
pos_1stbar=xl #position of first chart on x axis
pos_2dbar=[p + width for p in xl] #possition of 2nd chart on x axis, 
                                    #similarty for 3rd chart we can use p+width*2

GrLivArea_std=[50,75,100,80,75] #this is standard deviation by each lot type
GarageArea_std=[80,50,70,60,40]#this is standard deviation by each lot type

bar1=plt.bar(pos_1stbar,Lot_by_price.GrLivArea, label="Example two", 
        color='#F78F1E',width=width, yerr=GrLivArea_std)
bar2=plt.bar(pos_2dbar,Lot_by_price.GarageArea, label="Example two"
        , color='#FFC222',width=width,yerr=GarageArea_std)

plt.xlim(min(xl)-width, max(xl)+width*4) #where .25 is width of chart we cange it
plt.ylim([0, max(Lot_by_price.GrLivArea + Lot_by_price.GarageArea )])
plt.xticks([p + 1.5 * width for p in xl],Lot_by_price.index)
plt.legend(loc='upper right')
autolabel(bar1)
autolabel(bar2)
plt.show()


#to make stacked bar chart, we can change the botom of second chart as end of first chart
width =.25 #width of chart

bar1=plt.bar(xl,Lot_by_price.GrLivArea, label="Example two", 
        color='#F78F1E',width=width)
bar2=plt.bar(xl,Lot_by_price.GarageArea, label="Example two"
        , color='#FFC222',width=width, bottom=Lot_by_price.GrLivArea)

plt.xlim(min(xl)-width, max(xl)+width*4) #where .25 is width of chart we cange it
plt.ylim([0, max(Lot_by_price.GrLivArea + Lot_by_price.GarageArea )*1.1])
plt.xticks([p + 1.5 * width for p in xl],Lot_by_price.index)
plt.legend(loc='upper right')
autolabel(bar1)
autolabel(bar2)
plt.show()


#Horizontal bar chart***********************************
bar1=plt.barh(xl,Lot_by_price.GrLivArea, label="Example two", 
        color='#F78F1E')
plt.yticks(xl,Lot_by_price.index)
plt.show()




#************************************************************************************
plt.hist(train.SalePrice, 10, normed=10, facecolor='green', alpha=0.25) 
#alpha is tranperancy of the color and 10 is number of bins
plt.show()

#*************Generate normal distribution curve***********************************************
# the histogram of the data
import matplotlib.mlab as mlab

n, bins, patches = plt.hist(train.SalePrice, bins=50, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
mu=train.SalePrice.mean()
sigma=train.SalePrice.std()
y = mlab.normpdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=1)
plt.xlabel('Salas Bin')
plt.ylabel('Probability')
plt.title('Histrograme with mu={} and sigma={}'.format(mu,sigma))
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

#***********************************Scater plot********************************************
plt.scatter(train.SalePrice,train.GarageArea, label='skitscat', color='b', 
            s=25, marker="o")
plt.show()
#Scatter plot with transparancy in color
plt.scatter(train.SalePrice,train.GarageArea, label='skitscat', color='b', 
            s=25, marker="o",alpha=.25) #added alpha to bring transparancy
plt.show()
#Scatter plot with size of bubble
plt.scatter(train.SalePrice,train.GarageArea, s=train.GrLivArea, color='b', alpha=0.5)
plt.show()

import numpy as np

color_labels = list(train.GarageFinish.unique())
color_code=list(np.random.rand(len(color_labels)))
# Map label to RGB
color_map = dict(zip(color_labels, color_code))

#Scatter plot with multiple level color
plt.scatter(train.SalePrice,train.GarageArea,c=train.GarageFinish.map(color_map))

plt.show()

#next question is how can we show legend of the color bar

#**************************************bubble
year_by_data=train.groupby(by=['YrSold'],axis=0)['SalePrice','GarageArea','GrLivArea','TotalBsmtSF'].mean()
color_labels = list(year_by_data.index.unique())
color_code=list(np.random.rand(len(color_labels)))
# Map label to RGB
color_map = dict(zip(color_labels, color_code))


x=list(year_by_data['GarageArea'])
y=list(year_by_data.SalePrice)
size=list(year_by_data.TotalBsmtSF)
tx=list(year_by_data.index)
color=np.random.rand(len(x))

plt.scatter(x,y,s=size,c=color)
plt.xlim(min(x)*.99,max(x)*1.05)
plt.ylim(min(y)*.99,max(y)*1.05)
for i in range(len(x)):
    plt.text(x[i], y[i],tx[i],size=5,horizontalalignment='center')
plt.show()

#************************************STack chart
year_by_data=train.groupby(by=['YrSold'],axis=0)['SalePrice','GarageArea','GrLivArea','TotalBsmtSF'].mean()

plt.stackplot(year_by_data.index, year_by_data.TotalBsmtSF
              ,year_by_data.GarageArea,year_by_data.GrLivArea, colors=['m','c','r'])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.show()
#in above chart we do not know the color of each bar

plt.plot([],[],color='m', label='Basment Area', linewidth=5)
plt.plot([],[],color='c', label='Garage Area', linewidth=5)
plt.plot([],[],color='r', label='Living Area', linewidth=5)

plt.stackplot(year_by_data.index, year_by_data.TotalBsmtSF
              ,year_by_data.GarageArea,year_by_data.GrLivArea, colors=['m','c','r'])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()

#************************************Pie Chart

year_by_data=train.groupby(by=['YrSold'],axis=0)['SalePrice','GarageArea','GrLivArea','TotalBsmtSF'].mean()


plt.pie(year_by_data.SalePrice,
        labels=year_by_data.index,
        colors=['g','r','b','k','y'],# we can type manual color as lenth of x
        startangle=45, #change angel to move in clock wise
        shadow= True, #use False if you do not want any shadow in chart print
        explode=(0,0.1,0,.2,0), # we can use value between 0 to 1 to slice out the pie
        autopct='%1.1f%%')
plt.axis('equal') #[xmin, xmax, ymin, ymax],'off','equal','scaled','tight','auto','normal','square'
plt.title('Interesting Graph\nCheck it out')
plt.show()



plt.pie(year_by_data.SalePrice,
        labels=year_by_data.index,
        colors=['g','r','b','k','y'],# we can type manual color as lenth of x
        startangle=45, #change angel to move in clock wise
        shadow= True, #use False if you do not want any shadow in chart print
        explode=(0,0.1,0,.2,0), # we can use value between 0 to 1 to slice out the pie
        autopct='%1.1f%%')

plt.title('Interesting Graph\nCheck it out')
plt.show()


patches, texts = plt.pie(year_by_data.SalePrice,
        labels=year_by_data.index,
        colors=['g','r','b','k','y'],# we can type manual color as lenth of x
        startangle=45, #change angel to move in clock wise
        shadow= True #use False if you do not want any shadow in chart print
        )
plt.legend(patches, year_by_data.index, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()

#**********************donut chart

# The slices will be ordered and plotted counter-clockwise.
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
explode = (0, 0, 0, 0)  # explode a slice if required

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True)
        
#draw a circle at the center of pie to make it look like a donut
centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.show()  

    
#********************************Area chart
year_by_data=train.groupby(by=['YrSold'],axis=0)['SalePrice','GarageArea','GrLivArea','TotalBsmtSF'].mean()

plt.stackplot(year_by_data.index,year_by_data.GrLivArea,year_by_data.GarageArea)
plt.show()

#show legend in area chart
x=year_by_data.index
y1=year_by_data.GrLivArea
y2=year_by_data.GarageArea

import matplotlib.patches as mpatches

plt.stackplot(x,[y1,y2], colors=['#377EB8','#55BA87'])
plt.legend([mpatches.Patch(color='#377EB8'),  
            mpatches.Patch(color='#55BA87')]
            ,['Living area','Garage Area'])
plt.show()

#*******************************Box plot

# fake up some data

#train[['YrSold','SalePrice','GarageArea','GrLivArea','TotalBsmtSF']]


# basic plot
plt.boxplot(train['SalePrice'])
plt.show()

# notched plot
plt.figure()
plt.boxplot(train['SalePrice'], 1)
plt.show()

# change outlier point symbols
plt.figure()
plt.boxplot(train['SalePrice'], 0, 'gD')
plt.show()

# don't show outlier points
plt.figure()
plt.boxplot(train['SalePrice'], 0, '')
plt.show()

# horizontal boxes
plt.figure()
plt.boxplot(train['SalePrice'], 0, 'rs', 0)
plt.show()

# change whisker length
plt.figure()
plt.boxplot(train['SalePrice'], 0, 'rs', 0, 0.75)
plt.show()

# multiple box plots on one figure
data=np.array(train[['SalePrice','GarageArea','GrLivArea','TotalBsmtSF']])

plt.figure()
plt.boxplot(data)
plt.xticks([1, 2, 3,4], ['Price', 'GarageArea', 'GrLivArea','TotalBsmtSF'])
plt.show()



#********************************************contour chart

import numpy as np

y=np.array(train.SalePrice)
x1=train.TotalBsmtSF
x2=train.GrLivArea
x3=train.GarageArea

X=np.array([x1,x2,x3]).reshape(1460, 3)
X.shape


import sklearn.linear_model as lm
len(y)
regr = lm.LinearRegression()
regr.fit(X,y)
y1=regr.predict(X)

X1,X2=np.meshgrid(x1, x2)

y3=np.array(10*(y1-y))
z = y3.reshape((len(x1), len(x1)))

plt.figure()
CS = plt.contour(X1, X2, y3)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Simplest default with labels')


#********************************Radar chart
cat = ['Speed', 'Reliability', 'Comfort', 'Safety', 'Effieciency']
values = [90, 60, 65, 70, 40]

N = len(cat)

x_as = [n / float(N) * 2 * np.pi for n in range(N)]

# Because our chart will be circular we need to append a copy of the first 
# value of each list at the end of each list with data
values += values[:1]
x_as += x_as[:1]


# Set color of axes
plt.rc('axes', linewidth=0.5, edgecolor="#888888")


# Create polar plot
ax = plt.subplot(111, polar=True)

# Set clockwise rotation. That is:
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Set position of y-labels
ax.set_rlabel_position(0)

# Set color and linestyle of grid
ax.xaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)
ax.yaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)


# Set number of radial axes and remove labels
plt.xticks(x_as[:-1], [])

# Set yticks
plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"])


# Plot data
ax.plot(x_as, values, linewidth=0, linestyle='solid', zorder=3)

# Fill area
ax.fill(x_as, values, 'b', alpha=0.3)
# Set axes limits
plt.ylim(0, 100)

# Draw ytick labels to make sure they fit properly
for i in range(N):
    angle_rad = i / float(N) * 2 * np.pi

    if angle_rad == 0:
        ha, distance_ax = "center", 10
    elif 0 < angle_rad < np.pi:
        ha, distance_ax = "left", 1
    elif angle_rad == np.pi:
        ha, distance_ax = "center", 1
    else:
        ha, distance_ax = "right", 1

    ax.text(angle_rad, 100 + distance_ax, cat[i], size=10, horizontalalignment=ha, verticalalignment="center")


# Show polar plot
plt.show()


#plot multiple chart from data set in one go
data=train[['SalePrice','GarageArea','GrLivArea','TotalBsmtSF','MSZoning','LandContour','Utilities','LotConfig']]

fig = plt.figure(figsize=(20,15))
cols = 5
rows = np.ceil(float(data.shape[1]) / cols)
for i, column in enumerate(data.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    plt.title(column)
    if data.dtypes[column] == np.object:
        data[column].value_counts().plot(kind="bar", axes=ax)
    else:
        data[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)


