# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 10:51:41 2025

@author: gremi
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
flow1 = np.loadtxt("S1flow.txt", delimiter=',')
flow2 = np.loadtxt("S2flow.txt")

#Why have 0 in the last values ????
#last_index_1 = 2129 
#last_index_2 =7239

#flow1 = flow1[:last_index_1]
#flow2 = flow2[:last_index_2]


s1 = pd.Series(flow1)
s2 = pd.Series(flow2)

#%% Hydropower Technical Data

Hg1 = 350
dHg1 = 0.05 * Hg1
Hn1 = Hg1-dHg1

Hg2 = 500 
dHg2 = 0.05 * Hg2
Hn2 = Hg2-dHg2

outages = 0.05
eff = 0.9



#%% Electricity Tarif

# Base scenario
peak_b = 6 #Us Sc/kWh
off_b = 3 

# High demand
peak_h = 10
off_h = 5

#Times
t_peak = 8 #h/d
t_off = 16

period = 20 # [y]

opex = 0.02 # capex/year

#%% 

"""
Part1
"""



"""
a) Plot the last years of available data for the two stations S1 and S2 one vs the other to see if data are
correlated (fit a polynomial function of adequate degree) and use the correlation structure to fill in the missing
data. Neglect the noise;
"""

years = 365 * 3
# extract last year
lasty_f1 = flow1[-years:] 
lasty_f2 = flow2[-years:]

x = np.linspace(1,years,years) 

# Plot last year for both
plt.figure(figsize=(6, 3), dpi=300) 
plt.plot(x,lasty_f1, label = "flow 1")
plt.plot(x,lasty_f2, label = "flow 2")
plt.legend()
plt.show()

# Calculate Pearson correlation
correlation = np.corrcoef(lasty_f1, lasty_f2)[0, 1]
print("Pearson correlation:", correlation)
#%%

#Find relation between s1 and s2

#Step 1 : id if lag between s1 and s2
ind_max1 = np.argmax(lasty_f1)
ind_max2 = np.argmax(lasty_f2)
print(ind_max1)
print(ind_max2)

#Step 2 : fit a polynomial
n = 1 #degree
coef = np.polyfit(lasty_f2, lasty_f1, deg = n)
model = np.poly1d(coef) #test model

#Step 3 : id missing values in s1 and replace by model
ind_missing_s1_Start = 266
ind_missing_s1_end = 834

ddata = 834-266

ind_s2 = len(s2)-(len(s1)-834)

s1[ind_missing_s1_Start:ind_missing_s1_end] = model(s2[ind_s2-ddata:ind_s2])

#Step 4 : Plot to visualise
x = np.arange(1,ddata+1)

plt.figure(figsize=(6, 3), dpi=300) 
plt.plot(x,s1[ind_missing_s1_Start:ind_missing_s1_end])
plt.plot(x,s2[ind_s2-ddata:ind_s2])
plt.show()


#%% 
"""
b) Use the same correlation relationship to prolonge the S1 series to the same 
    length of the data serie in station 2. This will only be 20 years in total and
    we need to have at least 30 years. The swapping technique might be good, but we
    need to check if some temporal correlation affects the data before choosing the
    years to swap.
"""
#build S1 over 20 years
total_data_to_add = len(s2)-len(s1)

s1_20y = s2.copy()

s1_20y[:-len(s1)] = model(s2[:-len(s1)])
s1_20y[-len(s1):] = s1[:]


plt.figure(figsize=(6, 3), dpi=300) 
plt.plot(s1_20y)
plt.plot(s2)
plt.show()


#Annual Max serie
y = 20 
max_s1 = [np.nanmax(s1_20y[i*365:(i+1)*365]) for i in range(y)]
max_s1 = np.array(max_s1)

# Standardisation and autocorelated serie
mean_max = np.mean(max_s1)
std_max = np.std(max_s1)
z = (max_s1 - mean_max) / std_max

# Plot
plt.figure(figsize=(6, 3), dpi=300) 
plt.plot(range(1, y+1), z, marker='o')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Autocorrelated Serie")
plt.xlabel("Year")
plt.ylabel("Standardized Max Flow")
plt.show()


#%% Swapping Coding

#Selecting which years -> kind of random, just trying to mix by following trend
#such as fluctuating above and below the mean and mixing the intensity 

swap_y = [7,3,13,2,5,8,10,12,4,17]

s1_30y = s1_20y.copy()

for i in range(10):
    swap = s1_20y[swap_y[i] * 365 : (swap_y[i] + 1) * 365 ]  
    s1_30y = np.concatenate((s1_30y, swap))  

# Plot the new 30-year time series
plt.figure(figsize=(10, 4), dpi=300)
plt.plot(s1_30y, label='30-year series')
plt.xlabel('Days')
plt.ylabel('Flow')
plt.title('Flow 1 30 years swapping method')
plt.grid(True)
plt.legend()
plt.show()

#%%

"""
c) Build the flow duration curve of the 30 years dataset for reconstructed station 
    one data and calculate the reference minimal flow for instream flow protection 
    based on the Q 347 approach;
"""

#Weibull sorting
sortedQ = np.sort(s1_30y)[::-1]
rank = np.arange(1,len(sortedQ)+1)
prob = np.array([r / (len(sortedQ)+1) for r in rank]) 

plt.figure(figsize=(6, 3), dpi=300) 
plt.plot(prob, sortedQ)
plt.xlabel('Exceedance Probability [%]')
plt.ylabel('Flow [m³/s]')
plt.title('Flow Duration Curve')
plt.grid(True)
plt.show()

#Q347

ind = np.argmin(np.abs(prob - 0.95))# Find closest match index
print("Q347 is : ", sortedQ[ind])
print("Q347 prob. empirique: ",prob[ind])
print("Should be 0.95 otherwise need to make a curve fitting and find corresponding value for 0.95 !!!")


#%%%

"""
d) Obtain the daily mean annual behaviour from data;
"""
y = 30
daily_mean_annual = [np.mean(s1_30y[i*365:(i+1)*365]) for i in range(y)]
print(daily_mean_annual)


#%%%

"""
e) Build the monthly mean annual time series, which will be used for the 
    financial analysis;
"""

y = 30
month_days = [31,28,31,30,31,30,31,31,30,31,30,31]

monthly_mean = []

start = 0
for year in range(30):
    for i in month_days:
        month_s1 = s1_30y[start:start + i]
        monthly_mean.append(np.mean(month_s1))
        start += i

print(monthly_mean)

#!!! index 239 gives 0 -> related to year measured data where the last month was 0 Should we add it to the swap ??!!!
#%%
"""
f) Use the reconstructed data for S1 to build two new series as the sequence of
    wet periods and the sequence of dry periods. To the purpose, use the annual
    mean as discriminant value for the wet years (above the mean, Aug Dec) and
    viceversa. We will use these series later to build the Pareto frontier of 
    the system
"""

wet = []

dry = []


for j in range(len(daily_mean_annual)):
    for i in range(365):
        if s1_30y[i+ 365*j]  > daily_mean_annual[j]:
            wet.append(s1_30y[i+365*j])
        else:
            dry.append(s1_30y[i+365*j])

