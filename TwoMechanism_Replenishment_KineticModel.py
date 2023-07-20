# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 12:50:42 2023

@author: hboigenzahn
"""


from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import PercentFormatter
import numpy.ma as ma
import copy

def system(t,y,r):
    #REACTIONS - ODE reaction model
    
    # r[1] = 2G -> GG, r[2] = GG -> 2G
    # r[3] = G + GG -> GGG, r[4] = GGG -> GG + G
    # r[5] = GGG+G -> GGGG, r[6] = GGGG -> GGG + G
    # r[7] = GG+GG -> GGGG, r[8] = GGGG -> GG + GG
        
    k1, k2, k3, k4, k5, k6, k7, k8 = r
    G, GG, GGG, GGGG = y
    
    dGdt = (-2*k1*G**2 + 2*k2*GG - k3*G*GG + k4*GGG - k5*G*GGG + k6*GGGG)
    dGGdt = (k1*G**2 - k2*GG - k3*G*GG + k4*GGG - 2*k7*GG**2 + 2*k8*GGGG)
    dGGGdt = (k3*G*GG - k4*GGG - k5*G*GGG + k6*GGGG)
    dGGGGdt = (k5*G*GGG - k6*GGGG + k7*GG**2 - k8*GGGG)
    
    dydt = ([dGdt, dGGdt, dGGGdt, dGGGGdt])
    
    return dydt

def rp(rprate, x, y0):
    # Dilute all species by the replenishment rate, and replace what was removed with y0
    # rprate is the rate of replenishment and should be between 0 and 1 (0 to 100%)
    # x is a list of species concentrations, y0 is the condition replenished with
    
    for i in range(len(x)):
        x[i] = x[i]*(1-rprate) +y0[i]*(rprate)

    return x

def runM1(y0,t,t_eval = None):
    # Run M1 for a certain amount of time, return trajectory
    # y0 is the starting point for the run, t is the length of time to run for, t_eval defaults to None (chosen by the solver) but can be included as an argument
    # Return type is ODE solution object from integate method
    t_span = [0,t]

    # Alternative parameters (different time splits)
    #params1 = [0.106567846,0.003076314,0.093334029,0.019825199,0.0200961,0.019471472,0.001927418,0.020211512] # 3hr/21hr split
    #params1 = [0.161112422,0.000721841,0.077270125,0.026491839,0.025428152,0.027385108,0.014202678,0] # 5hr/19hr split
    #params1 = [0.172435463,0.001336482,0.098712589,0.020717753,0.030157617,0.009792794,0.023530584,0] # 6hr/18hr split
    
    # Main parameters (4hr/20hr split)
    params1 = [0.213687752,0.00060626,1.478122977,3.423448854,1.602220534,0.042698427,0.007399654,0.386333157] # Mechanism 1

    sol = solve_ivp(system, t_span=t_span, y0=y0, args=[params1],t_eval=t_eval)
    return sol

def runM2(y0,t,t_eval = None):
    # Run M2 for a certain amount of time,  return the trajectory
    # y0 is the starting point for the run, t is the length of time to run for, t_eval defaults to None (chosen by the solver) but can be included as an argument
    # Return type is ODE solution object from integate method

    t_span = [0,t]
    
    # Alternative parameters (different time splits)
    #params2 = [0.0126628,0.012683823,0.096389275,0.098991464,0,0.005970893,0.215470945,0.037294522] # 3hr/21hr split
    #params2 = [0.017344648,0.017328939,0.059384002,0.052806273,0.01711695,7.36E-10,0.081925088,0.01507396] # 5hr/19hr split
    #params2 = [0.002519014,0.005627788,0.033456142,0.02607303,0.011698127,0,0.045056553,0] # 6hr/18hr split
    
    # Main parameters (4hr/20hr split)
    params2 = [0.010829901,0.012362116,0.10555719,0.101100497,0,0,0.366750216,0.077550106] # Mechanism 2   
    
    sol = solve_ivp(system, t_span=t_span, y0=y0, args=[params2],t_eval=t_eval)
    return sol

def scale(x, ub, lb, sf = 1):
    # Scale an array so all values fall between 0 and 1
    # x is the array to scale
    # ub is the upper bound to scale by
    # lb is the lower bound to scale by
    # sf is an optional scaling factor
    if ub-lb <= 1e-5:
        m = 0
        b = 0
    else:
        m = sf/(ub-lb)
        b = -lb*sf/(ub-lb)

    return m*x+b

def calcAlternatingTrajectory(y0, ctstart, ctend, rprate, totaltime, t_eval):
    # Calculate alternating trajectory of M1 and M2
    # y0 is the initial peptide concentrations, ctstart and ctend are the timings of the cycle
    # rprate is the replenishment rate as a fraction of 1
    # total time is how long to solve the overall system for
    # t_eval is how often the ODE is solved within each part of the cycle
    # 
    # Return alternating (array showing peptide trajectories), timing (x values for alternating),
    # and m1 highlight (y values for plotting over the top of just M1 to see which section is which if unclear)
    cycletime = [ctstart, ctend]
    cycles = int(2*totaltime/(cycletime[0]+cycletime[1])) 
    
    alternating = []
    m1highlight = []
    timing = []
    ycurrent = y0
    
    # Just going to do this with append because calculating the correct array size is annoying with variable times
    for i in range(cycles):
        if i%2 == 0: 
            sol = runM1(ycurrent,cycletime[0],t_eval = np.linspace(0,cycletime[0],t_eval))
            alternating.append(sol.y)
            m1highlight.append(sol.y) # save just the M1 values to highlight while plotting
            timing.append(sol.t)
            ycurrent = copy.deepcopy(alternating[-1][:,-1])
        else: 
            sol = runM2(ycurrent,cycletime[1],t_eval = np.linspace(0,cycletime[1],t_eval))
            alternating.append(sol.y)
            nans = np.empty(np.shape(sol.y))
            nans[:] = np.nan
            m1highlight.append(nans) # save just the M1 values to highlight while plotting
            timing.append(sol.t)
            ycurrent = copy.deepcopy(alternating[-1][:,-1])
            if rprate != 0:
                # Note that currently, the point of dilution (but before replenishment) is not explicitly included or plotted
                ycurrent = rp(rprate, ycurrent, y0)
                
                
    # Flatten lists of lists for easier plotting
    alternating = np.concatenate(alternating, axis = 1)
    m1highlight = np.concatenate(m1highlight, axis = 1)

    # append timing arrays together using cycletime information to create the correct x-axis spacing for plotting alternating
    for i, tlist in enumerate(timing):
        if i != 0:
            timing[i] = timing[i] + timing[i-1][-1]
 
    timing = np.concatenate(timing)
    
    return [alternating, timing, m1highlight]

# In[1]: Read experimental data (only used for plotting, not required to run the model)

df = pd.read_csv("Data/GlyIter_Results.csv")
df_sd = pd.read_csv("Data/GlyIter_StdDev.csv")
df.drop('G5',axis=1) # drop unnecessary axes from data 
df_sd.drop('G5',axis=1)     

df2 = pd.read_csv("Data/Gly_Drying_Results.csv")
df2_sd = pd.read_csv("Data/Gly_Drying_StdDev.csv")
df2.drop('G5',axis=1)
df2_sd.drop('G5',axis=1)  

# In[2]: Set the goal time for the model
totaltime = 10*24 # hrs
y0 = [0.1, 0, 0, 0] # initial conditions (M)

species = ['G','GG','G$_3$','G$_4$']

# replenishment rate - 0.75 means that 75% was removed and replaced
# This can also be customized per individual trajectory later
rprate = 0 

tspan = [0,totaltime]

# In[2]: Calculate the trajectories for M1 and M2

solM1 = runM1(y0, totaltime)
solM2 = runM2(y0, totaltime)

# In[3]: Calculate alternating trajectories

# These parameters can be modified for individual calculations, but I found it convenient to keep these as "defaults"
t_eval = 10
cycletime = [4, 20] # [Mechanism 1 time, Mechanism 2 time]

alternating, timing, m1highlight = calcAlternatingTrajectory(y0, cycletime[0], cycletime[1], 0, totaltime, t_eval)
# Additional trajectories can be added here with customized cycle times, replenishment percentages, etc.


# In[4]: Find the average of each cycle by numerical integration

RPavg = np.zeros((4,int(np.shape(alternating)[1] / (t_eval*2))))
for i in range(int(np.shape(alternating)[1] / (t_eval*2))):
    yvals = alternating[:,i*t_eval*2:(i+1)*t_eval*2]
    tvals = timing[i*t_eval*2:(i+1)*t_eval*2]
    RPavg[:,i] = np.trapz(yvals, x=tvals)/sum(cycletime)
    
# This is the end of the cyclic model -- everything else is analysis or plotting

# In[5]: Analyze 24 hour data to determine MSEs for various time splits
df_gly24 = pd.read_csv("Data/Gly24_Avgs.csv")
df_gly24sd = pd.read_csv("Data/Gly24_StdDevs.csv") 
df_gly24 = df_gly24.drop(labels=15, axis=0)
df_gly24sd = df_gly24sd.drop(labels=15, axis=0)

# Calculate 24 hour trajectory for M1 and M2
# Change these timings and the overall parameters for M1 and M2 to achieve different time splits
m1t = 4
m2t = 20
m1 = runM1(y0,m1t,t_eval=list(range(m1t+1)))
m2 = runM2(m1.y[:,-1],m2t,t_eval=list(range(m2t+1)))

# In[5.5]:
# Calculate the MSE of different timings
ytest = np.concatenate((m1.y, m2.y[:,1:]), axis=1).T[df_gly24['Hrs'].tolist()] # this very elaborate one-liner concatenates the theoretical data, then deletes the points not included in the experimental data so they can be compared by the MSE calculation
ytrue = df_gly24.iloc[:,2:6].to_numpy()

# MSE needs to be scaled or else G is the most important species by far.
# Experimental data will set upper and lower bound, and everything will be scaled relative to that term - I think that's how I've done it in the past
maxs = np.max(ytrue,axis=0)
mins = np.min(ytrue,axis=0)

ytest_scaled = np.zeros(np.shape(ytest))
ytrue_scaled = np.zeros(np.shape(ytrue))
for i, s in enumerate(species):
    ytest_scaled[:,i] = scale(ytest[:,i],maxs[i],mins[i])
    ytrue_scaled[:,i] = scale(ytrue[:,i],maxs[i],mins[i])


mse24 = ((ytrue-ytest)**2).mean(axis=None)
mse24_scaled = ((ytrue_scaled-ytest_scaled)**2).mean(axis=None)

# In[6]: Figure 2
fig, axs = plt.subplots(2,2)
plt.subplots_adjust(wspace=0.05,
                    hspace=0.2)

for i, s in enumerate(species):
    axs.flat[i].errorbar(df_gly24['Hrs'], df_gly24.iloc[:,i+2]*1000, yerr=df_gly24sd.iloc[:,i+2]*1000,fmt='o',capsize=3, label = "Experiments", c='black')
    axs.flat[i].plot(m1.t,m1.y[i]*1000,'-', label = "Mechanism 1", zorder = 3, c='r')
    axs.flat[i].plot(m2.t+m1t,m2.y[i]*1000,'-', label = "Mechanism 2", zorder = 3, c='b')
    axs.flat[i].title.set_text(s)
    
    axs.flat[i].set_xlim([0,24])
    axs.flat[i].set_xticks(np.arange(0,25,4))
    
plt.legend(loc='upper center', bbox_to_anchor=(-0.35, -0.35), ncol =3, frameon = False)    
plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
fig.text(0.5, 0.03, 'Time (hrs)', ha='center', va='center')
fig.text(0.03, 0.5, 'Concentration (mM)', ha='center', va='center', rotation='vertical')


# In[7]: Figure 4
fig, axd = plt.subplot_mosaic([['A', 'C','D'],
                                ['B', 'C', 'D']], gridspec_kw={'wspace': 0.3, 'hspace': 0.35}, figsize=(9, 5), dpi=300)

i = 0 # species index, because I'm not sure how enumerate works with dictionaries and I don't care right now
for k in sorted(axd):
    axd[k].plot(solM1.t,solM1.y[i]*1000,c='C0', label='Mechanism 1')
    axd[k].plot(solM2.t,solM2.y[i]*1000,c='C1', label='Mechanism 2')
    axd[k].plot(timing, alternating[i]*1000,'C2', label='Cyclic')
    axd[k].plot(np.linspace(start=12, stop=totaltime, num=np.shape(RPavg)[1]), RPavg[i,:]*1000,'r--',label='Cyclic average') # plot the moving average of 24 hour cycles (i.e., to show average steady state values)
    
    axd[k].title.set_text(species[i])
    axd[k].set_xlim([0,24*10])
    axd[k].set_xticks(np.arange(24,24*10+1,48))
    
    i = i+1 # increment species index

fig.text(0.5, 0.02, 'Time (hrs)', ha='center', va='center',size=11)
fig.text(0.07, 0.5, 'Concentration (mM)', ha='center', va='center', rotation='vertical',size=11)

axd['C'].legend(['Mechanism 2','Cyclic','Cyclic average'],loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4, frameon = False)    

plt.show()   

# In[8]: Plotting - Figure 5 & 6

# Run one figure at a time, or they will start plotting on top of each other. Just comment out the other figures.

fig, axs = plt.subplots(2,2)

for i, s in enumerate(species):
    
    # Figure 5 
    axs.flat[i].plot(timing, alternating[i]*1000,'C2', label='Cyclic')
    axs.flat[i].plot(np.insert(timing, 0, 0)[0::t_eval*2], np.insert(alternating, 0, y0, axis=1)[i][0::t_eval*2]*1000, 'k--x',label="Iterative Model") # plots through the points at the END of each M2 cycle -- useful for comparison with experimental data. Had to add an additional entry before calling slices to get the numbering right
    axs.flat[i].errorbar(df.Time*24,df.iloc[:,i+1]*1000,yerr=df_sd.iloc[:,i+1]*1000,capsize=3,fmt='k-o',label="Iterative Experiment")

    # Figure 6
    # axs.flat[i].errorbar(df.Time*24,df.iloc[:,i+1]*1000,yerr=df_sd.iloc[:,i+1]*1000,capsize=3,fmt='k-o',label="Iterative Experiment")
    # axs.flat[i].errorbar(df2.Time*24,df2.iloc[:,i+1]*1000,yerr=df2_sd.iloc[:,i+1]*1000,capsize=3,fmt='b-o',label="Drying Experiment")

    axs.flat[i].set_xlim([0,24*7])
    axs.flat[i].set_xticks(np.arange(0,24*7+1,24))
    axs.flat[i].margins(0.15)
    axs.flat[i].set_title(s,fontdict = {'fontsize':11})
    
plt.legend(loc='upper center', bbox_to_anchor=(-0.15, -0.4), ncol =3, frameon = False)    
plt.subplots_adjust(wspace = 0.3, hspace = 0.56)
fig.text(0.5, 0.02, 'Time (hrs)', ha='center', va='center',size=11)
#fig.text(0.04, 0.5, 'Concentration (mM)', ha='center', va='center', rotation='vertical')
fig.text(0.03, 0.5, 'Concentration Ratio', ha='center', va='center', rotation='vertical',size=11)

plt.show()   

# In[9]: Import relevant experimental data for Figure 7
df3 = pd.read_csv("Data/GlyRP_90_Results.csv")
df3_sd = pd.read_csv("Data/GlyRP_90_StdDev.csv")
df3.drop('G5',axis=1) # drop unnecessary axes from data 
df3_sd.drop('G5',axis=1)   

df4 = pd.read_csv("Data/GlyRP_75_Results.csv")
df4_sd = pd.read_csv("Data/GlyRP_75_StdDev.csv")
df4.drop('G5',axis=1) # drop unnecessary axes from data 
df4_sd.drop('G5',axis=1)

df5 = pd.read_csv("Data/GlyRP_50_Results.csv")
df5_sd = pd.read_csv("Data/GlyRP_50_StdDev.csv")
df5.drop('G5',axis=1) # drop unnecessary axes from data 
df5_sd.drop('G5',axis=1)    

# In[10]:
# Normalize experimental data for serial dilution RP results relative to the first day
#   to minimize experimental errors not actually the result of doing different treatments

df3_normed = df3[1:]/df3.iloc[1]
df3sd_normed = df3_sd[1:]/df3.iloc[1]
df4_normed = df4[1:]/df4.iloc[1]
df4sd_normed = df4_sd[1:]/df4.iloc[1]
df5_normed = df5[1:]/df5.iloc[1]
df5sd_normed = df5_sd[1:]/df5.iloc[1]

# In[11]: Figure 7

fig, axs = plt.subplots(2,2)

for i, s in enumerate(species):

    axs.flat[i].errorbar(df3_normed.Time*24,df3_normed.iloc[:,i+1],yerr=df3sd_normed.iloc[:,i+1],capsize=3,c='C0',label="90%")
    axs.flat[i].errorbar(df4_normed.Time*24,df4_normed.iloc[:,i+1],yerr=df4sd_normed.iloc[:,i+1],capsize=3,c='C1',label="75%")
    axs.flat[i].errorbar(df5_normed.Time*24,df5_normed.iloc[:,i+1],yerr=df5sd_normed.iloc[:,i+1],capsize=3,c='C2',label="50%")

    axs.flat[i].set_xlim([24,24*7])
    axs.flat[i].set_xticks(np.arange(24,24*7+1,48))
    axs.flat[i].margins(0.15)
    axs.flat[i].set_title(s,fontdict = {'fontsize':12})
    
plt.legend(loc='upper center', bbox_to_anchor=(-0.15, -0.45), ncol =3, frameon = False)    
plt.subplots_adjust(wspace = 0.3, hspace = 0.56)
fig.text(0.5, 0.01, 'Time (hrs)', ha='center', va='center',size=12)
fig.text(0.04, 0.5, 'Concentration Ratio', ha='center', va='center', rotation='vertical',size=11)

plt.show()   


    