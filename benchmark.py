# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 08:55:02 2021

@author: ELIE
"""
from adaptive_MinMax import *
from MinMax import *
from Environment import *
from MPC import *

#inventory product parameters
product_name = "pump"
IL0 = 25
max_IL = 70
ordering_cost = 10 
holding_cost = 2
penalty = 50

#demand and lead time parameters
demand_param = 7
list_lead_time_param = [2, 4, 6]

#MinMax parameters
minimum = 3
maximum = 0.75*max_IL
min_default_order = int(maximum - minimum)

#training parameters
horizon = 200
n_episodes = 2000

#Plotting parameters
linewidth = 2

#plot poisson distribution for demand and lead time
x = np.arange(0, 50, 0.5)
demand_distribution = poisson_distribution(x, demand_param, loc = 0)

for lead_time_param in list_lead_time_param:
    
    #lead time poisson distribution
    LT_distribution = poisson_distribution(x, lead_time_param, loc = 0)
    
    #generate lead time and demand time series
    LT_ts =  generate_timeserie(lead_time_param, horizon)
    demand_ts =  generate_timeserie(demand_param, horizon)
    env = inventoryProductEnv(product_name, IL0, max_IL, ordering_cost, holding_cost, penalty, LT_ts, demand_ts, horizon)

    #Lists used for plotting
    list_s = np.array([])
    IL_ad_MinMax = np.array([])
    IL_MinMax = np.array([])
    backlog_MinMax = np.array([])
    backlog_ad_MinMax = np.array([])
    backlog_RL = np.array([])

    #Launch Minmax
    minmax = MinMax(env, minimum, maximum, min_default_order)

    for i in range(horizon):
        minmax.step()
        IL_MinMax = np.append(IL_MinMax, minmax.env.state[0])
        backlog_MinMax = np.append(backlog_MinMax, minmax.env.backlog)
                        
    score_minmax = sum(minmax.rewards)
    score_minmax = score_minmax*np.ones(n_episodes)
    
    backlog_MinMax = sum(backlog_MinMax)
    backlog_MinMax = backlog_MinMax*np.ones(n_episodes)
                    
    # Launch adaptive MinMax              
    ad_minmax = adaptive_MinMax(env, 0.85, 8)

    for i in range(horizon):
        ad_minmax.step()
        IL_ad_MinMax = np.append(IL_ad_MinMax, ad_minmax.env.state[0])
        list_s = np.append(list_s, ad_minmax.s)
        backlog_ad_MinMax = np.append(backlog_ad_MinMax, ad_minmax.env.backlog)

    score_ad_minmax = sum(ad_minmax.rewards)
    score_ad_minmax = score_ad_minmax*np.ones(n_episodes)
    
    backlog_ad_MinMax = sum(backlog_ad_MinMax)
    backlog_ad_MinMax = backlog_ad_MinMax*np.ones(n_episodes)

    #Launch DQN
    scores_dqn, action_list, backlog = dqn(n_episodes, env)
    backlog_RL = sum(backlog)
    backlog_RL = backlog_RL*np.ones(episodes)

    #Launch MPC
    np.random.seed(4)
    mpc = ModelPredictiveControl(env)
    inventory_levels, order_ledger, costs = mpc.run()
    score_mpc = np.sum(costs)
    #print('Total cost: {}'.format(np.sum(costs)))

    #Plotting the results
    
    #DQN vs MinMax(75% of max_IL, 3) vs MPC
    plot1 = plt.figure(1)
    plt.plot(-1*np.array(scores_dqn), label = "DQN", linewidth = linewidth)
    plt.plot(-1*score_minmax, label = "MinMax", linewidth = linewidth)
    plt.plot(-1*score_mpc, label = "MPC", linewidth = linewidth)
    plt.xlabel("Episode index")
    plt.ylabel("Cumulative cost per episode")
    title = 'IL= '+str(max_IL)+','+' demand=' + str(demand_param) +', lead time=' +str(lead_time_param)
    plt.title(title)
    plt.show()

    #DQN vs Adaptive MinMAx vs MPC
    plot2 = plt.figure(2)
    plt.plot(-1*np.array(scores_dqn), label = "DQN", linewidth = linewidth)
    plt.plot(-1*score_ad_minmax, label = "Adaptive MinMax", linewidth = linewidth)
    plt,plot(-1*score_mpc, label = "MPC", linewidth = linewidth)
    plt.xlabel("Episode index")
    plt.ylabel("Cumulative cost per episode")
    title = 'IL= '+str(max_IL)+','+' demand=' + str(demand_param) +', lead time=' +str(lead_time_param)
    plt.title(title)
    plt.show()
    
    #Inventory level fluctuations for Adaptive MinMax
    plot3 = plt.figure(3)
    plt.plot(IL_ad_MinMax, label = "IL", linewidth = linewidth)
    plt.plot(list_s, label = "ordering level", linewidth = linewidth)
    plt.xlabel("Episode index")
    plt.ylabel("Inventory level and ordering level")
    title = 'Adaptive MinMax: ' 'IL= '+str(max_IL)+','+' demand=' + str(demand_param) +', lead time=' +str(lead_time_param)
    plt.title(title)
    plt.show()
    
    #Inventory level fluctuations for MinMax
    plot4 = plt.figure(4)
    plt.plot(IL_MinMax, label = "IL", linewidth = linewidth)
    plt.plot(minimum, label = "ordering level", linewidth = linewidth)
    plt.xlabel("Episode index")
    plt.ylabel("Inventory level and ordering level")
    title = 'MinMax: ' 'IL= '+str(max_IL)+','+' demand=' + str(demand_param) +', lead time=' +str(lead_time_param)
    plt.title(title)
    plt.show()
    
    #Poisson distributions
    #Demand of parameter demand_parameter
    plot5 = plt.figure(5)
    plt.plot(x, demand_distribution)
    title = 'demand poisson distribution of parameter'+str(demand_param)
    plt.title(title)
    plt.show()
    
    #Lead time of parameter lead_time_param
    plot6 = plt.figure(6)
    plt.plot(x, LT_distribution)
    title = 'lead time poisson distribution of parameter'+str(lead_time_param)
    plt.title(title)
    plt.show()
    
    #Lost deamands
    plot7 = plt.figures(7)
    plt.plot(backlog_MinMax, label = "MinMax", linewidth = linewidth)
    plt.plot(backlog_ad_MinMax, label = "Adaptive MinMax", linewidth = linewidth)
    plt.plot(backlog_RL, label ="DQN", linewidth = linewidth )
    plt.xlabel("Episode index")
    plt.ylabel("Unsatisfied demands")
    title = 'Unsatisfied demands: ' 'IL= '+str(max_IL)+','+' demand=' + str(demand_param) +', lead time=' +str(lead_time_param)
    plt.title(title)
    plt.show()
    
    
    
    
        
    




