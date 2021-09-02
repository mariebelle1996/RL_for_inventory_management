import cvxpy as cp
import numpy as np
import numpy.random as nr
from Environment import *
import heapq


def generate_timeserie(param, length, lead_time=False):
    # Originally, this allowed lead times of zero
    if lead_time:
        return nr.poisson(param-1, length) + 1
    return nr.poisson(param, length)

class ModelPredictiveControl:
    def __init__(self, env, horizon=None):
        assert isinstance(env, inventoryProductEnv)
        self.env = env
        self.lead_times_ts = self.env.LT_ts.astype(np.int64)
        #if not (self.lead_times_ts > 0).all():
            #raise ValueError('Some lead times are zero or negative.')

        self.demand_ts = self.env.demand_ts.astype(np.int64)
        self.ordering_cost = self.env.basic_Co
        self.holding_cost = self.env.Ch
        self.penalty = self.env.Cp
        self.max_IL = self.env.max_IL

        self.check_initial_inventory()

        assert len(self.lead_times_ts) == len(self.demand_ts)

        if horizon is None:
            self.max_lead_time = np.max(self.lead_times_ts)
        else:
            self.max_lead_time = horizon

        self.orders = cp.Variable((self.max_lead_time,), integer=True) # These are projected future orders, to be used in MPC
        self.il_window = cp.Variable((self.max_lead_time,), integer=True)
        self.demand_window = cp.Parameter((self.max_lead_time,), integer=True)
        self.lead_time_coefficients = cp.Parameter((self.max_lead_time, self.max_lead_time), integer=True)
        self.current_inventory_and_orders = cp.Parameter((self.max_lead_time,), integer=True)
        self.current_inventory_and_orders.value = np.zeros(self.max_lead_time)
        self.current_inventory_and_orders.value[0] = self.env.IL0

        self.problem = self._create_problem()

        self.current_step = 0
        self.order_queue = [] # queue for orders. Tuples of the form (step_of_arrival, order_amount)


    def check_initial_inventory(self):
        """Checks to see if initial inventory is sufficent to last until first order can be received"""
        amt_req = self.demand_ts[:self.lead_times_ts[0]].sum()
        if self.env.IL0 < amt_req:
             print('Initial inventory {} insufficient to satisfy demand of {} '
                  'before first order can be received at step {}. '
                   'Setting initial inventory to {}'.format(self.env.IL0, amt_req, self.lead_times_ts[0], amt_req))
             self.env.IL0 = amt_req
        else:
            print('Initial inventory sufficient for initial demand')

    def reset(self):
        self.current_inventory_and_orders.value[0] = self.env.IL0
        self.current_step = 0
        self.order_queue = []
        self.problem = self._create_problem()


    def _create_problem(self):
        # objective = cp.Minimize(self.il_window[0])
        objective = cp.Minimize(cp.sum(self.il_window))
        constraints = [self._inventory_transition(), self.il_window >= 0,
                       # self.il_window <= self.max_IL,
                       self.orders >= 0]
        return cp.Problem(objective, constraints)

    def _inventory_transition(self):
        shift = np.roll(np.eye(self.max_lead_time), -1, axis=1)
        shift[:, -1] = 0
        return self.il_window == self.current_inventory_and_orders + shift @ self.il_window - \
               self.demand_window + self.lead_time_coefficients @ self.orders

    def set_and_solve(self, demand, lead_times):
        assert demand.shape == self.demand_window.shape
        assert len(lead_times) == self.max_lead_time

        self.demand_window.value = demand

        lead_times = lead_times.astype(np.int64)
        M = np.zeros(self.lead_time_coefficients.shape)
        for i, lt in enumerate(lead_times):
            try:
                M[i+lt, i] = 1
            except IndexError: # These orders are received outside of the horizon
                pass
        self.lead_time_coefficients.value = M

        self.problem.solve()
        return self.orders.value, self.il_window.value

    def compute_cost(self, inventory_level, order):
        """
        Compute the cost and orders/inventory level
        """
        #     def _calculate_cost(self):
        #
        #         return self.Co * self.Order + self.Ch * self.inventory_level + self.Cp * self.backlog
        return self.ordering_cost*order+self.holding_cost*inventory_level

    def use_inventory(self, demand):
        self.current_inventory_and_orders.value[0] -= demand
        assert self.current_inventory_and_orders.value[0] >= 0
        return self.current_inventory_and_orders.value[0]

    def receive_orders(self):
        received_orders = 0

        while len(self.order_queue) > 0 and self.order_queue[0][0] == self.current_step: # orders we recieve
            received_order = heapq.heappop(self.order_queue)
            received_orders += received_order[1]

        self.current_inventory_and_orders.value[0] += received_orders
        self.current_inventory_and_orders.value[1:] = 0

        if len(self.order_queue) > 0:
            for reception_time, order_amount in self.order_queue:
                assert reception_time > self.current_step
                if reception_time > self.current_step+self.max_lead_time:
                    continue
                else:
                    self.current_inventory_and_orders.value[reception_time-self.current_step] += order_amount

    def place_order(self, order_amount, lead_time, ledger):
        assert order_amount >= 0

        if order_amount > 0:
            new_order = (self.current_step + lead_time, order_amount) # (Order receive day, order amount)
            heapq.heappush(self.order_queue, new_order)

            ledger.append(Order(order_amount, self.current_step, self.current_step+lead_time))


    def run(self):
        """
        Call set and solve with windows and then compute costs
        """
        inventory_levels = [self.current_inventory_and_orders.value[0]]
        order_ledger = []
        costs = []

        n = len(self.demand_ts)- self.max_lead_time
        l = len(self.demand_ts)

        # for j in range(n):
        for j in range(l):
            self.current_step = j
            self.receive_orders()

            demand_window = self.demand_ts[j:j+self.max_lead_time]
            lead_time_window = self.lead_times_ts[j:j+self.max_lead_time]
            if j>n:
                demand_window = np.concatenate((demand_window, np.zeros(j-n, dtype=np.int64)))
                lead_time_window = np.concatenate((lead_time_window, np.ones(j-n, dtype=np.int64)))
            orders, inventory_forecast = self.set_and_solve(demand_window, lead_time_window)

            self.place_order(orders[0], lead_time_window[0], order_ledger)
            current_inventory = self.use_inventory(demand_window[0])
            cost = self.compute_cost(current_inventory, orders[0])
            inventory_levels.append(current_inventory)
            costs.append(cost)

        return inventory_levels, order_ledger, costs


class Order:
    def __init__(self, amount, time_placed, time_received):
        self.amount = amount
        self.time_placed = time_placed
        self.time_received = time_received

    def __repr__(self):
        return 'Order({}, {}, {})'.format(int(self.amount), self.time_placed, self.time_received)



   # env = inventoryProductEnv(product_name, IL0, max_IL, basic_Co, breakpoints, discounts, holding_cost, penalty, LT_ts, demand_ts, horizon, period, inflation_factor, discount_factor)
  #  mpc = ModelPredictiveControl(env)
 #   inventory_levels, order_ledger, costs = mpc.run()
    
