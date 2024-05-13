import numpy as np
import ast
import time

from gurobipy import Model, GRB, tupledict
import gurobipy as gp
import json

class Instance():
    def __init__(self, regions, areas, region_area_map, area_map, period_demands, period_couriers,
                 region_employees, min_hours_worked, max_hours_worked, max_unique_starts,
                 shifts_start, shifts_end,
                 outsourcing_cost_multiplier=1.5, n_periods = 8, n_days = 7, n_hours = 8):
        
        self.regions = regions
        self.areas = areas
        self.region_area_map = region_area_map
        self.n_areas = len(areas)
        self.area_map = area_map
        self.n_periods = n_periods
        self.n_days = n_days
        self.n_hours = n_hours

        self.cost_courier = 1
        self.cout = outsourcing_cost_multiplier
        self.period_demands = period_demands
        self.period_couriers = period_couriers

        self.region_employees = region_employees
        self.n_employees = sum(value for reg, value in region_employees.items())

        self.min_hours_worked = min_hours_worked
        self.max_hours_worked = max_hours_worked
        self.max_unique_starts = max_unique_starts
        
        self.shifts_start = shifts_start
        self.shifts_end = shifts_end
        self.n_shifts = max([len(shifts_start[s]) for s in shifts_start])


class Solver():
    def __init__(self, i):
        self.i = i

    def define_parameters(self):
        # Regions
        self.R = sorted(self.i.regions)
        
        # Areas 
        self.A = {}
        for r in self.R:
            self.A[r] = [ self.i.area_map[a] for a in self.i.region_area_map[r] ]

        # Set of all shifts available
        # TODO: Shifts are regional level
        self.P = {}
        self.shifts = [s for s in range(self.i.n_shifts)]
        for r in self.R:
            self.P[r] = self.shifts

        # Periods
        self.Theta = [i for i in range(self.i.n_periods)]

        # Days
        self.D = [d for d in range(self.i.n_days)]

        # Hours in shift
        self.h = {}
        for p in range(self.i.n_shifts):
            self.h[p] = self.i.n_hours

        # Number of deliveries to perform (n_{a theta d})
        self.deliveries = np.zeros((self.i.n_areas, self.i.n_periods, self.i.n_days))
        for i, a in enumerate(self.i.areas):
            for k, d in enumerate(self.D):
                self.deliveries[i, :, k] = self.i.period_demands[(a, d)]

        # Number of couriers needed (m_{a theta d})
        self.couriers_needed = np.zeros((self.i.n_areas, self.i.n_periods, self.i.n_days))
        for i, a in enumerate(self.i.areas):
            for k, d in enumerate(self.D):
                self.couriers_needed[i, :, k] = self.i.period_couriers[(a, d)]

        # Cost of employed courier (c_{a theta d})
        self.cost_couriers = np.zeros((self.i.n_areas, self.i.n_periods, self.i.n_days))
        
        for i, a in enumerate(self.i.areas):
            for j, t in enumerate(self.Theta):
                for k, d in enumerate(self.D):
                    self.cost_couriers[i, j, k] = self.i.cost_courier

        # Employees
        self.E = {}
        prev_start = 0
        for r in self.R:
            self.E[r] = [prev_start+i for i in range(self.i.region_employees[r])]
            prev_start += len(self.E[r])

        self.employees = [e for e in range(self.i.n_employees)]

        # Min hours worked for employee e
        self.h_min = {e: self.i.min_hours_worked for e in self.employees}

        # Max hours worked for employee e
        self.h_max = {e: self.i.max_hours_worked for e in self.employees}

        # Max differing starts
        self.b_max = {e: self.i.max_unique_starts for e in self.employees}


    def solve(self, time_limit=2*60):
        self.define_parameters()

        start = time.time()

        # DECISION VARIABLES
        self.m = Model()

        # Shift for employee each day: r_{e p d}
        self.r_var = self.m.addVars(self.i.n_employees, self.i.n_shifts, self.i.n_days, vtype=GRB.BINARY, name='r')

        # Periods for each employee in each area: k_{e a theta d}
        self.k_var = self.m.addVars(self.i.n_employees, self.i.n_areas, self.i.n_periods, self.i.n_days, vtype=GRB.BINARY, name='k')

        # Different starts: U_{e p}
        self.u_var = self.m.addVars(self.i.n_employees, self.i.n_shifts, vtype=GRB.BINARY, name='u')

        # Outsourcing costs: omega_{a theta d}
        self.omega_var = self.m.addVars(self.i.n_areas, self.i.n_periods, self.i.n_days, vtype=GRB.CONTINUOUS, lb=0, name='omega')

        # mega_value
        M = 999_999

        # CONSTRAINTS
        # 1. Connecting employees moving around areas to shift assignment p
        for r in self.R:
            for e in self.E[r]:
                for d in self.D:
                    Ar, Pr = self.A[r], self.P[r]
                    self.m.addConstr(
                        (sum([self.k_var[e, a, theta, d] for a in Ar for theta in self.Theta]) == 1/2 * sum([self.h[p] * self.r_var[e,p,d] for p in Pr])),
                        name = f'moving_areas_{r}_{e}_{d}'
                    )

        # 2. Employee can only be assigned to one area at a time 
        for r in self.R:
            for e in self.E[r]:
                for theta in self.Theta:
                    for d in self.D:
                        Ar = self.A[r]
                        self.m.addConstr((sum([self.k_var[e, a, theta, d] for a in Ar]) <= 1),
                            name = f'employee_{r}_{e}_{theta}_{d}'          
                        )

        # Shift start and end times
        for r in self.R:
            shifts_ = self.i.shifts_start[r]
            for shift, start in shifts_.items():
                for a in self.A[r]:
                    for e in self.E[r]:
                        for theta in self.Theta[:start]:
                            for d in self.D:
                                self.m.addConstr((self.k_var[e, a, theta, d] * self.r_var[e, int(shift), d] == 0),
                                    name = f'start_time_{e}_{a}_{shift}_{d}'          
                                )

        for r in self.R:
            shifts_ = self.i.shifts_end[r]
            for shift, end in shifts_.items():
                for a in self.A[r]:
                    for e in self.E[r]:
                        for theta in self.Theta[end+1:]:
                            for d in self.D:
                                self.m.addConstr((self.k_var[e, a, theta, d] * self.r_var[e, int(shift), d] == 0),
                                    name = f'end_time_{e}_{a}_{shift}_{d}'          
                                )

        # 3. Employee can only work one shift a day
        for r in self.R:
            for e in self.E[r]:
                for d in self.D:
                    Pr = self.P[r]
                    self.m.addConstr((sum([self.r_var[e, p, d] for p in Pr]) <= 1),
                        name = f'only_one_shift_{r}_{e}_{theta}_{d}'          
                    )

        # 4. One rest day a week
        for r in self.R:
            for e in self.E[r]:
                Pr = self.P[r]
                self.m.addConstr((sum([self.r_var[e, p, d] for p in Pr for d in self.D]) <= 6),
                    name = f'rest_day_{r}_{e}'          
                )

        # 5. Min - Max hours worked per week
        for r in self.R:
            Pr = self.P[r]
            for e in self.E[r]:
                min_hours = self.h_min[e]
                max_hours = self.h_max[e]

                self.m.addConstr((sum([ self.h[p] * self.r_var[e, p, d] for p in Pr for d in self.D]) >= min_hours),
                    name = f'min_hours_{r}_{e}'          
                )

                self.m.addConstr(( sum([ self.h[p] * self.r_var[e, p, d] for p in Pr for d in self.D]) <= max_hours),
                    name = f'max_hours_{r}_{e}'          
                )

        # 6. Different shifting times constraint
        for r in self.R:
            for e in self.E[r]:
                for p in self.P[r]:
                    self.m.addConstr((sum([self.r_var[e, p, d] for d in self.D]) <= self.u_var[e, p] * M),
                        name = f'start_times_{r}_{e}_1'          
                    )

        for r in self.R:
            for e in self.E[r]:
                Pr = self.P[r]
                self.m.addConstr((sum([self.u_var[e, p] for p in Pr]) <= self.b_max[e]),
                    name = f'start_times_{r}_{e}_2'          
                )

        # 7. Employee can't work then outsource
        for r in self.R:
            for a in self.A[r]:
                for theta in self.Theta:
                    for d in self.D:
                        if self.couriers_needed[a,theta,d] > 0:
                            factor = self.deliveries[a, theta, d] / self.couriers_needed[a, theta, d] * self.i.cout 
                        else:
                            factor = 0
                        self.m.addConstr(
                            ((self.couriers_needed[a, theta, d] - sum([self.k_var[e, a, theta, d] for e in self.E[r]])) * factor <= self.omega_var[a, theta, d]),
                            name = f'outsource_{r}_{a}_{theta}_{d}'   
                        )

        # OBJECTIVE FUNCTION
        self.m.setObjective(
            sum([
                sum([self.cost_couriers[a, theta, d] * self.k_var[e, a, theta, d] for e in self.E[r]]) 
                    + self.omega_var[a, theta, d]
            for r in self.R for a in self.A[r] for theta in self.Theta for d in self.D] #+ n_employees
            )
        )

        #self.m.setParam('OutputFlag', 0) # No logs 
        self.m.setParam('TimeLimit', time_limit)
        self.m.ModelSense = GRB.MINIMIZE
        
        # SOLVE
        start_time = time.time()
        self.m.optimize()

        self.exec_time = time.time() - start_time


    def summary_results(self):
        
        # Determine feasibility based on the optimization status
        if self.m.status == GRB.Status.INFEASIBLE:
            obj_value, hiring, outsourcing = np.nan, np.nan, np.nan
        else:

            hiring = sum([ sum([self.cost_couriers[a, theta, d] * self.k_var[e, a, theta, d].X for e in self.E[r]]) 
            for r in self.R for a in self.A[r] for theta in self.Theta for d in self.D])
        
            outsourcing = sum([self.omega_var[a, theta, d].X for r in self.R for a in self.A[r] for theta in self.Theta for d in self.D])
            obj_value = self.m.ObjVal
            


        return {'regions': self.i.regions, 'n_employees': self.i.n_employees, 'obj_value': obj_value, 
                'lower_bound' : self.m.objbound, 
                'elapsed_time': self.m.Runtime, 'status': self.m.status,
                'hiring_costs': hiring, 'outsourcing_costs': outsourcing, 'exec_time': self.exec_time,
                'n_variables': self.m.NumVars, 'n_constraints': self.m.NumConstrs, 'n_nonzeroes': self.m.NumNZs}

        

