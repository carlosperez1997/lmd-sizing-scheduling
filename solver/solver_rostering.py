
from argparse import ArgumentParser, Namespace
from gurobipy import Model, GRB, tupledict
from os.path import basename, splitext
from itertools import chain
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import pandas as pd


CAPACITY = 5
COST_PER_COURIER_AND_PERIOD = 1.0
COST_PER_PARCEL_AND_PERIOD = COST_PER_COURIER_AND_PERIOD / CAPACITY
EPS = 1e-6

#NEW
HOURS_IN_SHIFT_P = 8


class Instance:
    #arguments input to solver
    args: Optional[Namespace]
    #instance file dictionary
    i: dict
    #shift file dictionary
    j: dict
    #workforce size file dictionary
    k: dict

    #simulation parameters
    reg_multiplier: float #
    glb_multiplier: float #
    outsourcing_cost_multiplier: float #
    outsourcing_cost: float #

    #REGION LEVEL

    #regions
    dregions: dict
    n_regions: int
    regions: list 

    #areas
    n_areas: int
    areas: list

    #regions and areas
    reg_areas: dict
    area_regs: dict

    #employees
    n_employees: dict #keys - region: int - n_employees
    employees: dict #keys - region: list of employee index (employee index is distinct)

    #shifts in region R
    n_shifts: dict #keys - region: int - n_shifts
    shifts: dict #keys - region: list of theta start times

    #DAY LEVEL
    
    #days
    n_days: int
    days: list

    #theta
    n_periods: dict #keys - day: int - n_periods
    periods: dict #keys - day: list - thetas

    #demand: packages and required couriers
    n_scenarios: dict #keys - day: int - n_scenarios
    scenarios: dict #keys - day: list - scenarios
    sdemand: tupledict #number of packages (s, a, theta, day)
    srequired: tupledict #number of couriers (s, a, theta, day)

    #GLOBAL LEVEL

    #rostering constraints
    h_min: int #min hours worked by all employees
    h_max: int #max hours worked by all employees
    b_max: int #max number of different shifts

    #solver inputs
    model: str #which type of model
    name: str #model name
    ibasename: str #model name
    max_n_shifts = Optional[int] 
    instance_file: str #where all the information comes from
    shift_file: str #where region/shift information comes from
    workforce_file: str

    #unsure if needed
    ub_reg: dict #upper bound
    ub_global: int #upper bound

    def __init__(self, args: Optional[Namespace], **kwargs):
        self.args = args

        #initialize inputted data
        if self.args is None:
            self.instance_file = kwargs['instance_file']
            self.reg_multiplier = kwargs['regional_multiplier']
            self.glb_multiplier = kwargs['global_multiplier']
            self.outsourcing_cost_multiplier = kwargs['outsourcing_cost_multiplier']
            self.model = kwargs['model']
            self.shift_file = kwargs['shift_file']
            self.workforce_file = kwargs['workforce_file']
            self.n_days = kwargs['n_days']
        else:
            self.instance_file = self.args.instance_file
            self.reg_multiplier = self.args.regional_multiplier
            self.glb_multiplier = self.args.global_multiplier
            self.outsourcing_cost_multiplier = self.args.outsourcing_cost_multiplier
            self.model = self.args.model
            self.shift_file = self.args.shift_file
            self.n_employees = self.args.n_employees
            self.n_days = self.args.n_days

        #load json instance_file dictionary
        self.i = self.__load_instance(self.instance_file)
        self.j = self.__load_shift(self.shift_file)
        self.k = self.__load_workforce(self.workforce_file)

        #initialize the rest of the class arguments
        self.__compute_data(**kwargs)

    #initialize the other variables
    def __compute_data(self, **kwargs) -> None:
        self.outsourcing_cost = COST_PER_COURIER_AND_PERIOD * self.outsourcing_cost_multiplier

        #region
        self.dregions = self.i['geography']['city']['regions']
        self.n_regions = len(self.dregions)
        self.regions = [region['id'] for region in self.dregions]

        self.n_areas = sum(len(region['areas']) for region in self.dregions)
        self.areas = [area['id'] for region in self.dregions for area in region['areas']]

        self.reg_areas = {
            region['id']: [area['id'] for area in region['areas']] for region in self.dregions
        }
        self.area_regs = {
            area: [region for region in self.regions if area in self.reg_areas[region]][0] for area in self.areas
        }

        #region level variables

        #shifts (region)
        for region in self.regions:
            if region in list(self.j.keys()):
                self.n_shifts[region] = len(self.j[region]['shifts_start'].keys())
                self.shifts[region] = [self.j[region]['shifts_start'][index_] for index_ in self.j[region]['shifts_start'].keys()]
            else:
                self.n_shifts[region] = 0
                self.shifts[region] = []
        
        #employees (region) (employee index is distinct)
        counter_ = 0
        for region in self.regions:
            if region in list(self.k.keys()):
                self.n_employees[region] = self.k[region]
                self.employees[region] = [iter_ for iter_ in range(counter_, counter_ + self.n_employees[region])]
                counter_ += self.n_employees[region]
            else:
                self.n_employees[region] = 0
                self.employees[region] = []

        #day
        self.days = [iter_ for iter_ in range(self.n_days)]

        #day level variables

        #periods (day)
        for day in self.days:
            self.n_periods[day] = self.i['num_time_intervals']
            self.periods[day] = list(range(self.n_periods))

        #scenarios (day)
        for day in self.days:
            self.n_scenarios[day] = self.i['num_scenarios']
            self.scenarios[day] = list(range(self.n_scenarios))

        #demand/required (s, a, theta, day)

        self.sdemand = dict()
        self.srequired = dict()

        for scenario in self.i['scenarios']:
            s = scenario['scenario_num']
            for data in scenario['data']:
                a = data['area_id']
                for theta, d in enumerate(data['demand']):
                    for day in self.days:
                        self.sdemand[(s, a, theta, day)] = d

                for theta, m in enumerate(data['required_couriers']):
                    for day in self.days:
                        self.srequired[(s, a, theta, day)] = m

        self.ub_reg, self.ub_global = self.__get_ubs()

        if self.model == 'partflex':
            if self.args is None:
                self.max_n_shift = kwargs['max_n_shifts']
            else:
                self.max_n_shifts = self.args.max_n_shifts

        self.ibasename = splitext(basename(self.instance_file))[0]
        self.name = self.ibasename + f"_oc={self.outsourcing_cost_multiplier}_rm={self.reg_multiplier}_gm={self.glb_multiplier}"

    def __load_instance(self, instance_file: str) -> dict:
        with open(instance_file) as f:
            return json.load(f)

    def __load_shift(self, shift_file: str) -> dict:
        with open(shift_file, 'r') as file:
            data = json.load(file)
            df_shifts = pd.DataFrame(data)
        df_shifts = df_shifts[(df_shifts['outsourcing_cost_multiplier']==self.outsourcing_cost_multiplier)&(df_shifts['regional_multiplier']==self.reg_multiplier)&(df_shifts['global_multiplier']==self.glb_multiplier)]
        #fixed or flex
        if self.model in ['fixed','flex']:
            df_shifts = df_shifts[df_shifts['model']==self.model]
        #partflex
        else:
            df_shifts = df_shifts[(df_shifts['model']==self.model)&(df_shifts['max_n_shifts']==self.max_n_shifts)]
        df_shifts.reset_index(drop = True, inplace=True)
        df_shifts = df_shifts[['region','shifts_start','shifts_end']]
        dict_shifts = df_shifts.set_index('region').to_dict(orient='index')
        return dict_shifts
    
    def __load_workforce(self, workforce_file: str) -> dict:
        with open(workforce_file, 'r') as file:
            data = json.load(file)
            df_workforce = pd.DataFrame(data)
        df_workforce = df_workforce[(df_workforce['outsourcing_cost_multiplier']==self.outsourcing_cost_multiplier)&(df_workforce['regional_multiplier']==self.reg_multiplier)&(df_workforce['global_multiplier']==self.glb_multiplier)]
        #fixed or flex
        if self.model in ['fixed','flex']:
            df_workforce = df_workforce[df_workforce['model']==self.model]
        #partflex
        else:
            df_workforce = df_workforce[(df_workforce['model']==self.model)&(df_workforce['max_n_shifts']==self.max_n_shifts)]
        df_workforce.reset_index(drop = True, inplace=True)
        df_workforce = df_workforce[['region','workforce_size']]
        dict_workforce = df_workforce.set_index('region').to_dict(orient='index')
        return dict_workforce

    def __get_ubs(self) -> Tuple[int]:
        mhat1 = {
            (a, theta): np.mean([
                self.srequired[s, a, theta] for s in self.scenarios
            ]) for a in self.areas for theta in self.periods
        }
        mhat2 = {
            a: np.mean([
                mhat1[a, theta] for theta in self.periods
            ]) for a in self.areas
        }
        ub_reg = {
            region: int(self.reg_multiplier * sum(mhat2[a] for a in self.reg_areas[region])) \
            for region in self.regions
        }
        ub_global = int(self.glb_multiplier * sum(ub_reg.values()))

        return ub_reg, ub_global


class Solver:
    args: Namespace
    i: Instance

    m: Model
    
    #decision variables
    r: tupledict #(e,p,day)
    k: tupledict #(e,a,theta,day)
    U: tupledict #(e, p)
    omega: tupledict #(s,a,theta,day)

    def __init__(self, args: Namespace, i: Instance):
        self.args = args
        self.i = i

    def __build_model(self) -> None:
        self.m = Model()

        #r decision variable (e,p,d)
        self.r = {(employee, shift_start, day): self.m.addVar(vtype=GRB.BINARY, name='r') 
                for region in self.i.regions 
                for employee in self.i.employees[region] 
                for shift_start in self.i.shifts[region] 
                for day in self.i.days}

        #k decision variable (e,a,theta,d)
        self.k = {(employee, area, theta, day): self.m.addVar(vtype = GRB.BINARY, obj = 1, name = 'k')
                  for region in self.i.regions
                  for employee in self.i.employees[region]
                  for area in self.i.reg_areas[region]
                  for day in self.i.days
                  for theta in self.i.periods[day]}

        #U decision variable (e,p)
        self.U = {(employee, shift_start): self.m.addVar(vtype=GRB.BINARY, name='U') 
                for region in self.i.regions 
                for employee in self.i.employees[region] 
                for shift_start in self.i.shifts[region]}

        #omega decision variable (s,a,theta,day)
        self.omega = {(scenario, area, theta, day): self.m.addVar(vtype=GRB.CONTINUOUS, lb = 0, obj=1/self.i.n_scenarios[day], name='omega') 
                    for day in self.i.days 
                    for scenario in self.i.scenarios[day] 
                    for area in self.i.areas 
                    for theta in self.i.periods[day]}

        #constraints
        self.m.addConstrs((
            sum(self.k[(employee, area, theta, day)]
                for area in self.i.area_regs[region]
                for theta in range(shift_start, shift_start+4)
                )
            ==
            (1/2)*HOURS_IN_SHIFT_P*self.r[(employee, shift_start, day)]
                for region in self.i.regions
                for employee in self.i.employees[region]
                for shift_start in self.i.shifts[region]
                for day in self.i.days
        ), name = 'connect_employees_moving_areas')

        self.m.addConstrs((
            sum(self.k[(employee, area, theta, day)]
                for area in self.i.area_regs[region]
                )
            <= 1
                for region in self.i.regions
                for employee in self.i.employees[region]
                for day in self.i.days
                for theta in self.i.periods[day]
        ), name = 'one_area_at_a_time')

        self.m.addConstrs((
            sum(self.r[(employee, shift_start, day)]
                for shift_start in self.i.shifts[region]
                )
            <= 1
                for region in self.i.regions
                for employee in self.i.employees[region]
                for day in self.i.days
        ), name = 'one_shift_a_day')

        self.m.addConstrs((
            sum(self.r[(employee, shift_start, day)]
                for shift_start in self.i.shifts[region]
                for day in self.i.days
                )
            <= 6
                for region in self.i.regions
                for employee in self.i.employees[region]
        ), name = 'one_rest_day_per_week')

        self.m.addConstrs((
            sum(HOURS_IN_SHIFT_P*self.r[(employee, shift_start, day)]
                for shift_start in self.i.shifts[region]
                for day in self.i.days
            )
            >= self.i.h_min
                for region in self.i.regions
                for employee in self.i.employees[region]
        ), name = 'min_hours_worked')

        self.m.addConstrs((
            sum(HOURS_IN_SHIFT_P*self.r[(employee, shift_start, day)]
                for shift_start in self.i.shifts[region]
                for day in self.i.days
            )
            <= self.i.h_max
                for region in self.i.regions
                for employee in self.i.employees[region]
        ), name = 'max_hours_worked')

        self.m.addConstr((
            sum(self.r[(employee, shift_start, day)]
                for day in self.i.days
            )
            <= self.U[(employee, shift_start)]*(10000000000)
                for region in self.i.regions
                for employee in self.i.employees[region]
                for shift_start in self.shifts[region]
        ), name = 'max_different_start1')

        self.m.addConstr((
            sum(self.U[(employee, shift_start)]
                for shift_start in self.i.shifts[region]
            )
            <= self.i.b_max
                for region in self.i.regions
                for employee in self.i.employees[region]
        ), name = 'max_different_start2')

        self.m.addConstr((
            self.omega[(scenario, area, theta, day)] >= (self.i.srequired[(scenario, area, theta, day)] - sum(self.k[(employee, area, theta, day)] for employee in self.i.employees[region]))*(self.i.sdemand[(scenario, area, theta, day)/self.i.srequired[(scenario, area, theta, day)]])*(self.i.outsourcing_cost)
            for region in self.i.regions
            for area in self.i.reg_areas[region]
            for day in self.i.days
            for theta in self.i.periods[day]
            for scenario in self.i.scenarios[day]
        ), name = 'outsourcing_demand')

#write a function to return results

    # def __build_base_model(self) -> None:

    #     #objective function
    #     self.m = Model()
    #     self.x = self.m.addVars(self.i.areas, self.i.periods, vtype=GRB.INTEGER, lb=0, obj=1, name='x')
    #     self.omega = self.m.addVars(self.i.areas, self.i.periods, self.i.scenarios, vtype=GRB.CONTINUOUS, obj=1/self.i.n_scenarios, name='omega')

    #     self.m.addConstrs((
    #         sum(self.x[a, theta] for a in self.i.reg_areas[region]) <= self.i.ub_reg[region]
    #             for region in self.i.regions
    #             for theta in self.i.periods
    #     ), name='reg_bound')

#         self.m.addConstrs((
#             self.x.sum('*', theta) <= self.i.ub_global
#             for theta in self.i.periods
#         ), name='global_bound')

#         self.m.addConstrs((
#             self.i.srequired[s, a, theta] * self.omega[a, theta, s] >= \
#             (self.i.srequired[s, a, theta] - self.x[a, theta]) * self.i.sdemand[s, a, theta] * self.i.outsourcing_cost
#             for a in self.i.areas
#             for theta in self.i.periods
#             for s in self.i.scenarios
#         ), name='set_omega')

    
#     def __add_y_vars(self):
#         y_idx = [
#             (a1, a2, theta)
#             for a1 in self.i.areas
#             for a2 in self.i.areas
#             for theta in self.i.periods
#             if a1 != a2 and self.i.area_regs[a1] == self.i.area_regs[a2]
#         ]
#         self.y = self.m.addVars(y_idx, vtype=GRB.INTEGER, lb=0, obj=EPS, name='y')

#     def __build_fixed_model(self) -> None:
#         self.__build_base_model()
#         self.__add_y_vars()
        
#         self.m.addConstrs((
#             sum(self.x[a, theta] for a in self.i.reg_areas[region]) == sum(self.x[a, shift[0]] for a in self.i.reg_areas[region])
#             for region in self.i.regions
#             for shift in self.i.shifts
#             for theta in shift[1:]
#         ), name='fix_region_n_couriers_in_shift')

#         self.m.addConstrs((
#             self.x[a1, theta] == self.x[a1, theta - 1] + self.y.sum('*', a1, theta) - self.y.sum(a1, '*', theta)
#             for region in self.i.regions
#             for a1 in self.i.reg_areas[region]
#             for shift in self.i.shifts
#             for theta in shift[1:]
#         ), name='flow_balance')

#     def __build_flexible_model(self) -> None:
#         self.__build_base_model()
#         self.__add_y_vars()

#         self.zplus = self.m.addVars(self.i.areas, self.i.periods, vtype=GRB.INTEGER, lb=0, name='zplus')
#         self.zminus = self.m.addVars(self.i.areas, self.i.periods, vtype=GRB.INTEGER, lb=0, name='zminus')

#         if self.i.n_periods == 8:
#             shift_len = 4
#         else:
#             return NotImplementedError('Shift length only implemented for n_periods == 8')
        
#         for theta in range(self.i.n_periods - shift_len + 1, self.i.n_periods):
#             for a in self.i.areas:
#                 self.zminus[a, theta].UB = 0

#         for theta in range(shift_len - 1):
#             for a in self.i.areas:
#                 self.zplus[a, theta].UB = 0

#         #CM constraint needs to be updated
#         self.m.addConstrs((
#             sum(self.zminus[a, theta] for a in self.i.reg_areas[region]) == \
#             sum(self.zplus[a, theta + shift_len - 1] for a in self.i.reg_areas[region])
#             for region in self.i.regions
#             for theta in self.i.periods
#             if theta < self.i.n_periods + 1 - shift_len #changed from -1 to +1
#         ), name='fix_region_n_couriers_in_shift')

#         self.m.addConstrs((
#             self.x[a1, theta] == \
#             self.x[a1, theta - 1] + \
#             self.y.sum('*', a1, theta) - \
#             self.y.sum(a1, '*', theta) + \
#             self.zminus[a1, theta] - self.zplus[a1, theta - 1]
#             for region in self.i.regions
#             for a1 in self.i.reg_areas[region]
#             for theta in self.i.periods
#             if theta > 0
#         ), name='flow_balance')

#         self.m.addConstrs((
#             self.x[a, 0] == self.zminus[a, 0]
#             for a in self.i.areas
#         ), name='flow_balance_first_period')

#     def __build_partflex_model(self) -> None:
#         self.__build_flexible_model()

#         if self.i.n_periods == 8:
#             shift_len = 4
#         else:
#             return NotImplementedError('Shift length only implemented for n_periods == 8')

#         #JC constraint updated
#         w_idx = range(self.i.n_periods - shift_len - 1) #changed from a -1 to a +1
#         self.w = self.m.addVars(w_idx, vtype=GRB.BINARY, name='w')

#         self.m.addConstrs((
#             sum(self.zminus[a, theta] for a in self.i.reg_areas[region]) <= self.i.ub_reg[region] * self.w[theta]
#             for region in self.i.regions
#             for theta in w_idx
#         ), name='link_z_w')

#         self.m.addConstr(self.w.sum() <= self.i.max_n_shifts, name='limit_n_shifts')

#     def plot_couriers(self) -> None:
#         mtx = [
#             [
#                 sum(int(self.x[a, theta].X)
#                 for a in self.i.reg_areas[region])
#                 for theta in self.i.periods
#             ]
#             for region in self.regions
#         ]

#         fig, ax = plt.subplots(figsize=(self.i.n_periods, self.i.n_regions))
#         sns.heatmap(mtx, annot=True, linecolor='white', linewidths=1, square=True, cmap='Blues', ax=ax)
#         ax.set_xlabel('Period')
#         ax.set_ylabel('Region')
#         ax.set_yticklabels(self.i.regions)

#         fig.tight_layout()
#         fig.savefig('couriers.png', dpi=96, bbox_inches='tight')

#     def solve_base(self) -> dict:
#         self.__build_base_model()
#         self.m.optimize()
#         return self.__basic_results()
    
#     def return_solve_base(self):
#         self.__build_base_model()
#         self.m.optimize()
#         return (self.m.ObjVal, self.x, self.i)
#         #return self.__basic_results()

#     def solve_fixed(self) -> dict:
#         self.__build_fixed_model()
#         self.m.optimize()
#         return self.__fixed_results()

#     def solve_partflex(self) -> dict:
#         self.__build_partflex_model()
#         self.m.optimize()
#         return self.__flex_results()

#     def solve_flex(self) -> dict:
#         self.__build_flexible_model()
#         self.m.optimize()
#         return self.__flex_results()

#     def __basic_results(self) -> dict:
#         results = {
#             'instance': self.i.ibasename,
#             'model': self.args.model,
#             'city': self.i.ibasename.split('_')[0],
#             'DB': self.i.i['demand_baseline'],
#             'DT': self.i.i['demand_type'],
#             'OC': self.args.outsourcing_cost_multiplier,
#             'RM': self.i.reg_multiplier,
#             'GM': self.i.glb_multiplier,
#             'num_periods': self.i.i['num_time_intervals'],
#             'num_scenarios': self.i.i['num_scenarios'],
#             'obj_value': self.m.ObjVal,
#             'elapsed_time': self.m.Runtime,
#             'n_variables': self.m.NumVars,
#             'n_constraints': self.m.NumConstrs,
#             'n_nonzeroes': self.m.NumNZs
#         }

#         hiring_costs = sum(self.x[a, theta].X for a, theta in self.x)
#         outsourcing_costs = self.m.ObjVal - hiring_costs

#         hired_couriers = {
#             a: [int(self.x[a, theta].X) for theta in self.i.periods] for a in self.i.areas
#         }

#         outsourced_parcels = dict()
#         outsourced_parcels_pct = dict()
#         inhouse_parcels = dict()

#         for a in self.i.areas:
#             outsourced = list()
#             outsourced_pct = list()
#             inhouse = list()

#             for theta in self.i.periods:
#                 scenarios_with_demand = [s for s in self.i.scenarios if self.i.srequired[s, a, theta] > 0]

#                 if len(scenarios_with_demand) == 0:
#                     outsourced.append(0)
#                     outsourced_pct.append(0)
#                     inhouse.append(0)
#                     continue

#                 tot_outsourced = sum(
#                     (self.i.srequired[s, a, theta] - self.x[a, theta].X) * self.i.sdemand[s, a, theta] / self.i.srequired[s, a, theta] \
#                     for s in scenarios_with_demand
#                 )
#                 tot_outsourced_pct = sum(
#                     100 * (self.i.srequired[s, a, theta] - self.x[a, theta].X) / self.i.srequired[s, a, theta] \
#                     for s in scenarios_with_demand
#                 )
#                 tot_inhouse = sum(
#                     self.i.sdemand[s, a, theta] * self.x[a, theta].X / self.i.srequired[s, a, theta] \
#                     for s in scenarios_with_demand
#                 )

#                 # If we hire more couriers than we need, we don't outsource a negative amount,
#                 # we outsource zero.
#                 tot_outsourced = max(tot_outsourced, 0)
#                 tot_outsourced_pct = max(tot_outsourced_pct, 0)

#                 avg_outsourced = tot_outsourced / len(scenarios_with_demand)
#                 avg_outsourced_pct = tot_outsourced_pct / len(scenarios_with_demand)
#                 avg_inhouse = tot_inhouse / len(scenarios_with_demand)

#                 outsourced.append(avg_outsourced)
#                 outsourced_pct.append(avg_outsourced_pct)
#                 inhouse.append(avg_inhouse)

#             outsourced_parcels[a] = outsourced
#             outsourced_parcels_pct[a] = outsourced_pct
#             inhouse_parcels[a] = inhouse

#         regional_hired_pct = {
#             region: 100 * sum(self.x[a, theta].X for a in self.i.reg_areas[region] for theta in self.i.periods) / (self.i.ub_reg[region] * self.i.n_periods) \
#             for region in self.i.regions
#         }
#         regional_avg_hired_pct = np.mean(list(regional_hired_pct.values()))
#         global_avg_hired_pct = 100 * sum(self.x[a, theta].X for a, theta in self.x) / (self.i.ub_global * self.i.n_periods)

#         if hasattr(self, 'y') and self.y is not None:
#             pct_movement = list()

#             for r in self.i.regions:
#                 for theta in self.i.periods:
#                     movements = sum(
#                         self.y[a1, a2, theta].X
#                         for a1 in self.i.areas
#                         for a2 in self.i.areas
#                         if a1 != a2 and self.i.area_regs[a1] == r and self.i.area_regs[a2] == r
#                     )

#                     employed = sum(
#                         self.x[a, theta].X
#                         for a in self.i.reg_areas[r]
#                     )

#                     if employed == 0:
#                         pct_movement.append(0.0)
#                     else:
#                         pct_movement.append(100 * movements / employed)

#             results['courier_moved_pct'] = np.mean(pct_movement)

#         results['hiring_costs'] = hiring_costs
#         results['outsourcing_costs'] = outsourcing_costs
#         results['hired_couriers'] = hired_couriers
#         results['outsourced_parcels'] = outsourced_parcels
#         results['inhouse_parcels'] = inhouse_parcels
#         results['regional_hired_pct'] = regional_hired_pct
#         results['regional_avg_hired_pct'] = regional_avg_hired_pct
#         results['global_avg_hired_pct'] = global_avg_hired_pct

#         return results

#     def __fixed_results(self) -> dict:
#         results = self.__basic_results()
#         results['n_shift_start_periods'] = 2
#         return results

#     def __flex_results(self) -> dict:
#         results = self.__basic_results()

#         periods_with_start = 0
#         for theta in self.i.periods:
#             if not any((a, theta) in self.zminus for a in self.i.areas):
#                 continue
#             if any(self.zminus[a, theta].X > 0.1 for a in self.i.areas):
#                 periods_with_start += 1

#         results['periods_with_start'] = periods_with_start
#         results['periods_with_start_pct'] = 100 * periods_with_start / self.i.n_periods

#         return results


#     #output code
#     def __basic_output(self) -> dict:
#         output = {
#             'instance': [self.i.ibasename],
#             'model': [self.args.model],
#             'city': [self.i.ibasename.split('_')[0]],
#             'demand_baseline': [self.i.i['demand_baseline']],
#             'demand_type': [self.i.i['demand_type']],
#             'outsourcing_cost_multiplier': [self.args.outsourcing_cost_multiplier],
#             'region_multiplier': [self.i.reg_multiplier],
#             'global_multiplier': [self.i.glb_multiplier],
#         }
#         return output
    
#     def __fixed_output(self) -> dict:
#         basic_output = self.__basic_output()
#         basic_output['max_n_shift_start_periods'] = [2]
#         basic_output['actual_n_shift_start_periods'] = [2]

#         output = dict()
#         for val in basic_output.keys():
#             output[val] = []
#         output['region'] = []
#         output['area'] = []
#         output['theta'] = []
#         output['x__a_theta'] = []
#         for region in self.i.regions:
#             for area in self.i.reg_areas[region]:
#                 for theta in self.i.periods:
#                     output['region'].append(region)
#                     output['area'].append(area)
#                     output['theta'].append(theta)
#                     output['x__a_theta'].append(int(self.x[area,theta].X))
#         n_rows = len(output['region'])

#         for val in basic_output.keys():
#             output[val] = basic_output[val]*n_rows
#         print(output.keys())
#         print(len(output['region']))
#         return output

#     def __flex_output(self) -> dict:
#         basic_output = self.__basic_output()
#         basic_output['max_n_shift_start_periods'] = [np.nan]

#         periods_with_start = 0
#         for theta in self.i.periods:
#             if not any((a, theta) in self.zminus for a in self.i.areas):
#                 continue
#             if any(self.zminus[a, theta].X > 0.1 for a in self.i.areas):
#                 periods_with_start += 1
#         basic_output['actual_n_shift_start_periods'] = [periods_with_start]

#         output = dict()
#         for val in basic_output.keys():
#             output[val] = []
#         output['region'] = []
#         output['area'] = []
#         output['theta'] = []
#         output['x__a_theta'] = []
#         output['zminus__a_theta'] = []
#         for region in self.i.regions:
#             for area in self.i.reg_areas[region]:
#                 for theta in self.i.periods:
#                     output['region'].append(region)
#                     output['area'].append(area)
#                     output['theta'].append(theta)
#                     output['x__a_theta'].append(int(self.x[area,theta].X))
#                     output['zminus__a_theta'].append(int(self.zminus[area,theta].X))
#         n_rows = len(output['region'])

#         for val in basic_output.keys():
#             output[val] = basic_output[val]*n_rows
#         print(output.keys())
#         print(len(output['zminus__a_theta']))
#         return output

#     def solve_fixed_output(self) -> dict:
#         self.__build_fixed_model()
#         self.m.optimize()
#         return self.__fixed_output()

#     def solve_partflex_output(self) -> dict:
#         self.__build_partflex_model()
#         self.m.optimize()
#         return self.__flex_output()

#     def solve_flex_output(self) -> dict:
#         self.__build_flexible_model()
#         self.m.optimize()
#         return self.__flex_output()


# #function call run execution
# def run_solver_rostering(model, instance_file, outsourcing_cost_multiplier, regional_multiplier, global_multiplier, shift_file, workforce_file, n_days, max_n_shifts=None, output=None):
#     args = Namespace(
#         model=model,
#         instance_file=instance_file,
#         outsourcing_cost_multiplier=outsourcing_cost_multiplier,
#         regional_multiplier=regional_multiplier,
#         global_multiplier=global_multiplier,
#         shift_file = shift_file,
#         workforce_file = workforce_file,
#         n_days = n_days,
#         max_n_shifts=max_n_shifts,
#         output=output
#     )

#     # Assuming Instance and Solver classes are defined above in this script
#     i = Instance(args=args)
#     # solver = Solver(args=args, i=i)

#     def output_file(args, i):
#         if args.output is not None:
#             return args.output
#         elif args.model == 'fixed':
#             return f"../shifts/{i.name}_model=fixed.json"
#         elif args.model == 'partflex':
#             return f"../shifts/{i.name}_mu={str(args.max_n_shifts)}_model=partflex.json"
#         elif args.model == 'flex':
#             return f"../shifts/{i.name}_model=flex.json"

#     # Model execution logic
#     if args.model == 'fixed':
#         results = solver.solve_fixed_output()
#     elif args.model == 'partflex':
#         results = solver.solve_partflex_output()
#     elif args.model == 'flex':
#         results = solver.solve_flex_output()
#     else:
#         raise ValueError("Invalid model type provided")

#     #change the results to be in the shape of shift information
#     if args.model == 'fixed':
#         #get the regions and make a dictionary
#         dict_shifts = {}
#         for region in i.regions:
#             dict_shifts[region] = {}
#             dict_shifts[region]['shifts_start'] = {0:0,1:4}
#             dict_shifts[region]['shifts_end'] = {0:4,1:8}
#     elif args.model == 'partflex':
#         df_ = pd.DataFrame(results)
#         df_.sort_values(by = ['region','theta'], inplace = True)
#         df_ = df_[df_['zminus__a_theta']==1]
#         df_.drop_duplicates(subset = ['region','theta'], inplace = True)
#         dict_shifts = {}
#         for region in list(df_['region']):
#             dict_shifts[region] = {}
#             dict_shifts[region]['shifts_start'] = {}
#             dict_shifts[region]['shifts_end'] = {}
#             for index_, start_theta in enumerate(df_[df_['region']==region]['theta'].tolist()):
#                 dict_shifts[region]['shifts_start'][index_] = start_theta
#                 dict_shifts[region]['shifts_end'][index_] = start_theta + 4
#     elif args.model == 'flex':
#         df_ = pd.DataFrame(results)
#         df_.sort_values(by = ['region','theta'], inplace = True)
#         df_ = df_[df_['zminus__a_theta']==1]
#         df_.drop_duplicates(subset = ['region','theta'], inplace = True)
#         dict_shifts = {}
#         for region in df_['region'].unique().tolist():
#             dict_shifts[region] = {}
#             dict_shifts[region]['shifts_start'] = {}
#             dict_shifts[region]['shifts_end'] = {}
#             for index_, start_theta in enumerate(df_[df_['region']==region]['theta'].tolist()):
#                 dict_shifts[region]['shifts_start'][index_] = start_theta
#                 dict_shifts[region]['shifts_end'][index_] = start_theta + 4
#     else:
#         raise ValueError("Invalid model type provided")

#     # Save results
#     output_path = output_file(args, i)
#     with open(output_path, 'w') as f:
#         json.dump(dict_shifts, f, indent=2)
#     print(f"Results saved to {output_path}")

# #function call run execution
# def run_solver_shift_return(model, instance, outsourcing_cost_multiplier, regional_multiplier, global_multiplier, max_n_shifts=None, output=None):
#     args = Namespace(
#         model=model,
#         instance=instance,
#         outsourcing_cost_multiplier=outsourcing_cost_multiplier,
#         regional_multiplier=regional_multiplier,
#         global_multiplier=global_multiplier,
#         max_n_shifts=max_n_shifts,
#         output=output
#     )

#     # Assuming Instance and Solver classes are defined above in this script
#     i = Instance(args=args)
#     solver = Solver(args=args, i=i)

#     def output_file(args, i):
#         if args.output is not None:
#             return args.output
#         elif args.model == 'fixed':
#             return f"../shifts/{i.name}_model=fixed.json"
#         elif args.model == 'partflex':
#             return f"../shifts/{i.name}_mu={str(args.max_n_shifts)}_model=partflex.json"
#         elif args.model == 'flex':
#             return f"../shifts/{i.name}_model=flex.json"

#     # Model execution logic
#     if args.model == 'fixed':
#         results = solver.solve_fixed_output()
#     elif args.model == 'partflex':
#         results = solver.solve_partflex_output()
#     elif args.model == 'flex':
#         results = solver.solve_flex_output()
#     else:
#         raise ValueError("Invalid model type provided")

#     #change the results to be in the shape of shift information
#     if args.model == 'fixed':
#         #get the regions and make a dictionary
#         dict_shifts = {}
#         for region in i.regions:
#             dict_shifts[region] = {}
#             dict_shifts[region]['shifts_start'] = {0:0,1:4}
#             dict_shifts[region]['shifts_end'] = {0:4,1:8}
#     elif args.model == 'partflex':
#         df_ = pd.DataFrame(results)
#         df_.sort_values(by = ['region','theta'], inplace = True)
#         df_ = df_[df_['zminus__a_theta']==1]
#         df_.drop_duplicates(subset = ['region','theta'], inplace = True)
#         dict_shifts = {}
#         for region in list(df_['region']):
#             dict_shifts[region] = {}
#             dict_shifts[region]['shifts_start'] = {}
#             dict_shifts[region]['shifts_end'] = {}
#             for index_, start_theta in enumerate(df_[df_['region']==region]['theta'].tolist()):
#                 dict_shifts[region]['shifts_start'][index_] = start_theta
#                 dict_shifts[region]['shifts_end'][index_] = start_theta + 4
#     elif args.model == 'flex':
#         df_ = pd.DataFrame(results)
#         df_.sort_values(by = ['region','theta'], inplace = True)
#         df_ = df_[df_['zminus__a_theta']==1]
#         df_.drop_duplicates(subset = ['region','theta'], inplace = True)
#         dict_shifts = {}
#         for region in df_['region'].unique().tolist():
#             dict_shifts[region] = {}
#             dict_shifts[region]['shifts_start'] = {}
#             dict_shifts[region]['shifts_end'] = {}
#             for index_, start_theta in enumerate(df_[df_['region']==region]['theta'].tolist()):
#                 dict_shifts[region]['shifts_start'][index_] = start_theta
#                 dict_shifts[region]['shifts_end'][index_] = start_theta + 4
#     else:
#         raise ValueError("Invalid model type provided")
    
#     return dict_shifts, i.n_regions