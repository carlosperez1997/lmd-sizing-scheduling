
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
    workforce_dict: dict

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
            self.workforce_dict = kwargs['workforce_dict']
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
        # self.k = self.__load_workforce(self.workforce_file)
        self.k = self.workforce_dict

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
                self.max_n_shifts = kwargs['max_n_shifts']
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

    def __build_roster_model(self) -> None:
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

        #constraint - connecting employees moving areas (r and k decision variables)
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

        #constraint - ensuring employee can be in only one area at a time
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

        #constraint - ensuring employee can only work on shift in a day
        self.m.addConstrs((
            sum(self.r[(employee, shift_start, day)]
                for shift_start in self.i.shifts[region]
                )
            <= 1
                for region in self.i.regions
                for employee in self.i.employees[region]
                for day in self.i.days
        ), name = 'one_shift_a_day')

        #constraint - ensuring employee will have at least one rest day a week
        self.m.addConstrs((
            sum(self.r[(employee, shift_start, day)]
                for shift_start in self.i.shifts[region]
                for day in self.i.days
                )
            <= 6
                for region in self.i.regions
                for employee in self.i.employees[region]
        ), name = 'one_rest_day_per_week')

        #constraint - ensuring employees work a minimum number of hours
        self.m.addConstrs((
            sum(HOURS_IN_SHIFT_P*self.r[(employee, shift_start, day)]
                for shift_start in self.i.shifts[region]
                for day in self.i.days
            )
            >= self.i.h_min
                for region in self.i.regions
                for employee in self.i.employees[region]
        ), name = 'min_hours_worked')

        #constraint - ensuring employees work a maximum number of hours
        self.m.addConstrs((
            sum(HOURS_IN_SHIFT_P*self.r[(employee, shift_start, day)]
                for shift_start in self.i.shifts[region]
                for day in self.i.days
            )
            <= self.i.h_max
                for region in self.i.regions
                for employee in self.i.employees[region]
        ), name = 'max_hours_worked')

        #constraint - employees have a max number of different shift starting times
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

        #constraint - demand can be met by outsourcing
        self.m.addConstr((
            self.omega[(scenario, area, theta, day)] >= (self.i.srequired[(scenario, area, theta, day)] - sum(self.k[(employee, area, theta, day)] for employee in self.i.employees[region]))*(self.i.sdemand[(scenario, area, theta, day)/self.i.srequired[(scenario, area, theta, day)]])*(self.i.outsourcing_cost)
            for region in self.i.regions
            for area in self.i.reg_areas[region]
            for day in self.i.days
            for theta in self.i.periods[day]
            for scenario in self.i.scenarios[day]
        ), name = 'outsourcing_demand')

    def __basic_output(self) -> dict:
        output = {
            'instance': [self.i.ibasename],
            'model': [self.args.model],
            'city': [self.i.ibasename.split('_')[0]],
            'demand_baseline': [self.i.i['demand_baseline']],
            'demand_type': [self.i.i['demand_type']],
            'outsourcing_cost_multiplier': [self.args.outsourcing_cost_multiplier],
            'region_multiplier': [self.i.reg_multiplier],
            'global_multiplier': [self.i.glb_multiplier],
        }
        if self.i.model == 'partflex':
            output['max_n_shifts'] = [self.i.max_n_shifts]
        return output

    def __roster_results(self) -> dict:
        basic_output = self.__basic_output()

        total_employees = 0
        for region in self.i.regions:
            total_employees += self.i.n_employees[region]

        basic_output['objective_value'] = [self.m.ObjVal]
        basic_output['workforce_size'] = [total_employees]
        basic_output['hiring_costs'] = [total_employees]
        basic_output['objective_value_post_hire'] = [self.m.ObjVal - total_employees]

        return basic_output

    def __roster_output(self) -> dict:
        basic_output = self.__basic_output()
        #change this code here after testing
        return basic_output

    def solve_roster_results(self) -> dict:
        self.__build_roster_model()
        self.m.optimize()
        return self.__roster_results()

    def solve_roster_output(self) -> dict:
        self.__build_roster_model()
        self.m.optimize()
        return self.__roster_output()

#function call run execution
def run_roster_solver_results(model, instance_file, outsourcing_cost_multiplier, regional_multiplier, global_multiplier, shift_file, workforce_dict, n_days, max_n_shifts=None, output=None):
    args = Namespace(
        model=model,
        instance_file=instance_file,
        outsourcing_cost_multiplier=outsourcing_cost_multiplier,
        regional_multiplier=regional_multiplier,
        global_multiplier=global_multiplier,
        shift_file = shift_file,
        workforce_dict = workforce_dict,
        n_days = n_days,
        max_n_shifts=max_n_shifts,
        output=output
    )

    # Assuming Instance and Solver classes are defined above in this script
    i = Instance(args=args)
    solver = Solver(args=args, i=i)

    roster_results = solver.solve_roster_results()

    return roster_results


def run_roster_solver_output(model, instance_file, outsourcing_cost_multiplier, regional_multiplier, global_multiplier, shift_file, workforce_dict, n_days, max_n_shifts=None, output=None):
    args = Namespace(
        model=model,
        instance_file=instance_file,
        outsourcing_cost_multiplier=outsourcing_cost_multiplier,
        regional_multiplier=regional_multiplier,
        global_multiplier=global_multiplier,
        shift_file = shift_file,
        workforce_dict = workforce_dict,
        n_days = n_days,
        max_n_shifts=max_n_shifts,
        output=output
    )

    # Assuming Instance and Solver classes are defined above in this script
    i = Instance(args=args)
    solver = Solver(args=args, i=i)

    roster_results = solver.solve_roster_output()

    return roster_results
