
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
import re
import math

CAPACITY = 5
COST_PER_COURIER_AND_PERIOD = 1.0
COST_PER_PARCEL_AND_PERIOD = COST_PER_COURIER_AND_PERIOD / CAPACITY
EPS = 1e-6

#NEW
HOURS_IN_SHIFT_P = 8
N_DAYS = 7
N_WORK_DAYS = 6

class Instance:
    #arguments input to solver
    args: Optional[Namespace]
    #instance file dictionary
    i_weekday: dict
    i_weekend: dict
    #shift file dictionary
    j_weekday: dict
    j_weekend: dict
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
    n_shifts: dict #keys - (region, day): int - n_shifts
    shifts: dict #keys - (region, day): list of theta start times
    shifts_distinct: dict #keys - (region): list of distinct shift start times

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

    instance_file_weekday: str #where all the information comes from
    shift_file_weekday: str #where region/shift information comes from
    instance_file_weekend: str #where all the information comes from
    shift_file_weekend: str #where region/shift information comes from

    expand_workforce_to_regions = Optional[bool] #indicator to read in optiman workforce size
    workforce_file: Optional[str]
    workforce_dict: dict
    population: dict

    def __init__(self, args: Optional[Namespace], **kwargs):
        self.args = args

        #initialize inputted data
        if self.args is None:
            self.instance_file_weekday = kwargs['instance_file_weekday']
            self.shift_file_weekday = kwargs['shift_file_weekday']
            self.instance_file_weekend = kwargs['instance_file_weekend']
            self.shift_file_weekend = kwargs['shift_file_weekend']

            self.expand_workforce_to_regions = kwargs['expand_workforce_to_regions']            
            self.workforce_dict = kwargs['workforce_dict']

            self.outsourcing_cost_multiplier = kwargs['outsourcing_cost_multiplier']
            self.reg_multiplier = kwargs['regional_multiplier']
            self.glb_multiplier = kwargs['global_multiplier']

            self.model = kwargs['model']
            self.h_min = kwargs['h_min']
            self.h_max = kwargs['h_max']
            self.b_max = kwargs['max_n_diff']
        else:
            self.instance_file_weekday = self.args.instance_file_weekday
            self.shift_file_weekday = self.args.shift_file_weekday
            self.instance_file_weekend = self.args.instance_file_weekend
            self.shift_file_weekend = self.args.shift_file_weekend

            self.expand_workforce_to_regions = self.args.expand_workforce_to_regions            
            self.workforce_dict = self.args.workforce_dict

            self.outsourcing_cost_multiplier = self.args.outsourcing_cost_multiplier
            self.reg_multiplier = self.args.regional_multiplier
            self.glb_multiplier = self.args.global_multiplier

            self.model = self.args.model
            self.h_min = self.args.h_min
            self.h_max = self.args.h_max
            self.b_max = self.args.max_n_diff

        if self.model == 'partflex':
            if self.args is None:
                self.max_n_shifts = kwargs['max_n_shifts']
            else:
                self.max_n_shifts = self.args.max_n_shifts

        #load json instance_file dictionary
        self.i_weekday = self.__load_instance(self.instance_file_weekday)
        self.j_weekday = self.__load_shift(self.shift_file_weekday)
        self.i_weekend = self.__load_instance(self.instance_file_weekend)
        self.j_weekend = self.__load_shift(self.shift_file_weekend)

        #initialize the rest of the class arguments
        self.__compute_data(**kwargs)

    #initialize the other variables
    def __compute_data(self, **kwargs) -> None:
        self.outsourcing_cost = COST_PER_COURIER_AND_PERIOD * self.outsourcing_cost_multiplier

        #region
        self.dregions = self.i_weekday['geography']['city']['regions']
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

        #workforce size dictionary (region dependent)
        self.k = dict()

        self.population = dict()
        for region in self.regions:
            self.population[region] = self.i_weekday['geography']['city']['regions'][region]['population']

        #optimal workforce size already determined
        if self.expand_workforce_to_regions == True:

            #create workforce_file
            city_pattern = r'(\w+)_db'
            db_pattern = r'db=(\d+\.\d+)'

            city_match = re.search(city_pattern, self.shift_file_weekday)
            db_match = re.search(db_pattern, self.shift_file_weekday)

            city = city_match.group(1) if city_match else None
            demand_baseline = db_match.group(1) if db_match else None

            self.workforce_file = f'../workforce_size/{city}_db={demand_baseline}.json'

            #import in the optimal workforce size
            workforce_size = self.__load_workforce(self.workforce_file)

            #scale optimal workforce size to other regions
            self.workforce_dict = {}
            for region in self.regions:
                self.workforce_dict[region] = math.floor(workforce_size*(self.population[region]/self.population[0]))
                self.k[region] = math.floor(workforce_size*(self.population[region]/self.population[0]))
        #searching for optimal workforce size
        else:
            for region in self.regions:
                self.k[region] = self.workforce_dict[region]

        #employees (region) (employee index is distinct)
        counter_ = 0
        self.n_employees = dict()
        self.employees = dict()
        for region in self.regions:
            if region in list(self.k.keys()):
                self.n_employees[region] = self.k[region]
                self.employees[region] = [iter_ for iter_ in range(counter_, counter_ + self.n_employees[region])]
                counter_ += self.n_employees[region]
            else:
                self.n_employees[region] = 0
                self.employees[region] = []

        #day level variables

        #day
        self.n_days = N_DAYS
        self.days = [iter_ for iter_ in range(self.n_days)]

        #periods (day)
        self.n_periods = dict()
        self.periods = dict()
        for day in self.days:
            if day in [0,1,2,3,4]:
                #weekday
                self.n_periods[day] = self.i_weekday['num_time_intervals']
                self.periods[day] = list(range(self.n_periods[day]))
            else:
                #weekend
                self.n_periods[day] = self.i_weekend['num_time_intervals']
                self.periods[day] = list(range(self.n_periods[day]))

        #scenarios (day)
        self.n_scenarios = dict()
        self.scenarios = dict()
        for day in self.days:
            if day in [0,1,2,3,4]:
                #weekday
                self.n_scenarios[day] = self.i_weekday['num_scenarios']
                self.scenarios[day] = list(range(self.n_scenarios[day]))
            else:
                #weekend
                self.n_scenarios[day] = self.i_weekend['num_scenarios']
                self.scenarios[day] = list(range(self.n_scenarios[day]))

        #demand/required (s, a, theta, day)

        self.sdemand = dict()
        self.srequired = dict()

        for day in self.days:
            if day in [0,1,2,3,4]:
                #weekday
                for scenario in self.i_weekday['scenarios']:
                    s = scenario['scenario_num']
                    for data in scenario['data']:
                        a = data['area_id']
                        for theta, d in enumerate(data['demand']):
                            self.sdemand[(s, a, theta, day)] = d

                        for theta, m in enumerate(data['required_couriers']):
                            self.srequired[(s, a, theta, day)] = m
            else:
                #weekend
                for scenario in self.i_weekend['scenarios']:
                    s = scenario['scenario_num']
                    for data in scenario['data']:
                        a = data['area_id']
                        for theta, d in enumerate(data['demand']):
                            self.sdemand[(s, a, theta, day)] = d

                        for theta, m in enumerate(data['required_couriers']):
                            self.srequired[(s, a, theta, day)] = m

        #region, day level variables
        self.n_shifts = dict()
        self.shifts = dict()
        self.shifts_distinct = dict()
        #shifts (region, day)
        for region in self.regions:
            self.shifts_distinct[region] = []
            for day in self.days:
                if day in [0,1,2,3,4]:
                    #weekday
                    if region in list(self.j_weekday.keys()):
                        self.n_shifts[region,day] = len(self.j_weekday[region]['shifts_start'].keys())
                        self.shifts[region,day] = [self.j_weekday[region]['shifts_start'][index_] for index_ in self.j_weekday[region]['shifts_start'].keys()]
                        self.shifts_distinct[region].extend(self.shifts[region,day])
                    else:
                        self.n_shifts[region,day] = 0
                        self.shifts[region,day] = []
                else:
                    #weekend
                    if region in list(self.j_weekend.keys()):
                        self.n_shifts[region,day] = len(self.j_weekend[region]['shifts_start'].keys())
                        self.shifts[region,day] = [self.j_weekend[region]['shifts_start'][index_] for index_ in self.j_weekend[region]['shifts_start'].keys()]
                        self.shifts_distinct[region].extend(self.shifts[region,day])
                    else:
                        self.n_shifts[region,day] = 0
                        self.shifts[region,day] = []
            self.shifts_distinct[region] = list(set(self.shifts_distinct[region]))

        ibasename = splitext(basename(self.instance_file_weekday))[0]
        ibasename_list = ibasename.split('_')
        self.ibasename = f"{ibasename_list[0]}_{ibasename_list[1]}"
        self.name = self.ibasename + f"_oc={self.outsourcing_cost_multiplier}_model{self.model}"

        if self.model == 'partflex':
            self.name = self.ibasename + f"_oc={self.outsourcing_cost_multiplier}_model{self.model}_mu{self.max_n_shifts}"

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

    def __load_workforce(self, workforce_file: str) -> int:
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
        return (int(df_workforce['workforce_size_region0'].tolist()[0])-1)

class Solver:
    args: Namespace
    i: Instance

    m: Model
    
    #decision variables
    k: tupledict #(e,a,theta,day)
    omega: tupledict #(s,a,theta,day)
    r: tupledict #(e,p,day)
    U: tupledict #(e, p)

    def __init__(self, args: Namespace, i: Instance):
        self.args = args
        self.i = i

    def __build_baseline_model(self) -> None:
        self.m = Model()

        #k decision variable (e,a,theta,d)
        self.k = {(employee, area, theta, day): self.m.addVar(vtype = GRB.BINARY, obj = 1, name = 'k')
                  for region in self.i.regions
                  for employee in self.i.employees[region]
                  for area in self.i.reg_areas[region]
                  for day in self.i.days
                  for theta in self.i.periods[day]}

        #omega decision variable (s,a,theta,day)
        self.omega = {(scenario, area, theta, day): self.m.addVar(vtype=GRB.CONTINUOUS, lb = 0, obj=1/self.i.n_scenarios[day], name='omega') 
                    for day in self.i.days 
                    for scenario in self.i.scenarios[day] 
                    for area in self.i.areas 
                    for theta in self.i.periods[day]}

        #constraint - ensuring employee can be in only one area at a time
        self.m.addConstrs((
            sum(self.k[(employee, area, theta, day)]
                for area in self.i.reg_areas[region]
                )
            <= 1
                for region in self.i.regions
                for employee in self.i.employees[region]
                for day in self.i.days
                for theta in self.i.periods[day]
        ), name = 'one_area_at_a_time')

        #constraint - employee can't work more than number of periods available in a shift in a day
        self.m.addConstrs((
            sum(self.k[(employee, area, theta, day)]
                for area in self.i.reg_areas[region]
                for theta in self.i.periods[day]
                )
            <= (1/2)*HOURS_IN_SHIFT_P
                for region in self.i.regions
                for employee in self.i.employees[region]
                for day in self.i.days
        ), name = 'specific_periods_per_day')

        #constraint - demand can be met by outsourcing
        self.m.addConstrs(
            (
                self.i.srequired[(scenario, area, theta, day)] * self.omega[(scenario, area, theta, day)] >= \
                (self.i.srequired[(scenario, area, theta, day)] - 
                sum(
                    self.k[(employee, area, theta, day)] 
                    for employee in self.i.employees[region]
                )) * self.i.sdemand[(scenario, area, theta, day)] * self.i.outsourcing_cost
                
                for region in self.i.regions
                for area in self.i.reg_areas[region]
                for day in self.i.days
                for theta in self.i.periods[day]
                for scenario in self.i.scenarios[day]
            ), name='outsourcing_demand'
        )

    def __build_roster_model(self) -> None:

        #r decision variable (e,p,d)
        self.r = {(employee, shift_start, day): self.m.addVar(vtype=GRB.BINARY, name='r') 
                for region in self.i.regions 
                for employee in self.i.employees[region]
                for day in self.i.days
                for shift_start in self.i.shifts[region, day]}

        #U decision variable (e,p)
        self.U = {(employee, shift_start): self.m.addVar(vtype=GRB.BINARY, name='U') 
                for region in self.i.regions 
                for employee in self.i.employees[region]
                for shift_start in self.i.shifts_distinct[region]}

        # constraint - ensuring employees work whole of shift
        self.m.addConstrs((
            sum(
                self.k[(employee, area, theta, day)]
                for area in self.i.reg_areas[region]
            ) >=
            self.r[(employee, shift_start, day)]
            for region in self.i.regions
            for employee in self.i.employees[region]
            for day in self.i.days
            for shift_start in self.i.shifts[region, day]
            for theta in range(shift_start, shift_start+4)
        ), name='employees_work_whole_shift')

        # constraint - connecting employees moving areas (r and k decision variables)
        self.m.addConstrs((
            sum(
                self.k[(employee, area, theta, day)]
                for area in self.i.reg_areas[region]
                for theta in self.i.periods[day]
            ) ==
            (1/2)*HOURS_IN_SHIFT_P*sum(self.r[(employee, shift_start, day)] for shift_start in self.i.shifts[region,day])
            for region in self.i.regions
            for employee in self.i.employees[region]
            for day in self.i.days
        ), name='connect_employees_moving_areas_all_theta')

        #constraint - ensuring employee can only work on shift in a day
        self.m.addConstrs((
            sum(self.r[(employee, shift_start, day)]
                for shift_start in self.i.shifts[region, day]
                )
            <= 1
                for region in self.i.regions
                for employee in self.i.employees[region]
                for day in self.i.days
        ), name = 'one_shift_a_day')

        #constraint - ensuring employee will have at least one rest day a week
        self.m.addConstrs((
            sum(self.r[(employee, shift_start, day)]
                for day in self.i.days
                for shift_start in self.i.shifts[region, day]
                )
            <= N_WORK_DAYS
                for region in self.i.regions
                for employee in self.i.employees[region]
        ), name = 'n_rest_day_per_week')

        #constraint - ensuring employees work a minimum number of hours
        self.m.addConstrs((
            sum(HOURS_IN_SHIFT_P*self.r[(employee, shift_start, day)]
                for day in self.i.days
                for shift_start in self.i.shifts[region, day]
            )
            >= self.i.h_min
                for region in self.i.regions
                for employee in self.i.employees[region]
        ), name = 'min_hours_worked')

        #constraint - ensuring employees work a maximum number of hours
        self.m.addConstrs((
            sum(HOURS_IN_SHIFT_P*self.r[(employee, shift_start, day)]
                for day in self.i.days
                for shift_start in self.i.shifts[region, day]
            )
            <= self.i.h_max
                for region in self.i.regions
                for employee in self.i.employees[region]
        ), name = 'max_hours_worked')

        #constraint - employees have a max number of different shift starting times
        self.m.addConstrs(
            (self.r[(employee, shift_start, day)] <= self.U[(employee, shift_start)] 
                for region in self.i.regions
                for employee in self.i.employees[region]
                for day in self.i.days
                for shift_start in self.i.shifts[region, day]
            ), name='max_different_start1'
        )

        self.m.addConstrs((
            sum(self.U[(employee, shift_start)]
                for shift_start in self.i.shifts_distinct[region]
            )
            <= self.i.b_max
                for region in self.i.regions
                for employee in self.i.employees[region]
        ), name = 'max_different_start2')

    def __basic_output(self) -> dict:

        #decision variables
        #   k: tupledict #(e,a,theta,day)
        k = {}
        for key, value in self.k.items():
            if value.X > 0:
                k[key] = value.X

        #   omega: tupledict #(s,a,theta,day)
        omega = {}
        for key, value in self.omega.items():
            if value.X > 0:
                omega[key] = value.X

        output = {
            'instance': [self.i.ibasename],
            'city': [self.i.ibasename.split('_')[0]],
            'demand_baseline': [self.i.i_weekday['demand_baseline']],
            'outsourcing_cost_multiplier': [self.args.outsourcing_cost_multiplier],
            'region_multiplier': [self.i.reg_multiplier],
            'global_multiplier': [self.i.glb_multiplier],
            'model': [self.args.model],
            'elapsed_time': self.m.Runtime,
            'n_variables': self.m.NumVars,
            'n_constraints': self.m.NumConstrs,
            'n_nonzeroes': self.m.NumNZs,
            'k': k, 
            'omega': omega, 
        }
        if self.i.model == 'partflex':
            output['max_n_shifts'] = [self.i.max_n_shifts]
        else:
            output['max_n_shifts'] = [np.nan]
        return output

    def __roster_objval(self) -> dict:
        basic_output = self.__basic_output()

        total_employees = 0
        for region in self.i.regions:
            total_employees += self.i.n_employees[region]
        basic_output['workforce_size'] = [total_employees]

        #can only get optimal value if there was one
        if self.m.Status == GRB.OPTIMAL:
            k = sum(self.k[(employee, area, theta, day)].X for region in self.i.regions for area in self.i.reg_areas[region] for employee in self.i.employees[region] for day in self.i.days for theta in self.i.periods[day])
            basic_output['wage_costs'] = [k]
            basic_output['objective_value'] = [self.m.ObjVal]
            basic_output['objective_value_post_wage'] = [self.m.ObjVal - k]
        else:
            basic_output['wage_costs'] = [np.nan]
            basic_output['objective_value'] = [np.nan]
            basic_output['objective_value_post_wage'] = [np.nan]
        
        return basic_output

    def __roster_output(self) -> dict:
        basic_output = self.__roster_results()

        check_output = {}
        for key in basic_output.keys():
            check_output[key] = basic_output[key]
        check_output['region'] = []
        check_output['employee'] = []
        check_output['day'] = []
        check_output['sum_r'] = []
        check_output['shift_start'] = []
        for region in self.i.regions:
            for employee in self.i.employees[region]:
                for day in self.i.days:
                    sum_r = 0
                    shift_start_ = 0
                    for shift_start in self.i.shifts[region, day]:
                        if self.m.Status == GRB.OPTIMAL:
                            sum_r += int(self.r[(employee, shift_start, day)].X)
                            if int(self.r[(employee, shift_start, day)].X) > .9:
                                shift_start_ = shift_start
                    check_output['region'].append(region)
                    check_output['employee'].append(employee)
                    check_output['day'].append(day)
                    check_output['sum_r'].append(sum_r)
                    check_output['shift_start'].append(shift_start_)

        # check_output['sum_k'] = []
        # check_output['k_worked'] = []
        # for region in self.i.regions:
        #     for employee in self.i.employees[region]:
        #         for day in self.i.days:
        #             sum_k = 0
        #             k_worked = []
        #             for theta in self.i.periods[day]:
        #                 if sum_r > 0:
        #                     k = sum(
        #                         self.k[(employee, area, theta, day)].X
        #                         for area in self.i.reg_areas[region]
        #                     )
        #                 else:
        #                     k = 0

        #                 sum_k += k
        #                 if k > .9:
        #                     k_worked.append(theta)
        #             check_output['sum_k'].append(sum_k)
        #             check_output['k_worked'].append(k_worked)
        #             check_output['k_worked'].sort()

        len_ = len(check_output['region'])
        for key in basic_output.keys():
            check_output[key] = check_output[key]*len_

        return check_output

    def __roster_output(self):
        #decision variables
        #   k: tupledict #(e,a,theta,day)
        k = {}
        for key, value in self.k.items():
            if value.X > 0:
                k[key] = value.X

        #   omega: tupledict #(s,a,theta,day)
        omega = {}
        for key, value in self.omega.items():
            if value.X > 0:
                omega[key] = value.X

        #   r: tupledict #(e,p,day)
        r = {}
        for key, value in self.r.items():
            if value.X > 0:
                r[key] = value.X

        #   U: tupledict #(e, p)
        U = {}
        for key, value in self.U.items():
            if value.X > 0:
                U[key] = value.X

        results = {
            # Instance
            'elapsed_time': self.m.Runtime,
            'n_variables': self.m.NumVars,
            'n_constraints': self.m.NumConstrs,
            'n_nonzeroes': self.m.NumNZs,
            'regions': self.i.regions,
            'reg_areas': self.i.reg_areas,
            'shifts': self.i.shifts,
            'days': self.i.days,
            'employees': self.i.employees,
            'area': self.i.reg_areas,
            'periods': self.i.periods,
            'hmax': self.i.h_max,
            'hmin': self.i.h_min,
            # Model output
            'obj_val': self.m.ObjVal, 
            'status': self.m.status, 
            'gap': self.m.MIPGap,
            # Decision vars
            'k': k, 
            'omega': omega, 
            'r': r, 
            'U': U,
        }

        return results


    def solve_roster_objval(self) -> dict:
        self.__build_baseline_model()
        self.__build_roster_model()
        self.m.setParam("OutputFlag", 0) # No logs
        self.m.optimize()
        return self.__roster_objval()

    def solve_baseline_objval(self) -> dict:
        self.__build_baseline_model()
        self.m.setParam("OutputFlag", 0) # No logs
        self.m.optimize()
        return self.__roster_objval()

    def solve_roster_output(self) -> dict:
        self.__build_baseline_model()
        self.__build_roster_model()
        self.m.setParam("OutputFlag", 0) # No logs
        self.m.optimize()
        return self.__roster_output()

#function call run execution
def run_roster_solver_objval(model, instance_file_weekday, shift_file_weekday, instance_file_weekend, shift_file_weekend, workforce_dict, outsourcing_cost_multiplier, regional_multiplier, global_multiplier, h_min, h_max, max_n_diff, max_n_shifts=None, expand_workforce_to_regions=None):
    args = Namespace(
        model=model,
        instance_file_weekday=instance_file_weekday,
        shift_file_weekday = shift_file_weekday,
        instance_file_weekend=instance_file_weekend,
        shift_file_weekend = shift_file_weekend,

        expand_workforce_to_regions=expand_workforce_to_regions,
        workforce_dict = workforce_dict,

        outsourcing_cost_multiplier=outsourcing_cost_multiplier,
        regional_multiplier=regional_multiplier,
        global_multiplier=global_multiplier,

        h_min = h_min,
        h_max = h_max,
        max_n_diff = max_n_diff,
        max_n_shifts=max_n_shifts
    )

    i = Instance(args=args)
    solver = Solver(args=args, i=i)

    roster_results = solver.solve_roster_objval()

    return roster_results

def run_roster_solver_objval_w_baseline(model, instance_file_weekday, 
                                        shift_file_weekday, instance_file_weekend, 
                                        shift_file_weekend, workforce_dict, 
                                        outsourcing_cost_multiplier, regional_multiplier, global_multiplier, 
                                        h_min, h_max, max_n_diff, 
                                        max_n_shifts=None, expand_workforce_to_regions=None):
    args = Namespace(
        model=model,
        instance_file_weekday=instance_file_weekday,
        shift_file_weekday = shift_file_weekday,
        instance_file_weekend=instance_file_weekend,
        shift_file_weekend = shift_file_weekend,

        expand_workforce_to_regions=expand_workforce_to_regions,
        workforce_dict = workforce_dict,

        outsourcing_cost_multiplier=outsourcing_cost_multiplier,
        regional_multiplier=regional_multiplier,
        global_multiplier=global_multiplier,

        h_min = h_min,
        h_max = h_max,
        max_n_diff = max_n_diff,
        max_n_shifts=max_n_shifts
    )

    i = Instance(args=args)
    solver = Solver(args=args, i=i)

    baseline_results = solver.solve_baseline_objval()
    roster_results = solver.solve_roster_output()

    return baseline_results, roster_results

def run_roster_solver_output(model, instance_file_weekday, shift_file_weekday, instance_file_weekend, shift_file_weekend, workforce_dict, outsourcing_cost_multiplier, regional_multiplier, global_multiplier, h_min, h_max, max_n_diff, max_n_shifts=None, expand_workforce_to_regions=None):
    args = Namespace(
        model=model,
        instance_file_weekday=instance_file_weekday,
        shift_file_weekday = shift_file_weekday,
        instance_file_weekend=instance_file_weekend,
        shift_file_weekend = shift_file_weekend,

        expand_workforce_to_regions=expand_workforce_to_regions,
        workforce_dict = workforce_dict,

        outsourcing_cost_multiplier=outsourcing_cost_multiplier,
        regional_multiplier=regional_multiplier,
        global_multiplier=global_multiplier,

        h_min = h_min,
        h_max = h_max,
        max_n_diff = max_n_diff,
        max_n_shifts=max_n_shifts
    )

    # Assuming Instance and Solver classes are defined above in this script
    i = Instance(args=args)
    solver = Solver(args=args, i=i)

    roster_results, jonny_results = solver.solve_roster_output()

    return roster_results, jonny_results
