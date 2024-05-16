import pandas as pd
import json
from gurobipy import Model, GRB
import re

def create_roster_json(city, demand_baseline, model, week_demand_type, weekend_demand_type, regional_multiplier, global_multiplier, outsourcing_cost_multiplier):

    # Create week
    days = pd.DataFrame([
        [0, demand_baseline, week_demand_type],  # Monday
        [1, demand_baseline, week_demand_type],  # Tuesday
        [2, demand_baseline, week_demand_type],  # Wednesday
        [3, demand_baseline, week_demand_type],  # Thursday
        [4, demand_baseline, week_demand_type],  # Friday
        [5, demand_baseline, weekend_demand_type],  # Saturday
        [6, demand_baseline, weekend_demand_type],  # Sunday
    ], columns=['day', 'demand_baseline', 'demand_type'])

    # Create instance
    master_data = {
        'name': f'{city}_db={days.iloc[0]["demand_baseline"]}',
        'num_days': len(days),
        'num_time_intervals': [],
        'regional_multiplier': regional_multiplier,
        'global_multiplier': global_multiplier,
        'outsourcing_cost_multiplier': outsourcing_cost_multiplier,
        'model': model,
        'week_demand_type': week_demand_type,
        'weekend_demand_type': weekend_demand_type,
        'demand_baseline': [],
        'geography': [],
        'days': []
    }

    # Iterate through days
    for i, row in days.iterrows():
        day_index, demand_baseline, demand_type = row['day'], row['demand_baseline'], row['demand_type']

        # Load file
        instance_file = f'../instances/{city}_db={demand_baseline}_dt={demand_type}.json'
        with open(instance_file, 'r') as file:
            instance_data = json.load(file)

        # Save global information
        if day_index == 0:
            master_data['num_time_intervals'] = instance_data['num_time_intervals']
            master_data['demand_baseline'] = instance_data['demand_baseline']
            master_data['geography'] = instance_data['geography']


        # Load shift data
        shift_file = f'../shifts/{city}_db={demand_baseline}_dt={demand_type}.json'
        with open(shift_file, 'r') as file:
            shift_data = json.load(file)

        # Filter shift data based on the parameters
        if 'partflex' in model:
            model_word = re.findall(r'[a-zA-Z]+', model)[0]
            model_number = re.findall(r'\d+', model)[0]
            # Run the query for partflex
            shifts_df = (
                pd.DataFrame(shift_data)
                .query(f'outsourcing_cost_multiplier == {outsourcing_cost_multiplier} & demand_type == "{demand_type}" & model == "{model_word}" & max_n_shifts == {model_number}')
                .query(f'regional_multiplier == {regional_multiplier} & global_multiplier == {global_multiplier}')
            )
        else:
            shifts_df = (
                pd.DataFrame(shift_data)
                .query(f'outsourcing_cost_multiplier == {outsourcing_cost_multiplier} & demand_type == "{demand_type}" & model == "{model}"')
                .query(f'regional_multiplier == {regional_multiplier} & global_multiplier == {global_multiplier}')
            )

        # Save shift data
        shifts = []
        for _, shift in shifts_df.iterrows():
            shift_data = {
                'id': shift['region'],
                'shifts_start': list(shift['shifts_start'].values()),
                'shifts_end': list(shift['shifts_end'].values()),
            }
            shifts.append(shift_data)

        day_data = {
            'id': day_index,
            'num_scenarios': instance_data['num_scenarios'],
            'scenarios': instance_data['scenarios'],
            'region': shifts
        }
        master_data['days'].append(day_data)

    master_file = f'rostering_instances/{city}_db={demand_baseline}_weekdt={week_demand_type}_weekenddt={weekend_demand_type}_cout={outsourcing_cost_multiplier}_model={model}_rm={regional_multiplier}_gm={global_multiplier}.json'
    with open(master_file, 'w') as file:
        json.dump(master_data, file, indent=4)


class Instance:
    def __init__(self, instance_file, n_periods=8, n_hours=8, cost_courier=1, min_hours_worked=6*8, max_hours_worked=6*8, max_unique_starts = 2, shift_length = 4, working_days = 6):
        self.name = instance_file
        self.i = self.__load_instance(instance_file)
        self.n_periods = n_periods
        self.n_hours = n_hours
        self.cost_courier = cost_courier
        self.min_hours_worked = min_hours_worked
        self.max_hours_worked = max_hours_worked
        self.max_unique_starts = max_unique_starts
        self.shift_length = shift_length
        self.working_days = working_days
        self.__compute_data()

    # Load JSON instance
    def __load_instance(self, instance: str) -> dict:
        with open(instance) as f:
            return json.load(f)

    # Compute data
    def __compute_data(self) -> None:
        # Retrieve basic parameters from the instance file
        self.n_days = self.i['num_days']
        self.n_time_intervals = self.i['num_time_intervals']
        self.regional_multiplier = self.i['regional_multiplier']
        self.global_multiplier = self.i['global_multiplier']
        self.outsourcing_cost_multiplier = self.i['outsourcing_cost_multiplier']
        self.demand_baseline = self.i['demand_baseline']
        self.outsourcing_cost = 1.0 * self.outsourcing_cost_multiplier
        self.model = self.i['model']
        self.week_demand_type = self.i['week_demand_type']
        self.weekend_demand_type = self.i['weekend_demand_type']
    

        # Retrieve region data
        self.dregions = self.i['geography']['city']['regions']
        self.n_regions = len(self.dregions)
        self.n_areas = sum(len(region['areas']) for region in self.dregions)

        # Time intervals and periods
        self.periods = list(range(self.n_time_intervals))

        # Days data
        self.ddays = self.i['days']
        self.days = [day['id'] for day in self.ddays]

        # Extract regions and areas
        self.regions = [region['id'] for region in self.dregions]
        self.areas = [area['id'] for region in self.dregions for area in region['areas']]

        # Map areas to their respective regions
        self.region_area_map = {
            region['id']: [area['id'] for area in region['areas']] for region in self.dregions
        }
        self.area_map = {
            area: region for region, areas in self.region_area_map.items() for area in areas
        }

        # Number of scenarios
        self.n_scenarios = {
            day['id']: day['num_scenarios'] for day in self.ddays
        }

        # list of scenarios
        self.scenarios = {
            day['id']: [scenario['scenario_num'] for scenario in day['scenarios']] for day in self.ddays
        }

        # Shift 
        self.shifts = {
            day['id']: {region['id']: [[start_time + i for i in range(4)] for start_time in region['shifts_start']]
                for region in day['region']}
            for day in self.ddays
        }

        # Shift indexs 
        self.shifts_index = {
            day['id']: {region['id']: [i for i in range(len(region['shifts_start']))]
                for region in day['region']}
            for day in self.ddays
        }
    
        # Shift start times
        self.shifts_start = {
            day['id']: {region['id']: region['shifts_start'] for region in day['region']}
            for day in self.ddays
        }


        # Demand data
        self.period_demands = {
            (day['id'], scenario['scenario_num'], data['area_id'], theta): d
            for day in self.ddays
            for scenario in day['scenarios']
            for data in scenario['data']
            for theta, d in enumerate(data['demand'])
        }

        # Couriers 
        self.period_couriers = {
            (day['id'], scenario['scenario_num'], data['area_id'], theta): m
            for day in self.ddays
            for scenario in day['scenarios']
            for data in scenario['data']
            for theta, m in enumerate(data['required_couriers'])
        }

        # Regionaly employee population
        region_population = {region['id']: region['population'] for region in self.dregions} 

        self.region_employees = {
            region: 5
            for region in self.regions
        }
        self.n_employees = sum(value for reg, value in self.region_employees.items())


class Solver():
    def __init__(self, i):
        self.i = i

    def solve_base(self) -> dict:
        self.__build_base_model()
        self.m.optimize()
        self.i.model_name = "baseline"
        return self.__basic_results()
    
    def solve_roster(self) -> dict:
        self.__build_roster_model()
        self.m.optimize()
        self.i.model_name = "roster"
        return self.__basic_results()

    def __basic_results(self) -> dict:
        results = {
            'instance': self.i.name,
            'model': self.i.model_name,
            'shift_type': self.i.model,
            'week_demand_type': self.i.week_demand_type,
            'weekend_demand_type': self.i.weekend_demand_type,
            'obj_value': self.m.ObjVal,
            'elapsed_time': self.m.Runtime,
            'n_variables': self.m.NumVars,
            'n_constraints': self.m.NumConstrs,
            'n_nonzeroes': self.m.NumNZs
        }
        return results
        
    def __build_base_model(self):
         # Assign employees to regions
        self.E = {}
        prev_start = 0
        for r in self.i.regions:
            self.E[r] = [prev_start+i for i in range(self.i.region_employees[r])]
            prev_start += len(self.E[r])
        
        self.m = Model()

        ### VARIABLES
        # Courier assignment to area by period and day
        self.k_vars = {}
        for d in self.i.days:
            for r in self.i.regions:
                self.k_vars[d, r] = self.m.addVars(
                    self.E[r], 
                    self.i.region_area_map[r], 
                    self.i.periods, 
                    vtype=GRB.BINARY, 
                    obj = 1,
                    lb = 0,
                    name=f'k_{r}'
                )

        # Total employees assigned to area by period and day (not necessary but helpful for outsourcing constrains)
        self.n_hired = {}
        for d in self.i.days:
            for r in self.i.regions:
                self.n_hired[d, r] = self.m.addVars(
                    self.i.region_area_map[r], 
                    self.i.periods, 
                    vtype=GRB.INTEGER, 
                    name=f'n_{r}'
                )
        # Outsourcing cost
        self.omega_vars = {}
        for d in self.i.days:
            for r in self.i.regions:
                self.omega_vars[d, r] = self.m.addVars(
                    self.i.region_area_map[r],
                    self.i.periods, 
                    self.i.scenarios[d], 
                    vtype=GRB.CONTINUOUS,
                    obj=1/self.i.n_scenarios[d],
                    lb=0,
                    name=f'omega_{r}'
                )

        ### CONSTRAINTS
        # Employee can only be assigned to one area at a time 
        for d in self.i.days:
            for r in self.i.regions:
                Ar = self.i.region_area_map[r]
                self.m.addConstrs((
                    sum(self.k_vars[d, r][e, a, theta] for a in Ar) <= 1
                    for e in self.E[r]
                    for theta in self.i.periods),
                    name=f'employee_{r}_{d}_single_assignment'
                )

        # Employee can work max number of periods in a day
        for d in self.i.days:
            for r in self.i.regions:   
                self.m.addConstrs((
                    sum(self.k_vars[d, r][e, a, theta] for a in self.i.region_area_map[r] for theta in self.i.periods) <= self.i.shift_length
                    for e in self.E[r]),
                    name=f'employee_max_periods_{r}_{d}'
                )

        # Number assinged to area is the sum of the k_vars
        for d in self.i.days:
            for r in self.i.regions:
                self.m.addConstrs((
                    self.n_hired[d, r][a, theta] == sum(self.k_vars[d, r][e, a, theta] for e in self.E[r])
                    for a in self.i.region_area_map[r]
                    for theta in self.i.periods),
                    name=f'employee_{r}_{d}_assignment'
                )
    
        # Outsourcing cost
        for d in self.i.days: 
            for r in self.i.regions:      
                self.m.addConstrs((
                            self.i.period_couriers[d, s, a, theta] * self.omega_vars[d,r][a, theta, s] >= \
                            (self.i.period_couriers[d, s, a, theta] - self.n_hired[d, r][a, theta]) * self.i.period_demands[d, s, a, theta] * self.i.outsourcing_cost
                            for a in self.i.region_area_map[r]
                            for theta in self.i.periods
                            for s in self.i.scenarios[d]
                        ), name=f'set_omega_{d}')  

    def __build_roster_model(self):
        self.__build_base_model()

        ### VARIABLES
        # Shift start indicator
        self.r_vars = {}
        for d in self.i.days:
            for r in self.i.regions:
                self.r_vars[d, r] = self.m.addVars(
                    self.E[r], 
                    self.i.periods, 
                    vtype=GRB.BINARY, 
                    name=f'k_{r}'
                )

        # Count if employee starts a shift at a certain period
        self.u_vars=  {}
        for r in self.i.regions:
            self.u_vars[r] = self.m.addVars(
                self.E[r],
                self.i.periods, 
                vtype=GRB.BINARY, 
                name=f'shift_start_{r}'
            )

        ### CONSTRAINTS
        # Ensure employees are assigned to shifts, not sporadic periods
        for d in self.i.days:
            for r in self.i.regions:
                for s in self.i.shifts_start[d][r]:
                    self.m.addConstrs((
                        sum(self.k_vars[d, r][e, a, theta] for a in self.i.region_area_map[r] 
                            for theta in range(s, s+4)) == \
                        self.i.shift_length*self.r_vars[d, r][e, s]
                        for e in self.E[r]),
                        name=f'employee_{r}_{d}_shift_assignment'
                    )   
    
        # Limit the number of working days for each employee
        for r in self.i.regions:
            self.m.addConstrs((
                sum(self.r_vars[d, r][e, p] for d in self.i.days for p in self.i.periods) <= self.i.working_days
                for e in self.E[r]),
                name=f'employee_{r}_working_days'
            )

        # Min/max hours worked per employee per week
        for r in self.i.regions:
            self.m.addConstrs((
                sum(self.r_vars[d, r][e, p] for d in self.i.days for p in self.i.periods)*self.i.shift_length*2 >= self.i.min_hours_worked
                for e in self.E[r]),
                name=f'employee_{r}_min_hours'
            )
            self.m.addConstrs((
                sum(self.r_vars[d, r][e, p] for d in self.i.days for p in self.i.periods)*self.i.shift_length*2 <= self.i.max_hours_worked
                for e in self.E[r]),
                name=f'employee_{r}_max_hours'
            )
        
        # Link shift start indicator to start time count indicator
        for d in self.i.days:
            for r in self.i.regions:
                self.m.addConstrs((
                    self.r_vars[d, r][e, p] <= self.u_vars[r][e, p]  
                    for e in self.E[r]
                    for p in self.i.periods
                ), name=f'max_unique_starts_{r}_{d}')
        
        # Limit the number of unique shift starts
        for r in self.i.regions:
            self.m.addConstrs((
                sum(self.u_vars[r][e, p] for p in self.i.periods) <= self.i.max_unique_starts
                for e in self.E[r]),
                name=f'max_unique_starts_{r}'
            )

