import numpy as np

from bokeh.plotting import figure, output_file, show, gridplot
from bokeh.models import LinearAxis, Range1d
from datetime import datetime
import math
import itertools
from itertools import chain
from enum import Enum

#############################################################################################
# Utility functions
#############################################################################################


def log(*argv):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S:%f")
    print(" ".join([str(arg) for arg in [current_time] + list(argv)]))


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


# Adds two arrays element by element, with padding.  If one array is larger than the other, the smaller array
# will be 'padded' with the last value to match the size of the larger array.
# Example: [1, 2, 3, 0] + [4, 5] = [1 + 4, 2 + 5, 3 + 5, 0 + 5]
def add_arrays_wp(a1, a2):
    if len(a1) <= len(a2):
        a = a1
        b = a2
    else:
        a = a2
        b = a1

    c = b.copy()
    for i in range(len(a)):
        c[i] = a[i] + b[i]
    if len(a) > 0:
        for i in range(len(a), len(b)):
            c[i] = a[len(a) - 1] + b[i]

    return c


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def multiply_list(a, value):
    for i in range(len(a)):
        a[i] *= value


class MultiLayerDict:
    def __init__(self):
        self.dict = dict()

    def add(self, keys, values):
        dictionary = self.dict
        for index, key in enumerate(keys):
            if key not in dictionary:
                if index < len(keys) - 1:
                    dictionary[key] = dict()
                else:
                    dictionary[key] = set()
            if index < len(keys) - 1:
                dictionary = dictionary[key]
            else:
                leaf_set = dictionary[key]

        leaf_set.update(values)

    def get(self, keys):
        value = self.dict
        for key in keys:
            if not isinstance(value, dict):
                raise Error()
            if key not in value:
                return []
            value = value[key]
        return value

    def remove(self, keys, values = None):
        dictionary = self.dict
        self.remove_internal(dictionary, keys, 0, values)

    def remove_internal(self, dictionary, keys, index, values):
        key = keys[index]
        if key in dictionary:
            if index < len(keys) - 1:
                sub_dict = dictionary[key]
                self.remove_internal(sub_dict, keys, index + 1, values)
                if len(sub_dict) == 0:
                    dictionary.pop(key)
            else:
                leaf_set = dictionary[key]
                if values:
                    leaf_set.difference_update(values)
                else:
                    leaf_set = set()
                if len(leaf_set) == 0:
                    dictionary.pop(key)

    def enumerate(self):
        return self.enumerate_internal(self.dict, [])

    def enumerate_internal(self, container, keys):
        if isinstance(container, dict):
            return itertools.chain.from_iterable((self.enumerate_internal(value, keys + [key]) for key, value in container.items()))
        else:
            return itertools.chain.from_iterable((keys, value) for value in container)

#############################################################################################
# Random population generation and selection functions
#############################################################################################

# Gamma distribution
#
# Parameters: shape = k and scale = theta (used by numpy)
#
# Alternative representation:
# shape alpha = k, rate beta = 1 / theta
#
# Our parameters: mean and CV
#
# Mean = mi = alpha / beta = alpha * theta
# CV = stddev / mean
# Variance = Sigma2 = alpha / beta2 = alpha * theta2
# Stddev = sigma = sqrt(alpha)/beta = sqrt(alpha)*theta
# CV = sigma / mi = 1 / sqrt(alpha)
#
# alpha = 1 / CV2 // shape
# theta = mean / alpha
# theta = mean * CV2 // scale
class Gamma:
    def __init__(self, mean, CV):
        self.mean = mean
        self.CV = CV

    def generate_population(self, pop_size):
        shape = 1 / (self.CV * self.CV)
        scale = self.mean / shape
        log("shape", shape, "scale", scale)
        return np.random.gamma(shape, scale, pop_size)

    def select(self, probs, num_samples):
        return np.random.choice(len(probs), num_samples, False, probs)


class Constant:
    def __init__(self, value):
        self.value = value

    def generate_population(self, pop_size):
        return np.full(pop_size, self.value)

    # Do not pass a probability array since it is uniform, makes the selection faster.
    def select(self, probs, num_samples):
        return np.random.choice(len(probs), num_samples, False)


class Step:
    def __init__(self, steps, factor):
        self.steps = steps
        self.factor = factor

    def generate_population(self, pop_size):
        population = np.full(pop_size, 1)
        for i in range(0, pop_size):
            step = int(i * self.steps / pop_size)
            population[i] = math.pow(self.factor, step)
        return population * (1 / np.mean(population))

    def select(self, probs, num_samples):
        return np.random.choice(len(probs), num_samples, False, probs)

#############################################################################################
# SEIR model logic
#############################################################################################


class GlobalParameters:
    def __init__(self, days_exposed, days_infectious, r0, distribution):
        self.days_exposed = days_exposed
        self.days_infectious = days_infectious
        self.r0 = r0
        self.distribution = distribution

        self.daily_infectiousness = r0 / days_infectious


class PopulationParameters:
    def __init__(self, primary_population_size, secondary_population_size=0, tertiary_population_size=0,
                 num_secondary_locations=0, num_tertiary_locations=0, daily_primary_secondary_trips=0,
                 daily_secondary_tertiary_trips=0):
        self.primary_population_size = primary_population_size
        self.secondary_population_size = secondary_population_size
        self.tertiary_population_size = tertiary_population_size
        self.num_secondary_locations = num_secondary_locations
        self.num_tertiary_locations = num_tertiary_locations
        self.daily_primary_secondary_trips = daily_primary_secondary_trips
        self.daily_secondary_tertiary_trips = daily_secondary_tertiary_trips


class SeirState(Enum):
    S = 1,
    E = 2,
    I = 3,
    R = 4


# Represents the population at one location, for example, one city.
class Population:
    def __init__(self, population, pars):
        self.day = 0
        # An instance of GlobalParameters above.
        self.pars = pars

        # Each individual in the population has the following modelled attributes:
        # * An implicit 0-based ID.
        # * Relative susceptibility
        # * State: One of S, E, I, R of the SEIR model.
        # * State change day: The day the individual changed to the current state.
        # We capture each attribute in a separate array instead of having an Individual class, to make
        # some calculations faster.
        self.pop = population
        # Convert the relative susceptibility to a probability of selection of each individual.  It has to add up to 1.
        self.prob = self.pop * (1 / np.sum(self.pop))
        # SEIR state of each individual
        self.state = [SeirState.S for _ in population]
        # The day when the individual last changed its state.
        self.state_change_day = [0 for _ in population]

        # Number of individuals newly exposed today.  Is updated as individuals change into the E state.
        self.num_exposed_today = 0
        # Number of individuals newly exposed the day before.
        self.num_exposed_last_day = 0

        # Individuals grouped by state, to optimize querying individuals by state.
        self.state_sets = {
            SeirState.S: set(range(len(population))),
            SeirState.E: set(),
            SeirState.I: set(),
            SeirState.R: set()}

        # Individuals by state and day of change, to optimize querying individuals by state and date of state change.
        self.state_by_day_sets = MultiLayerDict()

        for i in range(len(population)):
            self.state_by_day_sets.add((self.state[i], self.state_change_day[i]), [i])

        # Optimization: Keep track of the sum of probabilities of already exposed individuals.
        self.exposed_probability = 0
        self.total_susceptibility = np.sum(self.pop)
        self.exposed_susceptibility = 0

        # The mean susceptibility of the non-exposed population.  Will be recalculated at the end of each day.
        self.ms = 0
        self.prev_ms = 0

        # The individuals who have traveled away.  They will not be taken into account when
        # calculating the newly exposed at this location.
        self.travelers = set()

        # The individuals from other locations who are visiting.  They can be exposed in this location.
        # If they are infectious, they can also infect others at this location.
        self.visitors = dict()

        self.calc_mean_susceptibility()

    def num_exposed(self):
        return self.size() - self.count(SeirState.S)

    # Exposes count individuals.
    def expose(self, count):
        (visitors_to_expose, locals_to_expose) = self.split_exposures_by_group(count)

        self.expose_visitors(visitors_to_expose)
        self.expose_locals_and_filter_out_travelers(locals_to_expose)

    def expose_locals_and_filter_out_travelers(self, locals_to_expose):
        # We need to loop because in case we select some travelers, those selections will be
        # discarded and we will try again.
        locals_exposed = 0

        while locals_exposed != locals_to_expose:
            draws = self.pars.distribution.select(self.prob, locals_to_expose - locals_exposed)
            draws = [draw for draw in draws if draw not in self.travelers]
            locals_exposed += len(draws)
            draws = [draw for draw in draws if self.state[draw] == SeirState.S]
            for draw in draws:
                self.set_state(draw, SeirState.E)

    # End-of-day updates: promote individuals between SEIR buckets, update
    # rolling variables.
    def roll(self):
        self.calc_mean_susceptibility()

        e_maturity_day = self.day - self.pars.days_exposed
        ei_move = list(self.state_by_day_sets.get((SeirState.E, e_maturity_day)))

        for index in ei_move:
            self.set_state(index, SeirState.I)

        i_maturity_day = self.day - self.pars.days_infectious
        ir_move = list(self.state_by_day_sets.get((SeirState.I, i_maturity_day)))

        for index in ir_move:
            self.set_state(index, SeirState.R)

        self.day += 1
        self.num_exposed_last_day = self.num_exposed_today
        self.num_exposed_today = 0

    def exposed_last_day(self):
        return self.num_exposed_last_day

    def mean_susceptibility(self):
        return self.ms

    def size(self):
        return len(self.pop)

    def calc_mean_susceptibility(self):
        num_susceptible = len(self.state_sets[SeirState.S])
        if num_susceptible == 0:
            self.ms = self.prev_ms
        else:
            self.prev_ms = self.ms
            self.ms = (self.total_susceptibility - self.exposed_susceptibility) / num_susceptible

    # Select a number of individuals to travel away.
    def travel(self, num_trips):
        travelers = self.select_travelers(num_trips)
        self.travelers.update(travelers)
        return travelers

    def select_travelers(self, num_travelers):
        new_travelers = set()
        while len(new_travelers) < num_travelers:
            # log("Attempting to select", num_travelers, "selected", len(new_travelers), "so far.")
            extra_travelers = np.random.choice(len(self.pop), num_travelers - len(new_travelers), False)
            extra_travelers = [t for t in extra_travelers if t not in self.travelers and t not in new_travelers]
            new_travelers.update(extra_travelers)
        # log("Done selecting travelers")
        return new_travelers

    # Returns the list of tavelers to this population.
    def return_travelers(self, travelers):
        if travelers:
            self.travelers.difference_update(travelers)
            # log("Returning: travelers has now", len(self.travelers))

    # Adds travelers from another location to this location's list of visitors.
    def visit(self, travelers, pop):
        self.visitors[pop] = travelers

    def enum_visitors(self):
        return chain.from_iterable([[(pop, visitor) for visitor in visitors]
                                    for pop, visitors in self.visitors.items()])

    def num_infectious_visitors(self):
        return len([1 for pop, visitor in self.enum_visitors() if pop.state[visitor] == SeirState.I])

    def num_infectious_travelers(self):
        return len([1 for traveler in self.travelers if self.state[traveler] == SeirState.I])

    def split_exposures_by_group(self, count):
        # relative probability of exposing someone from this population or exposing some visitor
        v_susceptibility = sum(pop.pop[visitor] for pop, visitor in self.enum_visitors())
        t_susceptibility = sum(self.pop[traveler] for traveler in self.travelers)

        local_probs = self.total_susceptibility - t_susceptibility
        choices = np.random.uniform(0, v_susceptibility + local_probs, count)
        visitors_exposed = np.count_nonzero(choices < v_susceptibility)
        locals_exposed = count - visitors_exposed
        return visitors_exposed, locals_exposed

    def expose_visitors(self, visitors_to_expose):
        visitors_and_pops = [(pop, visitor, pop.state[visitor]) for (pop, visitor) in self.enum_visitors()]

        if len(visitors_and_pops) == 0:
            return

        visitor_probs = [pop.pop[visitor] for (pop, visitor, _) in visitors_and_pops]
        factor = 1 / sum(visitor_probs)
        visitor_probs = [prob * factor for prob in visitor_probs]

        draws = self.pars.distribution.select(visitor_probs, visitors_to_expose)
        newly_exposed = (visitors_and_pops[draw][2] == SeirState.S for draw in draws)
        for exposed in newly_exposed:
            pop = visitors_and_pops[exposed][0]
            visitor = visitors_and_pops[exposed][1]
            pop.set_state(visitor, SeirState.E)

    def num_infectious_at_this_location(self):
        return self.count(SeirState.I) - self.num_infectious_travelers() + self.num_infectious_visitors()

    def spread(self):
        infectious = self.num_infectious_at_this_location()
        # log("Location ", self.name, " has ", infectious, " infectious individuals.")
        if infectious > 0:
            # How many people will get infected.  Throw a dice for each infectious person.
            # Count how many infectious individuals will actually infect someone.
            samples = np.random.uniform(0, 1, infectious)
            num_potentially_exposed = np.count_nonzero(samples <= self.pars.daily_infectiousness)

            self.expose(num_potentially_exposed)

    # Remove all individuals visiting from population pop
    def leave(self, pop):
        if pop in self.visitors:
            return self.visitors.pop(pop)
        return []

    # Change one individual's sate.
    def set_state(self, index, state):
        current_state = self.state[index]
        self.state_sets[current_state].discard(index)
        self.state_sets[state].add(index)
        self.state[index] = state
        last_state_change = self.state_change_day[index]
        self.state_change_day[index] = self.day
        self.state_by_day_sets.remove((current_state, last_state_change), [index])
        self.state_by_day_sets.add((state, self.day), [index])

        if state == SeirState.E and current_state != SeirState.E:
            self.exposed_probability = self.exposed_probability + self.prob[index]
            self.exposed_susceptibility = self.exposed_susceptibility + self.pop[index]
            self.num_exposed_today = self.num_exposed_today + 1

    def has_spread(self):
        return self.count(SeirState.I) > 0 or self.count(SeirState.E) > 0

    def count(self, state):
        return len(self.state_sets[state])


# Wraps a collection of populations into one object and provide some untility methods over the collection
# of populations.  Used to calculate global stats.
class PopulationsWrapper:
    def __init__(self, pops):
        self.pops = pops
        
    def size(self):
        return sum(pop.size() for pop in self.pops)

    def count(self, state):
        return sum(pop.count(state) for pop in self.pops)

    def exposed_last_day(self):
        return sum(pop.exposed_last_day() for pop in self.pops)

    def mean_susceptibility(self):
        return sum(pop.mean_susceptibility() * pop.count(SeirState.S) for pop in self.pops) / self.count(SeirState.S)
        

# Time series with evolution of stats for a population.
class Series:
    def __init__(self, location_name, population_size):
        self.location_name = location_name
        self.population_size = population_size

        # Time series that model the number of individuals in the standard buckets of the SEIR model.
        self.s = []
        self.e = []
        self.i = []
        self.r = []

        # Time series that models the Effective R
        self.er = []

        # This is what I call the 'implicit R0'.  The value of backing out the number of non-susceptible
        # people from the R calculated above.
        self.ir0 = []

        # This is the standard R, the R calculated by the simple models as r = r0 * num(s) / num(pop)
        # It is not used in the model, we only calculate it to plot the different Rs.
        self.sr = []

        # Captures the evolution of the population mean susceptibility
        self.ms = []

    def update(self, stats):
        self.e.append(stats.ecurr)
        self.s.append(stats.scurr)
        self.i.append(stats.icurr)
        self.r.append(stats.rcurr)
        self.er.append(stats.ercurr)
        self.sr.append(stats.srcurr)
        self.ir0.append(stats.ir0curr)
        self.ms.append(stats.mscurr)

    def normalize(self):
        population_size = self.population_size
        normalized = self.copy()
        normalized.s = [x / population_size for x in self.s]
        normalized.e = [x / population_size for x in self.e]
        normalized.i = [x / population_size for x in self.i]
        normalized.r = [x / population_size for x in self.r]
        return normalized

    def copy(self):
        result = Series(self.location_name, self.population_size)
        result.s = self.s.copy()
        result.e = self.e.copy()
        result.i = self.i.copy()
        result.r = self.r.copy()
        result.er = self.er.copy()
        result.sr = self.sr.copy()
        result.ir0 = self.ir0.copy()
        result.ms = self.ms.copy()
        return result

    def __iadd__(self, other):
        self.population_size += other.population_size
        self.s = add_arrays_wp(self.s, other.s)
        self.e = add_arrays_wp(self.e, other.e)
        self.i = add_arrays_wp(self.i, other.i)
        self.r = add_arrays_wp(self.r, other.r)
        self.er = add_arrays_wp(self.er, other.er)
        self.sr = add_arrays_wp(self.sr, other.sr)
        self.ir0 = add_arrays_wp(self.ir0, other.ir0)
        self.ms = add_arrays_wp(self.ms, other.ms)
        return self

    def __add__(self, other):
        result = Series(self.location_name, self.population_size + other.population_size)
        result += self
        return result

    def __itruediv__(self, factor):
        self.population_size /= factor
        multiply_list(self.s, 1/factor)
        multiply_list(self.e, 1/factor)
        multiply_list(self.i, 1/factor)
        multiply_list(self.r, 1/factor)
        multiply_list(self.er, 1/factor)
        multiply_list(self.sr, 1/factor)
        multiply_list(self.ir0, 1/factor)
        multiply_list(self.ms, 1/factor)
        return self

    @staticmethod
    def add_static(s1, s2):
        if not s1:
            return s2
        if not s2:
            return s1
        result = s1.add(s2)
        result.add(s2)
        return result


# Captures the snapshot state of the model for one population.
class State:
    def __init__(self, pop, par):
        # Computation parameters
        self.r0 = par.r0
        self.days_infectious = par.days_infectious

        # State that changes for every iteration
        self.scurr = pop.size()
        self.ecurr = 0
        self.icurr = 0
        self.rcurr = 0

        # Computed stats.
        self.ercurr = 0
        self.ir0curr = 0
        self.mscurr = 0
        self.srcurr = 0

        # Auxiliary values to calculate the computed stats.
        self.iprev = 0
        self.update_dependent_stats(pop)

    def roll(self, pop):
        self.iprev = self.icurr

        self.scurr = pop.count(SeirState.S)
        self.ecurr = pop.count(SeirState.E)
        self.icurr = pop.count(SeirState.I)
        self.rcurr = pop.count(SeirState.R)

        self.update_dependent_stats(pop)

    def update_dependent_stats(self, pop):
        # Effective R: number of new infections per Infectious individuals, multiplied by
        # the mean number of days spent infectious.
        if self.iprev > 0:
            self.ercurr = pop.exposed_last_day() * self.days_infectious / self.iprev
        else:
            self.ercurr = 0
        self.ir0curr = (self.ercurr * pop.size() / self.scurr) if self.scurr > 0 else 0
        self.mscurr = pop.mean_susceptibility()
        self.srcurr = self.r0 * self.scurr / pop.size()


class Location:
    def __init__(self, global_parameters, population_size, name):
        self.name = name
        self.pars = global_parameters
        self.population_size = population_size

        self.pop = Population(global_parameters.distribution.generate_population(population_size), global_parameters)
        self.series = Series(self.name, self.population_size)
        self.state = State(self.pop, global_parameters)

        # Destinations where our local individuals have traveled to.
        # Type: Location
        self.destinations = []

    # Expose num_exposures individuals to the virus.
    def seed(self, num_seeds):
        if num_seeds > 0:
            self.pop.expose(num_seeds)

    # Update the sate and reset any intraday counters.
    def roll(self):
        self.pop.roll()
        self.state.roll(self.pop)
        self.series.update(self.state)
        self.destinations = []

    def has_spread(self):
        return self.pop.has_spread()

    def spread(self):
        self.pop.spread()

    def travel(self, dest_location, num_trips):
        self.destinations.append(dest_location)
        travelers = self.pop.travel(num_trips)
        dest_location.visit(travelers, self)

    def travel_back(self):
        for dest_location in self.destinations:
            returning = dest_location.leave(self)
            self.pop.return_travelers(returning)

    def visit(self, travelers, source_location):
        self.pop.visit(travelers, source_location.pop)

    def leave(self, other):
        return self.pop.leave(other.pop)

    def normalized_series(self):
        return self.series.normalize()


class Locations:
    def __init__(self, population_params, global_params):
        self.primary_location = Location(global_params, population_params.primary_population_size, "primary")
        self.secondary_locations = []
        self.tertiary_locations = []
        self.params = global_params
        self.daily_primary_secondary_trips = population_params.daily_primary_secondary_trips
        self.daily_secondary_tertiary_trips = population_params.daily_secondary_tertiary_trips

        # Init all locations
        for i in range(population_params.num_secondary_locations):
            location = Location(global_params, population_params.secondary_population_size, "secondary-" + str(i))
            self.secondary_locations.append(location)
            sublocations = []
            for j in range(population_params.num_tertiary_locations):
                sublocation = Location(global_params, population_params.tertiary_population_size,
                                       "tertiary-"+str(i)+"-"+str(j))
                sublocations.append(sublocation)
            self.tertiary_locations.append(sublocations)

        # Wrapper around a collection of populations
        self.pop_wrapper = PopulationsWrapper([location.pop for location in self.all_locations()])
        self.series = Series("global-series", self.pop_wrapper.size())
        # Global stats.
        self.stats = State(self.pop_wrapper, global_params)

    def seed(self, num_seeds):
        self.primary_location.seed(num_seeds)

    def all_locations(self):
        return chain(
            [self.primary_location],
            self.secondary_locations,
            chain.from_iterable(self.tertiary_locations))

    def roll(self):
        for location in self.all_locations():
            location.roll()
        self.stats.roll(self.pop_wrapper)
        self.series.update(self.stats)

    def has_spread(self):
        return any(location.has_spread() for location in self.all_locations())

    def step(self):
        for location in self.secondary_locations:
            location.travel(self.primary_location, self.daily_primary_secondary_trips)
            self.primary_location.travel(location, self.daily_primary_secondary_trips)
        for i, locations in enumerate(self.tertiary_locations):
            for location in locations:
                location.travel(self.secondary_locations[i], self.daily_secondary_tertiary_trips)
                self.secondary_locations[i].travel(location, self.daily_secondary_tertiary_trips)
        for location in self.all_locations():
            location.spread()
        for location in self.all_locations():
            location.travel_back()

    def population_size(self):
        return sum(location.pop.size() for location in self.all_locations())

    def normalized_series(self):
        return self.series.normalize()


class SimulationResult:
    def __init__(self, global_series, series):
        self.global_series = global_series
        self.series = series

    def __add__(self, other):
        return SimulationResult(
            self.global_series + other.global_series,
            [s1 + s2 for s1, s2 in zip(self.series, other.series)])

    def __iadd__(self, other):
        self.global_series += other.global_series
        for i, s in enumerate(self.series):
            self.series[i] += other.series[i]
        return self

    def __itruediv__(self, value):
        self.global_series /= value
        for s in self.series:
            s /= value
        return self


class Simulation:
    @staticmethod
    def run(population_params, global_params, steps):
        result = Simulation.run_step(population_params, global_params)
        log("Ran 1 of ", steps)

        for i in range(1, steps):
            r2 = Simulation.run_step(population_params, global_params)
            log("Ran", i + 1, "of", steps)
            result += r2

        if steps > 1:
            result /= steps

        return result

    @staticmethod
    def run_step(population_params, global_params):
        locations = Locations(population_params, global_params)

        # Choose some individuals to be the first exposed.  If the value is too low,
        # the simulation may die-out at the very beginning.
        num_seeds = 10
        locations.seed(num_seeds)
        locations.roll()

        while locations.has_spread():
            locations.step()
            locations.roll()

            # Output some simulation stats.
            series = locations.series
            last = len(series.s) - 1
            log("t:", len(series.s) - 1, "s:", series.s[last], "e:", series.e[last], "i:",  series.i[last], "r:", series.r[last], "er:", series.er[last], "ms:", series.ms[last])

        # Convert the SEIR series to a number between 0 and 1
        global_series = locations.normalized_series()

        series = [location.normalized_series() for location in locations.all_locations()]

        return SimulationResult(global_series, series)


#############################################################################################
# Chart generation code.
#############################################################################################


class PlotOptions:
    def __init__(self):
        self.ms = True
        self.er = False
        self.sr = False
        self.ir0 = False


def generate_seir_compare(options, series1, series2):
    # output to static HTML file
    output_file("lines-2.html")

    # create a new plot with a title and axis labels
    p = figure(title="simple line example", x_axis_label='Days', y_axis_label='Population')
    p.y_range = Range1d(0, 1.1)

    plot_seir_series(options, series1, p, 'solid', "(Ext)")
    plot_seir_series(options, series2, p, 'dashed', "(Std)")

    p.legend.location = "top_right"
    p.legend.click_policy = "hide"

    # show the results
    show(p)


def generate_seir_charts(options, results):
    output_file("all-series.html")
    all_plots = []
    plot = generate_seir_chart(options, results.global_series)
    all_plots.append([plot])

    for s in results.series:
        plot = generate_seir_chart(options, s)
        all_plots.append(([plot]))

    p = gridplot(all_plots)
    show(p)



def generate_seir_chart(options, series):
    # output to static HTML file
    # output_file(series.location_name + ".html")

    # create a new plot with a title and axis labels
    p = figure(title=series.location_name, x_axis_label='Days', y_axis_label='Population')
    p.y_range = Range1d(0, 1.1)
    plot_seir_series(options, series, p, 'solid', None)

    if options.er or options.sr or options.ir0:
        p.extra_y_ranges = {"r_range": Range1d(start=0, end=np.max(series.er) * 1.1)}
        extra_axis = LinearAxis(y_range_name="r_range")
        extra_axis.axis_label = "R"
        p.add_layout(extra_axis, 'right')

        # Remove initial values for er and ir0
        index = 0
        for index in range(len(series.i)):
            if series.i[index] != 0:
                log("Breaking from i at index=", index)
                log("Previous values of er: ", series.er[:index])
                log("Next values of er: ", series.er[index:index+10])
                break

        index = index + 1
        if options.er:
            p.line(range(index, len(series.er)), series.er[index:], legend_label="Effective r", line_width=2, line_color="grey", y_range_name="r_range")
        if options.sr:
            p.line(range(len(series.sr)), series.sr, legend_label="Standard r", line_width=2, line_color="red", line_dash='dashed', y_range_name="r_range")
        if options.ir0:
            p.line(range(index, len(series.ir0)), series.ir0[index:], legend_label="Implicit r0", line_width=2, line_color="grey", line_dash='dashed', y_range_name="r_range")

    p.legend.location = "top_right"
    p.legend.click_policy = "hide"

    return p


def label(text, suffix):
    if suffix:
        return text + " " + suffix
    else:
        return text


def plot_seir_series(options, series, p, line_dash, suffix):
    # add a line renderer with legend and line thickness
    p.line(range(len(series.s)), series.s, legend_label=label("Susceptible", suffix), line_width=2, line_color="blue", line_dash=line_dash)
    p.line(range(len(series.e)), series.e, legend_label=label("Exposed", suffix), line_width=2, line_color="yellow", line_dash=line_dash)
    p.line(range(len(series.i)), series.i, legend_label=label("Infectious", suffix), line_width=2, line_color="red", line_dash=line_dash)
    p.line(range(len(series.r)), series.r, legend_label=label("Recovered", suffix), line_width=2, line_color="green", line_dash=line_dash)
    if options.ms:
        p.line(range(len(series.ms)), series.ms, legend_label=label("Mean Susceptibility", suffix), line_width=2, line_color="black", line_dash=line_dash)


def relative_population_params(primary_pop_size, num_secondary_locations, num_tertiary_locations, secondary_pop_factor,
                               tertiary_pop_factor, secondary_daily_traveler_ratio,
                               tertiary_daily_traveler_ratio):
    secondary_pop_size = int(primary_pop_size * secondary_pop_factor)
    tertiary_pop_size = int(secondary_pop_size* tertiary_pop_factor)
    num_secondary_travelers = int(secondary_pop_size * secondary_daily_traveler_ratio)
    num_tertiary_travelers = int(tertiary_pop_size * tertiary_daily_traveler_ratio)
    return PopulationParameters(100000, secondary_pop_size, tertiary_pop_size, num_secondary_locations,
                                num_tertiary_locations, num_secondary_travelers, num_tertiary_travelers)


def run():
    options = PlotOptions()
    options.er = True

    # shape = 1 / (self.CV * self.CV)
    # scale = self.mean / shape
    #

    pop_params = relative_population_params(100000, 0, 0, .25, .5, 0.001, 0.004)
    global_params = GlobalParameters(3, 11, 2.5, Gamma(1, math.sqrt(1/3)))

    # global_params = GlobalParameters(3, 11, 2.5, Constant(1))
    start = datetime.now()
    result = Simulation.run(pop_params, global_params, 1)
    generate_seir_charts(options, result)
    end  = datetime.now()
    delta = end - start
    log("Duration: ", delta)


def compare():
    options = PlotOptions()
    options.er = True

    pop_params = relative_population_params(100000, 0, 0, .25, .5, 0.001, 0.004)
    global_params_1 = GlobalParameters(3, 11, 2.5, Gamma(1, 2))
    # global_params_2 = GlobalParameters(3, 11, 2.5, Constant(1))
    global_params_2 = GlobalParameters(3, 11, 2.5, Gamma(1, 1))
    start = datetime.now()
    result_1 = Simulation.run(pop_params, global_params_1, 5)
    result_2 = Simulation.run(pop_params, global_params_2, 5)
    generate_seir_compare(options, result_1.global_series, result_2.global_series)
    end = datetime.now()
    delta = end - start
    log("Duration: ", delta)


if __name__ == "__main__":
    # run()
    compare()