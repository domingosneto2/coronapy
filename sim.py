import numpy as np

from bokeh.plotting import figure, output_file, show
from bokeh.models import LinearAxis, Range1d
from datetime import datetime
import math
import itertools
from itertools import chain
import functools

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


def swap(arr, i1, i2):
    val = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = val


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
    for i in range(len(a), len(b)):
        c[i] = a[len(a) - 1] + b[i]

    return c


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def rotate_left(a, value):
    dropped = a[0]
    for i in range(0, len(a) - 1):
        a[i] = a[i + 1]
    a[-1] = value
    return dropped


def multiply_list(a, value):
    for i in range(len(a)):
        a[i] = a[i] * value

#############################################################################################
# Random population generation and selection functions
#############################################################################################


class Gamma:
    def __init__(self, mean, CV):
        self.mean = mean
        self.CV = CV

    def generate_population(self, pop_size):
        shape = 1 / (self.CV * self.CV)
        scale = self.mean / shape
        # log("shape", shape, "scale", scale)
        return np.random.gamma(shape, scale, pop_size)

    def optimize_draw(self):
        return True

    def select(self, probs, num_samples):
        return np.random.choice(len(probs), num_samples, True, probs)


class Constant:
    def __init__(self, value):
        self.value = value

    def generate_population(self, pop_size):
        return np.full(pop_size, self.value)

    def optimize_draw(self):
        return False

    # Do not pass a probability array since it is uniform, makes the selection faster.
    def select(self, probs, num_samples):
        return np.random.choice(len(probs), num_samples, True)


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

    def optimize_draw(self):
        return True

    def select(self, probs, num_samples):
        return np.random.choice(len(probs), num_samples, True, probs)


# Some testing functions to validate the distributions above.  Not used in real code.
def plot_cdf(distribution, max_value = None):
    bins = 50
    pop = distribution.generate_population(1000000)
    histogram = np.histogram(pop, bins, density=True)
    sum = np.sum(histogram[0])
    cumsum = np.cumsum(histogram[0])
    probs = cumsum * 1 / sum

    y = list()
    y.append(0)
    y.extend(probs)
    x = histogram[1]
    output_file("dist.html")
    p = figure(title="Distribution", x_axis_label='X', y_axis_label='Y')
    p.y_range = Range1d(0, 1.1)
    if max_value is not None:
        p.x_range = Range1d(0, max_value)
    p.line(x, y, line_width=2, line_color="blue")

    show(p)


def plot_pdf(distribution, max_value = None):
    pop = distribution.generate_population(1000000)
    histogram = np.histogram(pop, 100, density=True)
    x = histogram[1]
    y = histogram[0]
    output_file("dist.html")
    p = figure(title="Distribution", x_axis_label='X', y_axis_label='Y')
    if max_value is not None:
        p.x_range = Range1d(0, max_value)
    p.line(x, y, line_width=2, line_color="blue")
    show(p)


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


# Represents the population at one location, for example, one city.
class Population:
    def __init__(self, population):
        # population is an array of relative susceptibilities.
        self.pop = population
        # Convert the relative susceptibility to a probability of selection of each individual.  It has to add up to 1.
        self.prob = self.pop * (1 / np.sum(self.pop))
        self.num_exposed = 0
        self.exposed_probability = 0
        self.total_susceptibility = np.sum(self.pop)
        self.exposed_susceptibility = 0
        self.ms = 0
        self.prev_ms = 0
        self.travelers = set()
        self.visitors_s = dict()
        self.visitors_e = dict()
        self.visitors_i = dict()
        self.current_exposed = []

        self.calc_mean_susceptibility()

    # Exposes count individuals.  Returns the number of newly exposed.
    def expose(self, distribution, count):
        # relative probability of exposing someone from this population or exposing
        # some visitor
        v_probs = self.compute_visitors_probs()
        travelers_probs = self.compute_travelers_probs()

        (visitors_exposed, natives_exposed) = self.compute_groups_exposed(v_probs, travelers_probs, count)

        self.compute_visitors_newly_exposed(distribution, visitors_exposed)

        locals_exposed = 0
        while locals_exposed != natives_exposed:
            log("Exposing local individuals...")
            locals_exposed += self.expose_local(distribution, natives_exposed - locals_exposed)
        log("Done local individuals.")

    def expose_local(self, distribution, natives_exposed):
        # Optimization: if there are a number of already exposed individuals,
        # coalesce all their probabilities in one prob value so we don't have to select
        # over the whole array.  ONLY WORKS WITH SELECTION WITH REPLACEMENT.
        if self.num_exposed > 0 and distribution.optimize_draw():
            prob_value = self.prob[-self.num_exposed]
            self.prob[-self.num_exposed] = self.exposed_probability
            draws = distribution.select(self.prob[:1 - self.num_exposed], natives_exposed)
            self.prob[-self.num_exposed] = prob_value
        else:
            draws = distribution.select(self.prob, natives_exposed)

        draws = [draw for draw in draws if draw not in self.travelers]

        self.current_exposed.extend(draws)
        return len(draws)

    def adjust(self):
        return self.adjust_for_exposure(self.current_exposed)

    def adjust_for_exposure(self, draws):
        # The population array is organized so that the susceptible individuals are
        # at the beginning of the array, and the non-susceptible individuals are at the
        # end of the array.  Just a programming trick to make it easier to compute the
        # exposed individuals.
        draws.sort()
        draws = np.flip(draws)
        previously_exposed = self.num_exposed
        poplen = len(self.pop)
        for draw in draws:
            # if this individual is in the susceptible bucket.
            if draw < len(self.pop) - self.num_exposed:
                # Update accumulated statistics.
                self.exposed_probability = self.exposed_probability + self.prob[draw]
                self.exposed_susceptibility = self.exposed_susceptibility + self.pop[draw]
                # move the individual to the end of array along with the other exposed, infected and
                # recovered individuals.
                swap(self.pop, draw, poplen - 1 - self.num_exposed)
                swap(self.prob, draw, poplen - 1 - self.num_exposed)
                self.num_exposed = self.num_exposed + 1

        self.calc_mean_susceptibility()
        return self.num_exposed - previously_exposed

    def mean_susceptibility(self):
        return self.ms

    def size(self):
        return len(self.pop)

    def calc_mean_susceptibility(self):
        num_susceptible = len(self.pop) - self.num_exposed
        if num_susceptible == 0:
            self.ms = self.prev_ms
        else:
            self.prev_ms = self.ms
            self.ms = (self.total_susceptibility - self.exposed_susceptibility) / num_susceptible

    def travel(self, num_trips):
        travelers = self.select_travelers(num_trips)
        self.travelers.update(travelers)
        return travelers

    def select_travelers(self, num_travelers):
        new_travelers = {}
        while len(new_travelers) < num_travelers:
            extra_travelers = np.random.choice(range(self.pop), num_travelers - len(new_travelers), False)
            extra_travelers = [t for t in extra_travelers if t not in self.travelers and t not in new_travelers]
            new_travelers.update(extra_travelers)
        return new_travelers

    def return_travelers(self, travelers):
        self.travelers.difference_update(travelers)

    def visit_susceptible(self, travel_s, pop):
        self.visitors_s[pop] = travel_s

    def visit_exposed(self, travel_e, pop):
        self.visitors_e[pop] = travel_e

    def visit_infectious(self, travel_i, pop):
        self.visitors_i[pop] = travel_i

    def num_infectious_visitors(self):
        return sum(1 for _ in itertools.chain.from_iterable(self.visitors_i.values()))

    def compute_visitors_probs(self):
        total_prob = 0
        for pop, visitors in itertools.chain(self.visitors_s.items(), self.visitors_e.items(), self.visitors_i.items()):
            total_prob += functools.reduce(lambda v1, v2: v1 + v2, map(lambda v: pop.prob[v], visitors))
        return total_prob

    def compute_travelers_probs(self):
        return functools.reduce(lambda v1, v2: v1 + v2, map(lambda t: self.prob[t], self.travelers), 0)

    def compute_groups_exposed(self, v_probs, travelers_probs, count):
        local_probs = 1 - travelers_probs
        choices = np.random.uniform(0, v_probs + local_probs, count)
        visitors_exposed = np.count_nonzero(choices < v_probs)
        locals_exposed = count - visitors_exposed
        return visitors_exposed, locals_exposed

    def compute_visitors_newly_exposed(self, distribution, visitors_exposed):
        visitors_and_pops = itertools.chain.from_iterable(itertools.chain(
                map(lambda pop, visitors: (pop, visitors, 's'), self.visitors_s.items()),
                map(lambda pop, visitors: (pop, visitors, 'e'), self.visitors_e.items()),
                map(lambda pop, visitors: (pop, visitors, 'i'), self.visitors_i.items())))

        visitors_and_pops = list(visitors_and_pops)

        if len(visitors_and_pops) == 0:
            return

        visitors_probs = [pop.prob[visitor] for pop, visitor in visitors_and_pops]
        visitors_probs = [prob / sum(visitors_probs) for prob in visitors_probs]

        draws = np.random.choice(len(visitors_probs), visitors_exposed, False, visitors_probs)
        newly_exposed = filter(lambda draw: visitors_and_pops[draw][2] == 's', draws)
        for exposed in newly_exposed:
            pop = visitors_and_pops[exposed][1]
            visitor = visitors_and_pops[exposed][0]
            self.visitors_s[pop].remove(visitor)
            if self.visitors_e[pop] :
                self.visitors_e[pop].append(visitor)
            else:
                self.visitors_e[pop] = [visitor]

    def leave_susceptible(self, pop):
        if pop in self.visitors_s:
            return self.visitors_s.pop(pop)
        return []

    def leave_exposed(self, pop):
        if pop in self.visitors_e:
            return self.visitors_e.pop(pop)
        return []

    def leave_infectious(self, pop):
        if pop in self.visitors_i:
            return self.visitors_i.pop(pop)
        return []


class Series:
    def __init__(self):
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

    def normalize(self, population_size):
        self.s = [x / population_size for x in self.s]
        self.e = [x / population_size for x in self.e]
        self.i = [x / population_size for x in self.i]
        self.r = [x / population_size for x in self.r]

    def add(self, other):
        self.s = add_arrays_wp(self.s, other.s)
        self.e = add_arrays_wp(self.e, other.e)
        self.i = add_arrays_wp(self.i, other.i)
        self.r = add_arrays_wp(self.r, other.r)
        self.er = add_arrays_wp(self.er, other.er)
        self.sr = add_arrays_wp(self.sr, other.sr)
        self.ir0 = add_arrays_wp(self.ir0, other.ir0)
        self.ms = add_arrays_wp(self.ms, other.ms)

    def divide(self, factor):
        multiply_list(self.s, 1/factor)
        multiply_list(self.e, 1/factor)
        multiply_list(self.i, 1/factor)
        multiply_list(self.r, 1/factor)
        multiply_list(self.er, 1/factor)
        multiply_list(self.sr, 1/factor)
        multiply_list(self.ir0, 1/factor)
        multiply_list(self.ms, 1/factor)


# Represents the evolution of the virus at one location.
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

        # In this simple model, the number of days spent in E and I buckets is fixed.  So we have
        # as many sub-buckets as days, and individuals move daily through sub-buckets.
        self.e_buckets = list(0 for i in range(par.days_exposed))
        self.i_buckets = list(0 for i in range(par.days_infectious))

        # Computed stats.
        self.ercurr = 0
        self.ir0curr = 0
        self.mscurr = 0
        self.srcurr = 0

        # Auxiliary values to calculate the computed stats.
        self.iprev = 0
        self.last_exposed = 0

        self.update_dependent_stats(pop)

    def roll(self, pop):
        self.iprev = self.icurr

        # The number of people who will move from s to e
        se_move = self.last_exposed

        # Roll from S to E
        self.scurr -= se_move
        self.ecurr += se_move
        ei_move = rotate_left(self.e_buckets, se_move)

        # Roll from E to I
        self.ecurr -= ei_move
        self.icurr += ei_move
        ir_move = rotate_left(self.i_buckets, ei_move)

        # Roll from I to R
        self.icurr -= ir_move
        self.rcurr += ir_move

        self.update_dependent_stats(pop)

        self.last_exposed = 0

    def update_dependent_stats(self, pop):
        # Effective R: number of new infections per Infectious individuals, multiplied by
        # the mean number of days spent infectious.
        if self.iprev > 0:
            self.ercurr = self.last_exposed * self.days_infectious / self.iprev
        else:
            self.ercurr = 0
        self.ir0curr = (self.ercurr * pop.size() / self.scurr) if self.scurr > 0 else 0
        self.mscurr = pop.mean_susceptibility()
        self.srcurr = self.r0 * self.scurr / pop.size()

    def add_exposures(self, num_exposed):
        self.last_exposed = self.last_exposed + num_exposed

    def has_spread(self):
        return self.icurr > 0 or self.ecurr > 0 or self.last_exposed > 0

    def is_s(self, x):
        return x < self.scurr - self.last_exposed

    def is_i(self, x):
        delta = self.scurr + self.ecurr
        return x >= delta and x - delta < self.icurr


class Location:
    def __init__(self, global_parameters, population_size):
        self.pars = global_parameters
        self.population_size = population_size

        self.pop = Population(global_parameters.distribution.generate_population(population_size))
        self.series = Series()
        self.state = State(self.pop, global_parameters)

        self.destinations = []

        self.infectious_away = 0

    def expose(self, num_exposures):
        if num_exposures > 0:
            self.pop.expose(self.pars.distribution, num_exposures)

    def roll(self):
        self.state.roll(self.pop)
        self.series.update(self.state)
        self.destinations = []

    def has_spread(self):
        return self.state.has_spread()

    def num_infectious_at_this_location(self):
        return self.state.icurr - self.infectious_away + self.pop.num_infectious_visitors()

    def spread(self):
        infectious = self.num_infectious_at_this_location()
        if infectious > 0:
            # How many people will get infected.  Throw a dice for each infectious person.
            # Count how many infectious individuals will actually infect someone.
            samples = np.random.uniform(0, 1, infectious)
            num_potentially_exposed = np.count_nonzero(samples <= self.pars.daily_infectiousness)

            # num_potentially_exposed is the number of people who will be exposed to the virus on this round.
            # But some of them may already have been exposed, so we draw from the population
            # and discard the ones that are not in the susceptible bucket.
            self.expose(num_potentially_exposed)

    def adjust(self):
        # TODO: also capture travelers exposed?
        num_exposed = self.pop.adjust()
        self.state.add_exposures(num_exposed)

    def travel(self, dest_location, num_trips):
        self.destinations.append(dest_location)
        travelers = self.pop.travel(num_trips)
        travel_s = filter(lambda x: self.state.is_s(x), travelers)
        travel_i = filter(lambda x: self.state.is_i(x), travelers)
        dest_location.visit_susceptible(travel_s, self)
        dest_location.visit_infectious(travel_i, self)
        self.infectious_away = self.infectious_away + len(travel_i)

    def travel_back(self):
        for dest_location in self.destinations:
            travel_s = dest_location.leave_susceptible(self)
            travel_e = dest_location.leave_exposed(self)
            travel_i = dest_location.leave_infectious(self)
            self.pop.return_travelers(travel_s)
            self.pop.return_travelers(travel_e)
            self.pop.return_travelers(travel_i)
            self.pop.adjust_for_exposure(travel_e)
        self.infectious_away = 0

    def visit_susceptible(self, travel_s, source_location):
        self.pop.visit_susceptible(travel_s, source_location.pop)

    def visit_infectious(self, travel_i, source_location):
        self.pop.visit_infectious(travel_i, source_location.pop)

    def leave_susceptible(self, other):
        self.pop.leave_susceptible(other.pop)

    def leave_exposed(self, other):
        self.pop.leave_exposed(other.pop)

    def leave_infectious(self, other):
        self.pop.leave_infectious(other.pop)


class Locations:
    def __init__(self, global_parameters,
                 primary_population_size, secondary_population_size, tertiary_population_size,
                 num_secondary_locations, num_tertiary_locations, daily_primary_secondary_trips,
                 daily_secondary_tertiary_trips):
        self.primary_location = Location(global_parameters, primary_population_size)
        self.secondary_locations = []
        self.tertiary_locations = []
        self.params = global_parameters
        self.daily_primary_secondary_trips = daily_primary_secondary_trips
        self.daily_secondary_tertiary_trips = daily_secondary_tertiary_trips
        for i in range(num_secondary_locations):
            self.secondary_locations.append(Location(global_parameters, secondary_population_size))
            sublocations = []
            for i in range(num_tertiary_locations):
                sublocations.append(Location(global_parameters, tertiary_population_size))
            self.tertiary_locations.append(sublocations)

    def seed(self, num_seeds):
        self.primary_location.expose(num_seeds)
        self.primary_location.adjust()

    def all_locations(self):
        return chain(
            [self.primary_location],
            self.secondary_locations,
            chain.from_iterable(self.tertiary_locations))

    def roll(self):
        for location in self.all_locations():
            location.roll()

    def has_spread(self):
        return any(location.has_spread() for location in self.all_locations())

    def spread(self):
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
        for location in self.all_locations():
            location.adjust()

    def series(self):
        all_series = map(lambda location: location.series, self.all_locations())
        return functools.reduce(lambda s1, s2: s1.add(s2), all_series)


def run_seir_multiple(population_size, params, steps):
    result = run_seir(population_size, params)
    log("Ran 1 of ", steps)

    for i in range(1, steps):
        r2 = run_seir(population_size, params)
        log("Ran", i + 1, "of", steps)
        result.add(r2)

    if steps > 1:
        result.divide(steps)

    return result


def run_seir(population_size, params):
    # How many people one infectious person will infect per day

    # Create the population.
    locations = Locations(params, population_size, 0, 0, 0, 0, 0, 0)

    # Choose some individuals to be the first exposed.  If the value is too low,
    # the simulation may die-out at the very beginning.
    num_seeds = 10
    locations.seed(num_seeds)
    locations.roll()

    log("locations.has_spread")
    while locations.has_spread():
        # Compute new infections
        log("locations.spread")
        locations.spread()
        log("locations.roll")
        locations.roll()

        # if len(series.s) % 10 == 1:
        #     log("t: ", len(series.s) - 1, "s: ", state.scurr, "e: ", state.ecurr, "i: ",  state.icurr, "r: ", state.rcurr)

    # Convert the SEIR series to a number between 0 and 1
    series = locations.series()
    series.normalize(population_size)

    # Smooth out the er and ir0 numbers
    # ir0 = running_mean(ir0, days_infectious)
    # er = running_mean(er, days_infectious)

    return series


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


def generate_seir_chart(options, series):
    # output to static HTML file
    output_file("lines-2.html")

    # create a new plot with a title and axis labels
    p = figure(title="simple line example", x_axis_label='Days', y_axis_label='Population')
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
            p.line(range(index, len(series.er)), series.er[index:], legend_label="Effective r", line_width=2, line_color="blue", line_dash='dashed', y_range_name="r_range")
        if options.sr:
            p.line(range(len(series.sr)), series.sr, legend_label="Standard r", line_width=2, line_color="red", line_dash='dashed', y_range_name="r_range")
        if options.ir0:
            p.line(range(index, len(series.ir0)), series.ir0[index:], legend_label="Implicit r0", line_width=2, line_color="grey", y_range_name="r_range")
    show(p)


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


options = PlotOptions()

start = datetime.now()
gamma_result = run_seir_multiple(100000, GlobalParameters(3, 11, 2.5, Gamma(1, 2)), 1)
# uniform_result = run_seir_multiple(100000, GlobalParameters(3, 11, 2.5, Constant(1)), 30)
# generate_seir_compare(options, gamma_result, uniform_result)
generate_seir_chart(options, gamma_result)
end  = datetime.now()
delta = end - start
log("Duration: ", delta)

#plot(run_seir_multiple(100000, 3, 11, 2.5, Step(5, 2), 10))
#plot(run_seir_multiple(100000, 3, 11, 2.5, Constant(1), 100))


# plot_cdf(Step(3, 2), 5)
# plot_pdf(Step(5, 2), 10)
