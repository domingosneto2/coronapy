import numpy as np

from bokeh.plotting import figure, output_file, show
from bokeh.models import LinearAxis, Range1d
from datetime import datetime
import math

#############################################################################################
# Utility functions
#############################################################################################


def log(*argv):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
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

        self.calc_mean_susceptibility()

    # Exposes count individuals.  Returns the number of newly exposed.
    def expose(self, distribution, count):
        # Optimization: if there are a number of already exposed individuals,
        # coalesce all their probabilities in one prob value so we don't have to select
        # over the whole array.  ONLY WORKS WITH SELECTION WITH REPLACEMENT.
        if self.num_exposed > 0 and distribution.optimize_draw():
            prob_value = self.prob[-self.num_exposed]
            self.prob[-self.num_exposed] = self.exposed_probability
            draws = distribution.select(self.prob[:1 - self.num_exposed], count)
            self.prob[-self.num_exposed] = self.prob_value
        else:
            draws = distribution.select(self.prob, count)

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


class State:
    def __init__(self, pop, r0, days_exposed, days_infectious):
        # Computation parameters
        self.r0 = r0
        self.days_infectious = days_infectious

        # State that changes for every iteration
        self.scurr = pop.size()
        self.ecurr = 0
        self.icurr = 0
        self.rcurr = 0

        # In this simple model, the number of days spent in E and I buckets is fixed.  So we have
        # as many sub-buckets as days, and individuals move daily through sub-buckets.
        self.e_buckets = list(0 for i in range(days_exposed))
        self.i_buckets = list(0 for i in range(days_infectious))

        # Computed stats.
        self.ercurr = 0
        self.ir0curr = 0
        self.mscurr = 0
        self.srcurr = 0

        # Auxiliary values to calculate the computed stats.
        self.iprev = 0
        self.last_exposed = 0

        self.update_dependent_stats(pop)

    def roll(self, pop, num_newly_exposed):
        self.iprev = self.icurr
        self.last_exposed = num_newly_exposed

        # The number of people who will move from s to e
        se_move = num_newly_exposed

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


def run_seir_multiple(population_size, days_exposed, days_infectious, r0, distribution, steps):
    result = run_seir(population_size, days_exposed, days_infectious, r0, distribution)
    log("Ran 1 of ", steps)

    for i in range(1, steps):
        r2 = run_seir(population_size, days_exposed, days_infectious, r0, distribution)
        log("Ran", i + 1, "of", steps)
        result.add(r2)

    if steps > 1:
        result.divide(steps)

    return result


def run_seir(population_size, days_exposed, days_infectious, r0, distribution):
    # How many people one infectious person will infect per day
    daily_infectiousness = r0 / days_infectious

    # Create the population.
    pop = Population(distribution.generate_population(population_size))

    series = Series()
    state = State(pop, r0, days_exposed, days_infectious)
    series.update(state)

    # Choose some individuals to be the first exposed.  If the value is too low,
    # the simulation may die-out at the very beginning.
    num_seeds = 10
    newly_exposed = pop.expose(distribution, num_seeds)
    state.roll(pop, newly_exposed)
    series.update(state)

    while state.ecurr > 0 or state.icurr > 0:
        # Compute new infections
        num_newly_exposed = 0
        if state.icurr > 0:
            # How many people will get infected.  Throw a dice for each infectious person.
            # Count how many infectious individuals will actually infect someone.
            samples = np.random.uniform(0, 1, state.icurr)
            num_potentially_exposed = np.count_nonzero(samples <= daily_infectiousness)
            num_newly_exposed = 0

            # num_potentially_exposed is the number of people who will be exposed to the virus on this round.
            # But some of them may already have been exposed, so we draw from the population
            # and discard the ones that are not in the susceptible bucket.
            if num_potentially_exposed > 0:
                num_newly_exposed = pop.expose(distribution, num_potentially_exposed)

        state.roll(pop, num_newly_exposed)
        series.update(state)

        # if len(series.s) % 10 == 1:
        #     log("t: ", len(series.s) - 1, "s: ", state.scurr, "e: ", state.ecurr, "i: ",  state.icurr, "r: ", state.rcurr)

    # Convert the SEIR series to a number between 0 and 1
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
gamma_result = run_seir_multiple(100000, 3, 11, 2.5, Gamma(1, 2), 30)
uniform_result = run_seir_multiple(100000, 3, 11, 2.5, Constant(1), 30)
generate_seir_compare(options, gamma_result, uniform_result)
# generate_seir_chart(options, uniform_result)
end  = datetime.now()
delta = end - start
log("Duration: ", delta)

#plot(run_seir_multiple(100000, 3, 11, 2.5, Step(5, 2), 10))
#plot(run_seir_multiple(100000, 3, 11, 2.5, Constant(1), 100))


# plot_cdf(Step(3, 2), 5)
# plot_pdf(Step(5, 2), 10)
