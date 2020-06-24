import numpy as np

from bokeh.plotting import figure, output_file, show
from bokeh.models import LinearAxis, Range1d
from datetime import datetime

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

    def select(self, probs, num_samples):
        return np.random.choice(len(probs), num_samples, False, probs)


class Constant:
    def __init__(self, value):
        self.value = value

    def generate_population(self, pop_size):
        return np.full(pop_size, self.value)

    def select(self, probs, num_samples):
        return np.random.choice(len(probs), num_samples, False)


class Normal:
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def generate_population(self, pop_size):
        return np.random.normal(self.mean, self.stddev, pop_size)

    def select(self, probs, num_samples):
        return np.random.choice(len(probs), num_samples, False, probs)


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


def run_seir_multiple(population_size, days_exposed, days_infectious, r0, distribution, steps):
    result = run_seir(population_size, days_exposed, days_infectious, r0, distribution)
    log("Ran 1 of ", steps)

    for i in range(1, steps):
        r2 = run_seir(population_size, days_exposed, days_infectious, r0, distribution)
        log("Ran", i + 1, "of", steps)
        for j in range(len(r2)):
            result[j] = add_arrays_wp(result[j], r2[j])

    if steps > 1:
        for i in range(len(result)):
            result[i] = [x / steps for x in result[i]]

    return result


def run_seir(population_size, days_exposed, days_infectious, r0, distribution):
    # How many people one infectious person will infect per day
    daily_infectiousness = r0 / days_infectious

    # Each individual in the population has one attribute: its relative susceptibility.
    pop = distribution.generate_population(population_size)
    # This is the probability of selection of each individual, when choosing who is going
    # to be infected.  It has to add up to 1.
    prob = pop * (1 / np.sum(pop))

    # Time series that model the number of individuals in the standard buckets of the SEIR model.
    s = []
    e = []
    i = []
    r = []

    # Time series that models the Effective R of the day.
    er = []

    # This is what I call the 'implicit R0'.  The value of backing out the number of non-susceptible
    # people from the R calculated above.
    ir0 = []

    # This is the standard R, the R calculated by the simple models as r = r0 * num(s) / num(pop)
    # It is not used in the model, we only calculate it to plot the different Rs.
    sr = []

    # Models the evolution of the population mean susceptibility
    ms = []

    # Initialize the 'current' values for the time series.
    scurr = population_size
    ecurr = 0
    icurr = 0
    rcurr = 0
    ercurr = 0
    ir0curr = ercurr * population_size / scurr
    mscurr = np.mean(pop)

    # In this simple model, the number of days spent in E and I buckets is fixed.  So we have
    # as many sub-buckets as days, and individuals move daily through sub-buckets.
    e_buckets = list(0 for i in range(days_exposed))
    i_buckets = list(0 for i in range(days_infectious))

    # for example, if days_exposed is 3, then e_buckets looks like:
    # [ <n exposed 2 days ago>, <n exposed yesterday>, <n exposed today>]
    # And then as we proceed with the simulation, we keep rotating this array left
    # and adding the newly exposed count to the right.  The count at e_buckets[0] gets
    # appended at the end of i_buckets.

    s.append(scurr)
    e.append(ecurr)
    i.append(icurr)
    r.append(rcurr)
    er.append(ercurr)
    sr.append(r0 * scurr / population_size)
    ir0.append(ir0curr)
    ms.append(mscurr)

    # Choose some individuals to be the first exposed.  If the value is too low,
    # the simulation may die-out at the very beginning.
    num_seeds = 10
    choices = distribution.select(prob, num_seeds)

    # The population array is organized so that the susceptible individuals are
    # at the beginning of the array, and the non-susceptible individuals are at the
    # end of the array.  Just a programming trick to make it easier to compute the
    # mean susceptibility of the population.
    choices.sort()
    choices = np.flip(choices)

    # Move newly exposed individuals to the back of the array.
    for index, choice in enumerate(choices):
        swap(pop, choice, scurr - 1 - index)
        swap(prob, choice, scurr - 1 - index)

    scurr = population_size - num_seeds

    # Put the number of newly exposed today at the end of the array.
    e_buckets.append(num_seeds)
    e_buckets = e_buckets[1:]
    ecurr = num_seeds
    mscurr = np.mean(pop[:scurr])

    s.append(scurr)
    e.append(ecurr)
    i.append(icurr)
    r.append(rcurr)
    ercurr = 0
    ir0curr = ercurr * population_size / scurr
    er.append(ercurr)
    sr.append(r0 * scurr / population_size)
    ir0.append(ir0curr)
    ms.append(mscurr)

    while ecurr > 0 or icurr > 0:
        # Compute new infections
        num_newly_exposed = 0
        if icurr > 0:
            # How many people will get infected.  Throw a dice for each infectious person.
            # Count how many infectious individuals will actually infect someone.
            samples = np.random.uniform(0, 1, icurr)
            num_potentially_exposed = np.count_nonzero(samples <= daily_infectiousness)
            num_newly_exposed = 0

            # num_potentially_exposed is the number of people who have been exposed to the virus on this round.
            # But some of them may be are already exposed, infected or recovered, so we draw from the population
            # and discard the ones that are not in teh susceptible bucket.  Our population is organized so that
            # the susceptible individuals are at the beginning of the array.
            if num_potentially_exposed > 0:
                draws = distribution.select(prob, num_potentially_exposed)
                draws.sort()
                draws = np.flip(draws)
                for draw in draws:
                    # if this individual is in the susceptible bucket.
                    if draw < scurr - num_newly_exposed:
                        # move it to the end of array along with the other exposed, infected and cured individuals.
                        swap(pop, draw, scurr - 1 - num_newly_exposed)
                        swap(prob, draw, scurr - 1 - num_newly_exposed)
                        num_newly_exposed = num_newly_exposed + 1

        # Do this calculation before updating the current values
        if icurr > 0:
            ercurr = num_newly_exposed * days_infectious / icurr
        else:
            ercurr = 0

        ir0curr = ercurr * population_size / scurr

        srcurr = r0 * scurr / population_size

        # The number of people who will move between compartments
        se_move = num_newly_exposed
        ir_move = i_buckets[0]
        ei_move = e_buckets[0]

        # Roll from S to E
        scurr -= se_move
        ecurr += se_move
        e_buckets.append(se_move)

        # Roll from E to I
        ecurr -= ei_move
        icurr += ei_move
        e_buckets = e_buckets[1:]
        i_buckets.append(ei_move)

        # Roll from I to R
        icurr -= ir_move
        rcurr += ir_move
        i_buckets = i_buckets[1:]

        mscurr = np.mean(pop[:scurr])

        # Update series
        s.append(scurr)
        e.append(ecurr)
        i.append(icurr)
        r.append(rcurr)

        er.append(ercurr)
        sr.append(srcurr)
        ir0.append(ir0curr)
        ms.append(mscurr)

    # Convert the SEIR series to a number between 0 and 1
    s = [x / population_size for x in s]
    e = [x / population_size for x in e]
    i = [x / population_size for x in i]
    r = [x / population_size for x in r]

    # Smooth out the er and ir0 numbers
    # ir0 = running_mean(ir0, days_infectious)
    # er = running_mean(er, days_infectious)

    return [s, e, i, r, er, sr, ir0, ms]


#############################################################################################
# Chart generation code.
#############################################################################################


def generate_seir_chart_from_series(x):
    generate_seir_chart(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])


def generate_seir_chart(s, e, i, r, er, sr, ir0, ms):
    # output to static HTML file
    output_file("lines.html")

    # create a new plot with a title and axis labels
    p = figure(title="simple line example", x_axis_label='Days', y_axis_label='Population')
    p.y_range = Range1d(0, 1.1)

    # add a line renderer with legend and line thickness
    p.line(range(len(s)), s, legend_label="Susceptible", line_width=2, line_color="blue")
    p.line(range(len(e)), e, legend_label="Exposed", line_width=2, line_color="yellow")
    p.line(range(len(i)), i, legend_label="Infectious", line_width=2, line_color="red")
    p.line(range(len(r)), r, legend_label="Recovered", line_width=2, line_color="green")
    p.line(range(len(ms)), ms, legend_label="Mean Susceptibility", line_width=2, line_color="blue", line_dash="dashed")

    p.extra_y_ranges = {"r_range": Range1d(start=0, end=np.max(er) * 1.1)}
    extra_axis = LinearAxis(y_range_name="r_range")
    extra_axis.axis_label = "R"
    p.add_layout(extra_axis, 'right')

    # Remove initial values for er and ir0
    for index in range(len(i)):
        if i[index] != 0:
            log("Breaking from i at index=", index)
            log("Previous values of er: ", er[:index])
            log("Next values of er: ", er[index:index+10])
            break

    index= index + 1

    p.line(range(index, len(er)), er[index:], legend_label="Effective r", line_width=2, line_color="black", y_range_name="r_range")
    # p.line(range(len(sr)), sr, legend_label="Standard r", line_width=2, line_color="black", line_dash='dashed', y_range_name="r_range")
    # p.line(range(index, len(ir0)), ir0[index:], legend_label="Implicit r0", line_width=2, line_color="grey", y_range_name="r_range")

    # show the results
    show(p)


def plot(result):
    generate_seir_chart_from_series(result)


plot(run_seir_multiple(100000, 3, 11, 2.5, Gamma(1, 3), 10))
# plot(run_seir_multiple(1000000, 3, 11, 2.5, Constant(1), 1))


# plot_cdf(Gamma(1, 2), 20)
# plot_pdf(Gamma(1, 3), 10)
