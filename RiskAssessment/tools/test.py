#!/bin/env python3

import random
from typing import Sequence, Tuple, Callable, TypeVar
from math import sin, cos, sqrt, exp, pi, e
from deap import base, creator, tools, algorithms
from itertools import repeat


def spherical(xi: Sequence) -> Tuple:
    f = 0.0
    for x in xi:
        f += x*x
    return f,


def ackley(xi: Sequence) -> Tuple:
    c1 = 0.
    c2 = 0.
    n = len(xi)
    for x in xi:
        c1 += x*x
        c2 += cos(2*pi*x)
    return 20 + e - 20*exp(-0.2*sqrt(c1/n)) - exp(c2/n),


def rastrigin(xi: Sequence) -> Tuple:
    f = 0.
    for x in xi:
        f += x*x - 10*cos(2*pi*x)
    return 10*len(xi) + f,


def griewangk(xi: Sequence) -> Tuple:
    c1 = 0.
    c2 = 1.
    for i, x in enumerate(xi, start=1):
        c1 += x*x
        c2 *= cos(x/sqrt(i))
    n = len(xi)
    return 1 + 0.0025*c1/n - c2,


def schwefei(xi: Sequence) -> Tuple:
    f = 0.
    for x in xi:
        f += -x*sin(sqrt(abs(x)))
    return f,


def rosenbroch(xi: Sequence) -> Tuple:
    f = 0.
    for i in range(len(xi) - 1):
        f += 100*(xi[i+1] - xi[i]*xi[i])**2 + (xi[i] - 1)**2
    return f,

LUType = TypeVar('LUType', float, Sequence)
T = TypeVar("T")


def mutUniform(individual: T, low: LUType, up: LUType, indpb: float=0.5):
    """This function applies a uniform mutation on the input individual.
    This mutation expects a :term:`sequence` individual composed of real valued attributes.
    The *indpb* argument is the probability of each attribute to be mutated.

    :param individual: Individual to be mutated.
    :param low: A value or a :term:`python:sequence` of values that
                is the lower bound of the search space.
    :param up: A value or a :term:`python:sequence` of values that
                is the upper bound of the search space.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.

    This function uses the :func:`~random.random` and :func:`~random.gauss`
    functions from the python base :mod:`random` module.
    """
    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low limit must be at least the size of individual: %d < %d" % (len(low), size))

    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up limit must be at least the size of individual: %d < %d" % (len(up), size))

    for i, l, u in zip(range(size), low, up):
        if random.random() < indpb:
            individual[i] = random.uniform(l, u)

    return individual


# Simulated Annealing
def sa(func: Callable[[Sequence], Tuple], ndim: int, llimit: LUType, rlimit: LUType) -> None:
    if len(llimit) != ndim or len(rlimit) != ndim:
        raise IndexError("Bound limit must be the same size as of individual.")

    # Simulated Annealing parameter
    t = 1.e+8  # initial temperature
    factor = 0.98

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()

    # A list of function objects to be called in order to initialize the individual.
    attribute_n = [lambda: random.uniform(l, r) for l, r in zip(llimit, rlimit)]
    toolbox.register("individual", tools.initCycle, creator.Individual, attribute_n)

    ind = toolbox.individual()
    ind.fitness.values = func(ind)

    hof = tools.HallOfFame(1)

    while t > 1.:
        for i in range(100):
            indi = toolbox.individual()
            indi.fitness.values = func(indi)
            delta = indi.fitness.values[0] - ind.fitness.values[0]
            if delta < 0:
                hof.update([indi])
                ind = indi
                continue
            if random.random() < exp(-delta/t):
                ind = indi
        t *= factor
        ind = hof[0]
    print(hof[0].fitness.values)
    return hof[0].fitness.values


# Differential Evolution
def de(func: Callable[[Sequence], Tuple], ndim: int, llimit: LUType, rlimit: LUType) -> None:
    if len(llimit) != ndim or len(rlimit) != ndim:
        raise IndexError("Bound limit must be the same size as of individual.")

    # DE parameters
    ngen = 2000  # generation limit
    cr = 0.9  # crossover ratio
    f = 0.6  # scale factor
    mu = 50  # population size

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()

    # A list of function objects to be called in order to fill the individual.
    attribute_n = [lambda: random.uniform(l, r) for l, r in zip(llimit, rlimit)]
    toolbox.register("individual", tools.initCycle, creator.Individual, attribute_n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # benchmark function
    toolbox.register("evaluate", func)

    pop = toolbox.population(n=mu)

    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof = tools.HallOfFame(1)

    for g in range(ngen):
        for k, agent in enumerate(pop):
            # Pick three agents a, b and c from the population at random, they must be distinct from each other
            # as well as from agent x
            a = random.randrange(mu)
            while a == k:
                a = random.randrange(mu)
            b = random.randrange(mu)
            while b == k or b == a:
                b = random.randrange(mu)
            c = random.randrange(mu)
            while c == k or c ==b or c == a:
                c = random.randrange(mu)

            xa = pop[a] # rand
            xb = pop[b]
            xc = pop[c]
            # Pick a random index R between [0, ndim)
            r = random.randrange(ndim)

            # Compute the agent's potentially new position y as follows
            y = toolbox.clone(agent)
            for i, value in enumerate(agent):
                if i == r or random.random() < cr:  # bin
                # if random.random() < cr:  # exp
                    y[i] = xa[i] + f * (xb[i] - xc[i])
                    #
                    if y[i] > rlimit[i]:
                        y[i] = (xa[i] + rlimit[i])/2.
                    if y[i] < llimit[i]:
                        y[i] = (xa[i] + llimit[i])/2.
                #
            y.fitness.values = toolbox.evaluate(y)
            if y.fitness > agent.fitness:
                pop[k] = y
            # end of agent
        # end of generation
    # end of evolution

    hof.update(pop)
    print(hof[0].fitness)

    return hof[0].fitness.values


# Genetic Algorithm
def ga(func: Callable[[Sequence], Tuple], ndim: int, llimit: LUType, rlimit: LUType) -> None:
    if len(llimit) != ndim or len(rlimit) != ndim:
        raise IndexError("Bound limit must be the same size as of individual.")

    # GA parameters
    ngen = 2000  # generation limit
    cxpb = 0.45  # crossover probability
    mutpb = 0.45 # mutation probability
    mu = 50  # population size
    lamda = 100

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    # A list of function objects to be called in order to fill the individual.
    attribute_n = [lambda: random.uniform(l, r) for l, r in zip(llimit, rlimit)]
    toolbox.register("individual", tools.initCycle, creator.Individual, attribute_n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxUniform, indpb=1.)
    toolbox.register("mutate", tools.mutUniformInt, low=llimit, up=rlimit, indpb=0.5)
    toolbox.register("select", tools.selBest)

    # benchmark function
    toolbox.register("evaluate", func)

    # population initialization
    pop = toolbox.population(n=mu)
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof = tools.HallOfFame(1)

    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=mu, lambda_=lamda, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                             stats=None, halloffame=hof, verbose=False)
    # Begin the generational process
    # for gen in range(ngen):
    #     chosen = toolbox.select(pop)
    #     offspring = []
    #     for _ in range(lamda):
    #         op_choice = random.random()
    #         if op_choice < cxpb:  # Apply crossover
    #             ind1, ind2 = map(toolbox.clone, random.sample(chosen, 2))
    #             ind1, ind2 = toolbox.mate(ind1, ind2)
    #             del ind1.fitness.values
    #             del ind2.fitness.values
    #             offspring.append(ind1)
    #             offspring.append(ind2)
    #         elif op_choice < cxpb + mutpb:  # Apply mutation
    #             ind = toolbox.clone(random.choice(chosen))
    #             ind, = toolbox.mutate(ind)
    #             del ind.fitness.values
    #             offspring.append(ind)
    #         else:  # Apply reproduction
    #             offspring.append(random.choice(chosen))
    #     #
    #     # update fitness of spring
    #     fitnesses = toolbox.map(toolbox.evaluate, offspring)
    #     for ind, fit in zip(offspring, fitnesses):
    #         ind.fitness.values = fit
    #
    #     # Select the next generation population
    #     pop[:] = tools.selBest(pop + offspring, mu)
    #     hof.update(pop)
    # # end of evolution
    #
    print(hof[0].fitness.values)

    return hof[0].fitness.values

if __name__ == "__main__":
    dims = 20
    left = [-512.]*dims
    right = [512.]*dims

    f = 0.
    for i in range(50):
        print(i,": ", end="")
        f += ga(rosenbroch, dims, left, right)[0]
    print(f/50)

    exit()
