#!/bin/env python3

import os
import csv
import random
import configparser
from math import sqrt, sin, cos, radians, degrees, acos, e
from typing import Sequence, Tuple, Callable, TypeVar
from deap import base, creator, tools

if 'tools' not in os.listdir():
    os.chdir("..")
os.chdir('tools')

_PNP_ = 0.995


def mean_std(elements: Sequence) -> float:
    n = len(elements)
    m = 0.
    for element in elements:
        m += element
    m /= n

    s = 0.
    for element in elements:
        s += (element - m)**2
    s = sqrt(s/n)
    return m, s


class OptiConfig:
    # output
    outfile = None

    # penalty factor
    fpenalty = 1000.

    # DE parameters
    ngen = 2000  # generation limit
    cr = 0.9  # crossover ratio
    f = 0.6  # scale factor
    mu = 50  # population size
    low = None
    up = None


class Shield:
    SHIELDS = {'swi', 'swe', 'wsi', 'wse', 'swsi', 'swse'}

    # coefficients
    swia = []
    swea = []
    wsia = []
    wsea = []
    swsia = []
    swsea = []

    # parameters
    swp = []
    wsp = []
    swsp = []

    #
    nrho = []
    narea = []
    nsurf = []
    nvec = []
    ntype = []
    nsamples = []
    nvmax = []

    @staticmethod
    def loadcsv():
        wdir = os.getcwd()
        hdir, _ = os.path.split(wdir)

        # Single wall internal
        fcsv = open(hdir + '/input/swi.csv', mode='r', newline='', encoding='gb2312')
        reader = csv.reader(fcsv, dialect='excel')
        for i, line in enumerate(reader):
            if i < 3:
                continue
            elif i == 3:
                Shield.swia.append(line[1:5])
            elif line[0] == '':
                break
            else:
                Shield.swia.append(list(map(float, line[1:5])))
        fcsv.close()

        # Single wall external
        fcsv = open(hdir + '/input/swe.csv', mode='r', newline='', encoding='gb2312')
        reader = csv.reader(fcsv, dialect='excel')
        for i, line in enumerate(reader):
            if i < 3:
                continue
            elif i == 3:
                Shield.swea.append(line[1:5])
            elif line[0] == '':
                break
            else:
                Shield.swea.append(list(map(float, line[1:5])))
        fcsv.close()

        # Whipple shield internal
        fcsv = open(hdir + '/input/wsi.csv', mode='r', newline='', encoding='gb2312')
        reader = csv.reader(fcsv, dialect='excel')
        for i, line in enumerate(reader):
            if i < 3:
                continue
            elif i == 3:
                Shield.wsia.append(line[1:11])
            elif line[0] == '':
                break
            else:
                Shield.wsia.append(list(map(float, line[1:11])))
        fcsv.close()

        # Whipple shield external
        fcsv = open(hdir + '/input/wse.csv', mode='r', newline='', encoding='gb2312')
        reader = csv.reader(fcsv, dialect='excel')
        for i, line in enumerate(reader):
            if i < 3:
                continue
            elif i == 3:
                Shield.wsea.append(line[1:11])
            elif line[0] == '':
                break
            else:
                Shield.wsea.append(list(map(float, line[1:11])))
        fcsv.close()

        # Stuffed Whipple Shield internal
        fcsv = open(hdir + '/input/swsi.csv', mode='r', newline='', encoding='gb2312')
        reader = csv.reader(fcsv, dialect='excel')
        for i, line in enumerate(reader):
            if i < 3:
                continue
            elif i == 3:
                Shield.swsia.append(line[1:12])
            elif line[0] == '':
                break
            else:
                Shield.swsia.append(list(map(float, line[1:12])))
        fcsv.close()

        # Stuffed Whipple Shield external
        fcsv = open(hdir + '/input/swse.csv', mode='r', newline='', encoding='gb2312')
        reader = csv.reader(fcsv, dialect='excel')
        for i, line in enumerate(reader):
            if i < 3:
                continue
            elif i == 3:
                Shield.swsea.append(line[1:14])
            elif line[0] == '':
                break
            else:
                Shield.swsea.append(list(map(float, line[1:14])))
        fcsv.close()

        # Single wall parameters
        fcsv = open(hdir + '/input/sw.csv', mode='r', newline='', encoding='gb2312')
        reader = csv.reader(fcsv, dialect='excel')
        for i, line in enumerate(reader):
            if i == 0:
                Shield.swp.append(['thickness', 'rhop', 'rhot', 'yield', 'HB', 'C'])
            elif line[0] == '':
                break
            else:
                Shield.swp.append(list(map(float, line[3:9])))
        fcsv.close()

        # Whipple shield parameters
        fcsv = open(hdir + '/input/ws.csv', mode='r', newline='', encoding='gb2312')
        reader = csv.reader(fcsv, dialect='excel')
        for i, line in enumerate(reader):
            if i == 0:
                Shield.wsp.append(['tb', 'tw', 'rhop', 'rhob', 'rhow', 'yeildb', 'yieldw', 'theta', 'S'])
            elif line[0] == '':
                break
            else:
                Shield.wsp.append(list(map(float, line[3:12])))
        fcsv.close()

        # Stuffed Whipple shield parameters
        fcsv = open(hdir + '/input/sws.csv', mode='r', newline='', encoding='gb2312')
        reader = csv.reader(fcsv, dialect='excel')
        for i, line in enumerate(reader):
            if i == 0:
                Shield.swsp.append(['tb', 'tw', 'rhop', 'rhob', 'rhow', 'mcb', 'mcw',
                                    'yieldb', 'yieldw', 'sigmcb', 'sigmcw', 'theta', 'S'])
            elif line[0] == '':
                break
            else:
                Shield.swsp.append(list(map(float, line[3:16])))
        fcsv.close()

        return

    # Single Wall Internal
    # function [ dc ] = single_internal(a1,a2,a3,a4,sigma,rout,roup,C1,t,v)
    @staticmethod
    def swi(a, p, v, theta):
        v *= cos(radians(theta))

        a1, a2, a3, a4 = a

        # p = ['thickness', 'rhop', 'rhot', 'yield', 'HB', 'C']
        t, roup, rout, sigma, _, c1 = p

        dc = a1 * t * (sigma/(rout * c1**2))**a2 * (roup/rout)**a3 * (v/c1)**a4
        return dc

    # Single Wall External
    # def swe(a1, a2, a3, a4, bh, rout, roup, c1, t, v):
    @staticmethod
    def swe(a, p, v, theta):
        v *= cos(radians(theta))

        a1, a2, a3, a4 = a

        # p = ['thickness', 'rhop', 'rhot', 'yield', 'HB', 'C']
        t, roup, rout, _, bh, c1 = p

        dc = ((t / 1.8) * (bh**a1 * (rout / roup)**a2) / (a3 * (v / c1)**a4))**0.9473684210526315
        return dc

    # Whipple Shield Internal
    # def wsi(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, tb, tw, roup, roub, rouw, sigmb, sigmw, theta, s, v):
    @staticmethod
    def wsi(a, p, v, theta):
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = a

        # p = ['tb', 'tw', 'rhop', 'rhob', 'rhow', 'yeildb', 'yieldw', 'theta', 'S']
        tb, tw, roup, roub, rouw, sigmb, sigmw, _, s = p

        theta = radians(theta)
        vls = 3. * cos(theta) ** -0.5
        vhs = 7. * cos(theta) ** -0.3333333333333333

        if v < vls:
            dc = a1 * (a2 * tb * (roub/rouw) * (sigmb/sigmw)**a3 + tw) * (rouw/roup)**a4 * (roup * v**2/sigmw)**a5
        elif v > vhs:
            dc = a6 * tb**a7 * tw**a8 * s**(1. - a8 - a7) * (rouw / roup)**a9 * (roup * v**2 / sigmw)**a10
        else:
            dc_1 = a1 * (a2 * tb * (roub/rouw) * (sigmb/sigmw)**a3 + tw) * (rouw/roup)**a4 * (roup * vls**2/sigmw)**a5
            dc_2 = a6 * tb**a7 * tw**a8 * s**(1. - a8 - a7) * (rouw / roup)**a9 * (roup * vhs**2 / sigmw)**a10
            dc = dc_1 * (7. - v)/4. + dc_2 * (v - 3.)/4.
        return dc

    # Whipple Shield External
    # def wse(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, tb, tw, roup, roub, sigmw, theta, s, v):
    @staticmethod
    def wse(a, p, v, theta):
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = a

        # p = ['tb', 'tw', 'rhop', 'rhob', 'rhow', 'yeildb', 'yieldw', 'theta', 'S']
        tb, tw, roup, roub, _, _, sigmw, _, s = p

        theta = radians(theta)
        ctheta = cos(theta)
        if v < 3.:
            dc = (tw * (sigmw / 40.)**a1 + tb) / (a2 * roup**a3 * v**a4 * ctheta**a5)**0.9473684210526315
        elif v > 7.:
            dc = (a6 * tw**0.6666666666666666 * s**0.3333333333333333 * (sigmw/70.)**a7) / (roup**a8 * roub**a9 * v**a5 * ctheta**a10)
        else:
            dc_1 = (tw * (sigmw / 40.)**a1 + tb) / (a2 * roup**a3 * 3.**a4 * ctheta**a5)**0.9473684210526315
            dc_2 = (a6 * tw**0.6666666666666666 * s**0.3333333333333333 * (sigmw / 70.)**a7) / (roup**a8 * roub**a9 * 7.**a5 * ctheta**a10)
            dc = dc_1 * (7. - v)/4 + dc_2 * (v - 3.)/4
        return dc

    # Stuffed Whipple Shield internal
    # def swsi(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11,
    #          tb, tw, roup, rouw, mroucb, mroucw, sigmb, sigmw, sigmcb, sigmcw, theta, s, v):
    @staticmethod
    def swsi(a, p, v, theta):
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = a

        # p = ['tb', 'tw', 'rhop', 'rhob', 'rhow', 'mcb', 'mcw', 'yieldb', 'yieldw', 'sigmcb', 'sigmcw', 'theta', 'S']
        tb, tw, roup, _, rouw, mroucb, mroucw, sigmb, sigmw, sigmcb, sigmcw, _, s = p

        theta = radians(theta)
        ctheta = cos(theta)
        vls = 2.6 * cos(theta)**-0.5
        vhs = 6.5 * cos(theta)**-0.3333333333333333
        if v < vls:
            dc = a1*(tw + tb * (sigmb/sigmw)**a2 + mroucb/rouw * (sigmcb/sigmw)**a2 + mroucw/rouw * (sigmcw/sigmw)**a2)\
                 * (rouw/roup)**a3 * (roup * (v * ctheta)**2/sigmw)**a4
        elif v > vhs:
            dc = a5 * (tb**a6 * tw**a7 * s**(1. - a6 - a7) + a8 * (mroucb/rouw)**a9 * (mroucw/rouw)**(1. - a9))\
                 * (rouw/roup)**a10 * (roup * (v * ctheta)**2/sigmw)**a11
        else:
            dc_1 = a1*(tw+tb * (sigmb/sigmw)**a2 + mroucb/rouw * (sigmcb/sigmw)**a2 + mroucw/rouw * (sigmcw/sigmw)**a2)\
                   * (rouw/roup)**a3 * (roup * (vls * ctheta)**2/sigmw)**a4
            dc_2 = a5 * (tb**a6 * tw**a7 * s**(1. - a6 - a7) + a8 * (mroucb/rouw)**a9 * (mroucw/rouw)**(1 - a9))\
                * (rouw/roup)**a10 * (roup * (vhs * ctheta)**2/sigmw)**a11
            dc = dc_1 * (6.5 - v)/3.9 + dc_2 * (v - 2.6)/3.9
        return dc

    # Stuffed Whipple Shield external
    #     def swse(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13,
    #              tb, tw, roup, roub, rouw, mroucb, mroucw, sigmw, theta, s, v):
    @staticmethod
    def swse(a, p, v, theta):
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 = a

        # p = ['tb', 'tw', 'rhop', 'rhob', 'rhow', 'mcb', 'mcw', 'yieldb', 'yieldw', 'sigmcb', 'sigmcw', 'theta', 'S']
        tb, tw, roup, roub, rouw, mroucb, mroucw, _, sigmw, _, _, _, s = p

        theta = radians(theta)
        ctheta = cos(theta)
        vls = 2.6 / ctheta**0.5
        vhs = 6.5 / ctheta**0.75
        if v < vls:
            dc = a1 * (tw * (sigmw/40.)**a2 + a3 * (roub * tb + mroucb + mroucw))/(roup**a5 * v**a6 * ctheta**a4)
        elif v >= vhs:
            dc = a7 * (tw * rouw)**a8 * s ** a9 * (sigmw/40.)**a10/(roup**a11 * v ** a12 * ctheta**a13)
        else:
            dc_1 = a1*(tw * (sigmw/40.)**a2 + a3 * (roub * tb + mroucb + mroucw))/(roup**a5 * vls**a6 * ctheta**a4)
            dc_2 = a7 * (tw * rouw)**a8 * s**a9 * (sigmw/40.)**a10/(roup**a11 * vhs**a12 * ctheta**a13)
            dc = dc_1 * (vhs - v)/(vhs - vls) + dc_2 * (v - vls)/(vhs - vls)
        return dc


# Area of polygon
def apolygon(nodes):
    xy = []
    # Points should be labeled sequentially in the counterclockwise direction for a convex polygon
    n1 = nodes[0]
    n2 = nodes[1]
    x = [n2[0] - n1[0], n2[1] - n1[1], n2[2] - n1[2]]
    xn = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])
    x = [x[0]/xn, x[1]/xn, x[2]/xn]
    y = npolygon(nodes)
    y = [y[1]*x[2] - y[2]*x[1], y[2]*x[0] - y[0]*x[2], y[0]*x[1] - y[1]*x[0]]

    for i in range(len(nodes)):
        if i == 0:
            xy.append([0, 0])
        else:
            le = nodes[i]
            lh = nodes[i-1]
            l = [le[0] - lh[0], le[1] - lh[1], le[2] - lh[2]]
            lx = l[0]*x[0] + l[1]*x[1] + l[2]*x[2]
            ly = l[0]*y[0] + l[1]*y[1] + l[2]*y[2]
            lx = xy[-1][0] + lx
            ly = xy[-1][1] + ly
            xy.append([lx, ly])

    area = 0
    for i in range(len(xy)):
        if i == 0:
            area += xy[-1][0]*xy[0][1] - xy[-1][1]*xy[0][0]
        else:
            area += xy[i-1][0]*xy[i][1] - xy[i-1][1]*xy[i][0]
    return area/2.


# normal of polygon
def npolygon(nodes):
    # Points should be labeled sequentially in the counterclockwise direction for a convex polygon
    n1 = nodes[0]
    n2 = nodes[1]
    n3 = nodes[2]
    x1, y1, z1 = n2[0] - n1[0], n2[1] - n1[1], n2[2] - n1[2]
    x2, y2, z2 = n3[0] - n2[0], n3[1] - n2[1], n3[2] - n2[2]
    nv = [y1*z2 - z1*y2, z1*x2 - x1*z2, x1*y2 - y1*x2]
    n = sqrt(nv[0]*nv[0] + nv[1]*nv[1] + nv[2]*nv[2])
    return [nv[0]/n, nv[1]/n, nv[2]/n]


def loadsample(infile):
    sample = []
    for line in infile:
        d, v, az, el, p = map(float, line.split())
        sample.append([d, v, az, el, p])
    # end of infile
    return sample


def load3dmodel(infile):
    surf = []
    node = []
    for line in infile:
        if line[0] == '#':
            continue
        elif line[0] == 'N' or line[0] == 'n':
            _, x, y, z = line.split()
            x, y, z = map(float, (x, y, z))
            node.append([x, y, z])
        elif line[0] == 'S' or line[0] == 's':
            sn = []
            ss = line[1:].split()
            stype2 = int(ss.pop())
            stype1 = int(ss.pop())
            s = ss.pop()
            for i in ss:
                nidx = int(i) - 1
                sn.append(node[nidx])
            surf.append([sn, s, stype1, stype2])
            # end surface
        # end comment + node + surface
        else:
            raise Exception(line)
    # end infile
    return surf


# Hyper Velocity impact test
def hvitest(nvec, shieldtype, n1, n2, samples, vmax):
    if shieldtype == 'swi':
        a = Shield.swia[n1]
        p = Shield.swp[n2]
        ble = Shield.swi
    elif shieldtype == 'swe':
        a = Shield.swea[n1]
        p = Shield.swp[n2]
        ble = Shield.swe
    elif shieldtype == 'wsi':
        a = Shield.wsia[n1]
        p = Shield.wsp[n2]
        ble = Shield.wsi
    elif shieldtype == 'wse':
        a = Shield.wsea[n1]
        p = Shield.wsp[n2]
        ble = Shield.wse
    elif shieldtype == 'swsi':
        a = Shield.swsia[n1]
        p = Shield.swsp[n2]
        ble = Shield.swsi
    elif shieldtype == 'swse':
        a = Shield.swsea[n1]
        p = Shield.swsp[n2]
        ble = Shield.swse
    else:
        a = None
        p = None
        ble = None

    # critical condition with maxium velocity and direct impact
    dcritical = ble(a, p, vmax, 0)

    sump = 0
    clim = cos(radians(85.))  # maximum theta
    for sample in samples:
        dia, vmag, vaz, vel, pop = sample
        if dia < dcritical:
            continue

        vaz = radians(vaz)
        vel = radians(vel)
        v = [vmag*cos(vel)*cos(vaz), vmag*cos(vel)*sin(vaz), vmag*sin(vel)]
        ctheta = -1 * (v[0]*nvec[0] + v[1]*nvec[1] + v[2]*nvec[2])/vmag

        if ctheta < clim:
            continue

        theta = degrees(acos(ctheta))
        if ble(a, p, vmag, theta) < dia:
            sump += pop*ctheta
    # end of samples
    return sump


def pnp():
    total = 0.
    for i, every in enumerate(Shield.ntype):
        samples = Shield.nsamples[i]

        if len(samples) == 0:
            continue
        vmax = Shield.nvmax[i]
        fp_m2_year = hvitest(Shield.nvec[i], *every, samples, vmax)
        np = fp_m2_year * Shield.nsurf[i]  # for time span of one year
        total += np
    # end of shield
    return e**-total


def fmass(t: Sequence):
    # update to Shield
    i = 0
    for m in Shield.ntype:
        if m[0] == "swi" or m[0] == "swe":
            # swp ['thickness', 'rhop', 'rhot', 'yield', 'HB', 'C']
            Shield.swp[m[2]][0] = t[i]
            i += 1
        elif m[0] == "wsi" or m[0] == "wse":
            # wsp ['tb', 'tw', 'rhop', 'rhob', 'rhow', 'yeildb', 'yieldw', 'theta', 'S']
            Shield.wsp[m[2]][0] = t[i]
            i += 1
            Shield.wsp[m[2]][1] = t[i]
            i += 1
        elif m[0] == "swsi" or m[0] == "swse":
            # swsp ['tb', 'tw', 'rhop', 'rhob', 'rhow', 'mcb', 'mcw',...]
            Shield.swsp[m[2]][0] = t[i]
            i += 1
            Shield.swsp[m[2]][1] = t[i]
            i += 1
        else:
            raise Exception(m)
    # end of update

    mass = 0.
    for i, ti in enumerate(t):
        mass += ti*Shield.nrho[i]*Shield.narea[i]
    # end of thickness list

    p = pnp()
    if p < _PNP_:
        penalty = 1. - p/_PNP_
        mass *= e**(OptiConfig.fpenalty*penalty)
    return mass,

LUType = TypeVar('LUType', float, Sequence)


# Differential Evolution
def de(func: Callable[[Sequence], Tuple], ndim: int, llimit: LUType, rlimit: LUType) -> None:
    if len(llimit) != ndim or len(rlimit) != ndim:
        raise IndexError("Bound limit must be the same size as of individual.")

    # DE parameters
    ngen = OptiConfig.ngen  # generation limit
    cr = OptiConfig.cr  # crossover ratio
    f = OptiConfig.f  # scale factor
    mu = OptiConfig.mu  # population size

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
            while c == k or c == b or c == a:
                c = random.randrange(mu)

            xa = pop[a]  # rand
            xb = pop[b]
            xc = pop[c]
            # Pick a random index R between [0, ndim)
            r = random.randrange(ndim)

            # Compute the agent's potentially new position y as follows
            y = toolbox.clone(agent)
            for i, value in enumerate(agent):
                # if random.random() < cr:  # exp
                if i == r or random.random() < cr:  # bin
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
        fvalues = tuple(ind.fitness.values[0] for ind in pop)
        mean, std = mean_std(fvalues)
        minimum = min(fvalues)
        maximum = max(fvalues)
        print("Generation:", g, minimum, mean, maximum, std)
        print("Generation:", g, minimum, mean, maximum, std, file=OptiConfig.outfile)

    # end of evolution

    hof.update(pop)
    print(hof[0], file=OptiConfig.outfile)
    print(hof[0])

    return hof[0].fitness.values


def opti():
    outfile = open("optimization.txt", mode='wt')
    OptiConfig.outfile = outfile

    # load 3D model
    wdir = os.getcwd()
    hdir, _ = os.path.split(wdir)
    mpath = hdir + '/input/model.txt'
    try:
        modelfile = open(mpath, mode='rt')
    except OSError as err:
        print(err, file=outfile)
        outfile.close()
        return
    try:
        model = load3dmodel(modelfile)
    except Exception as err:
        print(err, file=outfile)
        outfile.close()
        return

    # load sample
    try:
        samplefile = open("sample.txt", mode='rt')
    except OSError as err:
        print(err, file=outfile)
        outfile.close()
        return
    samples = loadsample(samplefile)

    # Load shield configurations
    Shield.loadcsv()

    ndim = 0
    for m in model:
        # m =[[node_list], "shield_type", n1, n2]
        ms = apolygon(m[0])
        if m[1] == "swi" or m[1] == "swe":
            # swp ['thickness', 'rhop', 'rhot', 'yield', 'HB', 'C']
            Shield.nrho.append(Shield.swp[m[3]][2])
            Shield.narea.append(ms)
            Shield.nsurf.append(ms)
            Shield.nvec.append(npolygon(m[0]))
            Shield.ntype.append([m[1], m[2], m[3]])
            ndim += 1
        elif m[1] == "wsi" or m[1] == "wse":
            # wsp ['tb', 'tw', 'rhop', 'rhob', 'rhow', 'yeildb', 'yieldw', 'theta', 'S']
            Shield.nrho.append(Shield.wsp[m[3]][3])
            Shield.nrho.append(Shield.wsp[m[3]][4])
            Shield.narea.append(ms)
            Shield.narea.append(ms)
            Shield.nsurf.append(ms)
            Shield.nvec.append(npolygon(m[0]))
            Shield.ntype.append([m[1], m[2], m[3]])
            ndim += 2
        elif m[1] == "swsi" or m[1] == "swse":
            # swsp ['tb', 'tw', 'rhop', 'rhob', 'rhow', 'mcb', 'mcw', ...]
            Shield.nrho.append(Shield.swsp[m[3]][3])
            Shield.nrho.append(Shield.swsp[m[3]][4])
            Shield.narea.append(ms)
            Shield.narea.append(ms)
            Shield.nsurf.append(ms)
            Shield.nvec.append(npolygon(m[0]))
            Shield.ntype.append([m[1], m[2], m[3]])
            ndim += 2
        else:
            raise Exception(m)
    # end of model

    for sample in samples:
        _, vmag, vaz, vel, _ = sample
        vaz = radians(vaz)
        vel = radians(vel)

        v = [cos(vel) * cos(vaz), cos(vel) * sin(vaz), sin(vel)]
        for i, nvec in enumerate(Shield.nvec):
            if len(Shield.nvmax) < i + 1:
                Shield.nvmax.append(0.0)
                Shield.nsamples.append([])
            ctheta = -1*(v[0]*nvec[0] + v[1]*nvec[1] + v[2]*nvec[2])
            if ctheta > 0:
                Shield.nsamples[i].append(sample)
                if Shield.nvmax[i] < vmag:
                    Shield.nvmax[i] = vmag
        # end of surface
    # end od samples

    # load optimization config
    config = configparser.ConfigParser()
    config.read(hdir + "/input/opticonfig.txt")
    OptiConfig.fpenalty = config.getfloat("optimization configure", "fpenalty")
    OptiConfig.ngen = config.getint("optimization configure", "ngen")
    OptiConfig.cr = config.getfloat("optimization configure", "cr")
    OptiConfig.f = config.getfloat("optimization configure", "f")
    OptiConfig.mu = config.getint("optimization configure", "mu")
    OptiConfig.low = config.get("optimization configure", "low")
    OptiConfig.up = config.get("optimization configure", "up")

    low = list(map(float, OptiConfig.low.split(",")))
    up = list(map(float, OptiConfig.up.split(",")))

    de(fmass, ndim, low, up)

    return

if __name__ == '__main__':
    opti()
    exit()
