#!/bin/env python3
import re
import random
from math import log10, ceil, trunc, degrees, radians, sqrt, cos, sin, atan2, acos

import os
if 'tools' not in os.listdir():
    os.chdir("..")
os.chdir('tools')

# VDIM = 23
# ADIM = 36


def fromdat(filename, skip_header, skip_footer, usecols):
    try:
        file = open(filename)
        lines = file.readlines()[skip_header:-skip_footer]
    except OSError as e:
        print(e)
        return None
    file.close()

    res = [[] for _ in usecols]
    for line in lines:
        l = line.split()
        l = list(map(float, l))
        for i, c in enumerate(usecols):
            res[i].append(l[c])
    return res


def gettags(filename):
    try:
        vdat = open(filename)
    except OSError as e:
        print(e)
        return None, None, None

    expr = re.compile(" >(.+[umc]?m)\s+(\d+-\d+).+\s(\d+)\s+\d+\.\d+")
    while "size       Vmag".upper() not in vdat.readline().upper():
        continue
    dsize = []
    vrang = []
    angle = []
    for l in vdat:
        dsi, vra, ang = expr.findall(l)[0]
        if dsi not in dsize:
            dsize.append(dsi)
        if vra not in vrang:
            vrang.append(vra)
        if ang not in angle:
            angle.append(ang)
    vdat.close()

    return dsize, vrang, angle


def getthreshold(filename):
    try:
        cmd = open(filename)
    except OSError as e:
        print(e)
        return None
    expr = re.compile("\sthreshold=(.+m)\s")
    th = expr.findall(cmd.readline())[0]
    return th


class FluxOfSize:
    def __init__(self, x, y, yp1=None, ypn=None):
        self.n = len(x)
        self.m = 2
        self.x = list(x)
        self.y = list(y)
        self.y2 = [0.0]*self.n
        self.jsave = -1
        self.correlated = 0
        self.dj = min(1, self.n**0.25)
        self.modifiedb = 0.0
        self.modifiede = 0.0
        self.sety2(x, y, yp1, ypn)
        self.modify()
        return

    # Set up for interpolating on a table of x’s and y’s of length m
    # Given a value x, return a value j such that x is (insofar as possible) centered in the sub-range
    # xx[j..j+mm-1], where xx is the stored pointer. The values in xx must be monotonic, either
    # increasing or decreasing. The returned value is not less than 0, nor greater than n-1.
    def locate(self, xp):
        if self.n < 2 or self.m < 2 or self.m > self.n:
            raise ValueError("locate size error.")

        n = self.n
        ascnd = (self.x[n-1] >= self.x[0])  # True if ascending order of table, false otherwise
        jl = 0  # Initialize lower limits
        ju = n - 1  # Initialize upper limits

        while ju - jl > 1:  # If we are not yet done, compute a midpoint
            jm = (ju+jl) >> 1
            if (xp >= self.x[jm]) == ascnd:
                jl = jm
            else:
                ju = jm

        self.correlated = 0 if abs(jl - self.jsave) > self.dj else 1  # Decide whether to use hunt or locate next time.
        self.jsave = jl
        return max(0, min(n - self.m, jl - ((self.m - 2) >> 1)))

    # Given a value x, return a value j such that x is (insofar as possible) centered in the sub-range
    # xx[j..j+mm-1], where xx is the stored pointer. The values in xx must be monotonic, either
    # increasing or decreasing. The returned value is not less than 0, nor greater than n-1.
    def hunt(self, xp):
        if self.n < 2 or self.m < 2 or self.m > self.n:
            raise ValueError("hunt size error.")

        n = self.n
        ascnd = (self.x[n-1] >= self.x[0])  # True if ascending order of table, false otherwise
        jl = self.jsave
        inc = 1

        if jl < 0 or jl > n - 1:  # Input guess not useful. Go immediately to bisection
            jl = 0
            ju = n-1
        else:
            if (xp >= self.x[jl]) == ascnd:
                while True:
                    ju = jl + inc
                    if ju >= n-1:
                        ju = n-1
                        break
                    elif (xp < self.x[ju]) == ascnd:
                        break
                    else:
                        jl = ju
                        inc += inc
            else:
                ju = jl
                while True:
                    jl = jl - inc
                    if jl <= 0:
                        jl = 0
                        break
                    elif (xp >= self.x[jl]) == ascnd:
                        break
                    else:
                        ju = jl
                        inc += inc

        while ju - jl > 1:
            jm = (ju+jl) >> 1
            if (xp >= self.x[jm]) == ascnd:
                jl = jm
            else:
                ju = jm

        self.correlated = 0 if abs(jl - self.jsave) > self.dj else 1
        self.jsave = jl
        return max(0, min(n - self.m, jl - ((self.m - 2) >> 1)))

    # Given a value x, and using pointers to data xx and yy, and the stored vector of second derivatives
    # y2, this routine returns the cubic spline interpolated value y.
    def interp(self, xp):
        jlo = self.hunt(xp) if self.correlated == 1 else self.locate(xp)

        klo = jlo
        khi = jlo + 1

        h = self.x[khi] - self.x[klo]
        if h == 0.0:
            raise ValueError("Bad input to spline interpolation.")
        a = (self.x[khi] - xp) / h
        b = (xp - self.x[klo]) / h
        yp = a*self.y[klo] + b*self.y[khi] + ((a**3 - a)*self.y2[klo] + (b**3 - b)*self.y2[khi])*(h*h)/6.0
        return yp

    # This routine stores an array y2[0..n-1] with second derivatives of the interpolating function
    # at the tabulated points pointed to by xv, using function values pointed to by yv. If yp1 and/or
    # ypn are equal to 1*10^99 or larger, the routine is signaled to set the corresponding boundary
    # condition for a natural spline, with zero second derivative on that boundary; otherwise, they are
    # the values of the first derivatives at the endpoints.
    def sety2(self, x, y, yp1, ypn):
        n = self.n
        u = [0.0]*(n-1)
        if yp1:                         # or else to have a specified first derivative.
            self.y2[0] = -0.5
            u[0] = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1)
        else:                           # The lower boundary condition is set either to be “natural"
            self.y2[0] = u[0] = 0.0

        for i in range(1, n-1):         # This is the decomposition loop of the tridiagonal algorithm.
            sig = (x[i]-x[i-1]) / (x[i+1]-x[i-1])
            p = sig * self.y2[i-1]+2.0
            self.y2[i] = (sig-1.0) / p
            u[i] = (y[i+1]-y[i]) / (x[i+1]-x[i]) - (y[i]-y[i-1]) / (x[i]-x[i-1])
            u[i] = (6.0 * u[i] / (x[i+1]-x[i-1])-sig * u[i-1]) / p

        if ypn:                         # The upper boundary condition is set either to be “natural"
            qn = 0.5
            un = (3.0 / (x[n - 1] - x[n - 2])) * (ypn - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]))
        else:                           # or else to have a specified first derivative.
            qn = un = 0.0

        self.y2[n-1] = (un - qn*u[n-2])/(qn*self.y2[n-2] + 1.0)
        for k in range(n-2, -1, -1):    # This is the back substitution loop of the tridiagonal algorithm.
            self.y2[k] = self.y2[k] * self.y2[k+1] + u[k]
        return

    def modify(self):
        step = 0.05
        i = self.x[0]
        while i < self.x[-1]:
            if self.interp(i) < self.interp(i + step):
                minx = golden_section_search(self.interp, i - step, i + step)
                maxx = golden_section_search(self.interp, i + step, self.x[-1], kind='max')
                self.modifiedb = bisection_search(self.interp, self.x[0], minx, self.interp(maxx))
                self.modifiede = bisection_search(self.interp, maxx, self.x[-1], self.interp(minx))
            i += step
        return

    def modified_interp(self, xp):
        if self.modifiedb == self.modifiede:
            return self.interp(xp)
        elif self.modifiedb < xp < self.modifiede:
            b = self.interp(self.modifiedb)
            e = self.interp(self.modifiede)
            return b + (e - b) / (self.modifiede - self.modifiedb) * (xp - self.modifiedb)
        else:
            return self.interp(xp)

    def cdf(self, size):
        if size < self.x[0] or size > self.x[-1]:
            raise ValueError("size out of range.")

        return 10**(self.modified_interp(size) - self.y[0])

    def random_size(self, threshold):
        # random sample according to the flux vs size distribution
        pcondition = self.cdf(threshold)
        try:
            s = bisection_search(self.cdf, self.x[0], self.x[-1], random.random()*pcondition)
        except ValueError:
            s = self.x[-1]

        return s


def golden_section_search(f, a, b, tolerance=1e-10, kind='min'):
    # f should be a strictly unimodal function
    if kind != 'max' and kind != 'min':
        raise ValueError("Unknown kind of search.")

    # Golden ration
    gr = (sqrt(5) + 1)/2

    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(c - d) > tolerance:
        if kind == 'min':
            if f(c) < f(d):
                b = d
            else:
                a = c
        else:
            if f(c) > f(d):
                b = d
            else:
                a = c
        # recompute both c and d to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return (b + a) / 2


def bisection_search(f, a, b, f0, tolerance=1e-10):
    # f should be a monotone increasing or decreasing function
    fa = f(a) - f0
    fb = f(b) - f0
    if fa*fb > 0:
        # print(a, b, fa, fb, f0)
        raise ValueError("ill-conditioned search range.")

    while abs(a - b) > tolerance:
        fab = f((a + b)/2) - f0
        if fab == 0:
            break
        if fab*fa > 0:
            a = (a + b)/2
        else:
            b = (a + b)/2
    return (a + b)/2


def main():
    f10um, f100um, f1mm, f1cm, f10cm, f1m = fromdat("results/TABLESC.DAT",
                                                    skip_header=5,
                                                    skip_footer=2,
                                                    usecols=(3, 4, 5, 6, 7, 8))
    flux = [f10um, f100um, f1mm, f1cm, f10cm, f1m]

    th = getthreshold('ORDEM2000.CMD')
    thresholds = {'10um': 0, '100um': 1, '1mm': 2, '1cm': 3, '10cm': 4, '1m': 5}
    thidx = thresholds[th]

    segs = len(f10um)
    size, vrang, angle = gettags("results/VREL0001.DAT")

    # bfly10um = [[0.0] * ADIM for _ in range(VDIM)]
    # bfly100um = [[0.0] * ADIM for _ in range(VDIM)]
    # bfly1mm = [[0.0] * ADIM for _ in range(VDIM)]
    # bfly1cm = [[0.0] * ADIM for _ in range(VDIM)]
    # bfly10cm = [[0.0] * ADIM for _ in range(VDIM)]
    # bfly1m = [[0.0] * ADIM for _ in range(VDIM)]
    # butterfly = [bfly10um, bfly100um, bfly1mm, bfly1cm, bfly10cm, bfly1m]

    outfile = open("sample.txt", mode='wt')

    expr = re.compile(" >.+[umc]?m\s+\d+-\d+.+\s\d+\s+(\d+\.\d+)")
    expe = re.compile(".+:\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)")
    vstride = len(angle)
    sstride = len(angle)*len(vrang)
    for i in range(segs):
        # print("\r{:5.1f}%".format(i/segs*100), end='')
        try:
            vdat = open("results/VREL{:0>4}.DAT".format(i+1))
        except OSError as e:
            # print(e)
            return
        while "S/C velocity components (km/s)".upper() not in vdat.readline().upper():
            continue
        else:
            r, veast0, vnorth0 = map(float, expe.findall((vdat.readline()))[0])
            a0 = atan2(vnorth0, veast0)
        while "size       Vmag".upper() not in vdat.readline().upper():
            continue

        percentage = []
        for l in vdat:
            per = expr.findall(l)[0]
            percentage.append(float(per)/100.)
        vdat.close()

        # flux vs size distribution in local
        sizex = [-3., -2., -1., 0., 1., 2.]     # log10(size) for size in 0.001cm 0.01cm 0.1cm 1cm 10cm 100cm
        fluxy = [log10(f10um[i]), log10(f100um[i]), log10(f1mm[i]), log10(f1cm[i]), log10(f10cm[i]), log10(f1m[i])]
        foz = FluxOfSize(sizex, fluxy)

        for s, stag in enumerate(size):
            if s != thidx:
                continue
            for j, vtag in enumerate(vrang):
                v1, v2 = map(float, vtag.split('-'))
                for k, atag in enumerate(angle):
                    if percentage[s*sstride + j*vstride + k] < 1e-8:
                        continue

                    t = flux[s][i]/segs * percentage[s*sstride + j*vstride + k]
                    n = max(10, 10**ceil(log10(t)))
                    #n = 1
                    fraction = t/n

                    a1 = float(angle[k])
                    a2 = a1 + 10.  # Default to 10 degrees according to ORDEM2000 specification

                    for ii in range(n):
                        # randomize the diameter according to the flux vs size distribution
                        try:
                            z = 10**foz.random_size(sizex[thidx])
                        except ValueError:
                            continue

                        # randomize the velocity and angle
                        v = random.uniform(v1, v2)
                        a = random.uniform(a1, a2)
                        a = radians(a)
                        veast = v*cos(a)
                        vnorth = v*sin(a)
                        # relative velocity
                        relv = sqrt((veast-veast0)**2 + (vnorth-vnorth0)**2)
                        # relative angle in -180 to 180 degrees
                        rela = degrees(acos(cos(a)*cos(a0) + sin(a)*sin(a0)))
                        if cos(a)*-sin(a0) + sin(a)*cos(a0) < 0:
                            rela += -1.
                        print("{:>8.5f} {:>8.4f} {:>7.1f} {:>7.1f} {:>11.4e}".format(
                            z, relv, rela, 0, fraction), file=outfile)

                        # rearrange the relative angle in 0 to 360 degrees
                        # angl = rela - a0 if rela - a0 > 0 else rela - a0 + 360
                        # vi = trunc(relv)
                        # ai = trunc(angl/10.)
                        # butterfly[s][vi][ai] += fraction
                        # end single test
                    # end angle range
                # end velocity range
            # end all size
        # end all segments
    outfile.close()
    # print("\rfinished")

    # bflyfile = open("10um", mode='wt')
    # for s, stag in enumerate(size):
    #     for j in range(VDIM):
    #         for k in range(ADIM):
    #             print(butterfly[s][j][k], file=bflyfile, end=' ')
    #         print(file=bflyfile)
    #     break
    # bflyfile.close()
    return

if __name__ == '__main__':
    main()

