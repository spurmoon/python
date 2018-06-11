#!/bin/env python3
import csv
import os
from math import sin, cos, radians, degrees, sqrt, acos, e


class Shield:
    SHIELDS = {'swi', 'swe', 'wsi', 'wse', 'swsi', 'swse'}
    #
    swia = []
    swea = []
    swp = []
    wsia = []
    wsea = []
    wsp = []
    swsia = []
    swsea = []
    swsp = []

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

        a1 = a[0]
        a2 = a[1]
        a3 = a[2]
        a4 = a[3]

        # p = ['thickness', 'rhop', 'rhot', 'yield', 'HB', 'C']
        sigma = p[3]
        rout = p[2]
        roup = p[1]
        c1 = p[5]
        t = p[0]

        dc = a1 * t * (sigma/(rout * c1**2))**a2 * (roup/rout)**a3 * (v/c1)**a4
        return dc

    # Single Wall External
    # def swe(a1, a2, a3, a4, bh, rout, roup, c1, t, v):
    @staticmethod
    def swe(a, p, v, theta):
        v *= cos(radians(theta))

        a1 = a[0]
        a2 = a[1]
        a3 = a[2]
        a4 = a[3]

        # p = ['thickness', 'rhop', 'rhot', 'yield', 'HB', 'C']
        bh = p[4]
        rout = p[2]
        roup = p[1]
        c1 = p[5]
        t = p[0]

        dc = ((t/1.8) * (bh ** a1 * (rout / roup) ** a2) / (a3 * (v / c1) ** a4)) ** (18 / 19)
        return dc

    # Whipple Shield Internal
    # def wsi(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, tb, tw, roup, roub, rouw, sigmb, sigmw, theta, s, v):
    @staticmethod
    def wsi(a, p, v, theta):
        a1 = a[0]
        a2 = a[1]
        a3 = a[2]
        a4 = a[3]
        a5 = a[4]
        a6 = a[5]
        a7 = a[6]
        a8 = a[7]
        a9 = a[8]
        a10 = a[9]

        # p = ['tb', 'tw', 'rhop', 'rhob', 'rhow', 'yeildb', 'yieldw', 'theta', 'S']
        tb = p[0]
        tw = p[1]
        roup = p[2]
        roub = p[3]
        rouw = p[4]
        sigmb = p[5]
        sigmw = p[6]
        s = p[8]

        vls = 3 * cos(radians(theta))**(-1/2)
        vhs = 7 * cos(radians(theta))**(-1/3)

        if v < vls:
            dc = a1 * (a2 * tb * (roub/rouw) * (sigmb/sigmw)**a3 + tw) * (rouw/roup) ** a4 * (roup * v**2/sigmw)**a5
        elif v > vhs:
            dc = a6 * tb**a7 * tw**a8 * s ** (1 - a8 - a7) * (rouw / roup) ** a9 * (roup * v ** 2 / sigmw) ** a10
        else:
            dc_1 = a1 * (a2 * tb * (roub/rouw) * (sigmb/sigmw)**a3 + tw) * (rouw/roup)**a4 * (roup * vls**2/sigmw)**a5
            dc_2 = a6 * tb**a7 * tw**a8 * s ** (1 - a8 - a7) * (rouw / roup) ** a9 * (roup * vhs ** 2 / sigmw) ** a10
            dc = dc_1 * (7 - v)/4 + dc_2 * (v - 3)/4
        return dc

    # Whipple Shield External
    # def wse(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, tb, tw, roup, roub, sigmw, theta, s, v):
    @staticmethod
    def wse(a, p, v, theta):
        a1 = a[0]
        a2 = a[1]
        a3 = a[2]
        a4 = a[3]
        a5 = a[4]
        a6 = a[5]
        a7 = a[6]
        a8 = a[7]
        a9 = a[8]
        a10 = a[9]

        # p = ['tb', 'tw', 'rhop', 'rhob', 'rhow', 'yeildb', 'yieldw', 'theta', 'S']
        tb = p[0]
        tw = p[1]
        roup = p[2]
        roub = p[3]
        sigmw = p[6]
        s = p[8]

        theta = radians(theta)
        if v < 3:
            dc = (tw * (sigmw/40)**a1 + tb) / (a2 * roup**a3 * v**a4 * cos(theta)**a5)**(18/19)
        elif v > 7:
            dc = (a6 * tw**(2/3) * s**(1/3) * (sigmw/70)**a7) / (roup**a8 * roub**a9 * v**a5 * cos(theta)**a10)
        else:
            dc_1 = (tw * (sigmw/40)**a1 + tb) / (a2 * roup**a3 * 3.**a4 * cos(theta)**a5)**(18/19)
            dc_2 = (a6 * tw**(2/3) * s**(1/3) * (sigmw/70)**a7) / (roup**a8 * roub**a9 * 7.**a5 * cos(theta)**a10)
            dc = dc_1 * (7. - v)/4 + dc_2 * (v - 3.)/4
        return dc

    # Stuffed Whipple Shield internal
    # def swsi(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11,
    #          tb, tw, roup, rouw, mroucb, mroucw, sigmb, sigmw, sigmcb, sigmcw, theta, s, v):
    @staticmethod
    def swsi(a, p, v, theta):
        a1 = a[0]
        a2 = a[1]
        a3 = a[2]
        a4 = a[3]
        a5 = a[4]
        a6 = a[5]
        a7 = a[6]
        a8 = a[7]
        a9 = a[8]
        a10 = a[9]
        a11 = a[10]

        # p = ['tb', 'tw', 'rhop', 'rhob', 'rhow', 'mcb', 'mcw', 'yieldb', 'yieldw', 'sigmcb', 'sigmcw', 'theta', 'S']
        tb = p[0]
        tw = p[1]
        roup = p[2]
        rouw = p[4]
        mroucb = p[5]
        mroucw = p[6]
        sigmb = p[7]
        sigmw = p[8]
        sigmcb = p[9]
        sigmcw = p[10]
        s = p[12]

        vls = 2.6 * cos(radians(theta))**(-1/2)
        vhs = 6.5 * cos(radians(theta))**(-1/3)
        if v < vls:
            dc = a1*(tw + tb * (sigmb/sigmw)**a2 + mroucb/rouw * (sigmcb/sigmw)**a2 + mroucw/rouw * (sigmcw/sigmw)**a2)\
                 * (rouw/roup)**a3 * (roup * (v * cos(radians(theta)))**2/sigmw)**a4
        elif v > vhs:
            dc = a5 * (tb**a6 * tw**a7 * s**(1 - a6 - a7) + a8 * (mroucb/rouw)**a9 * (mroucw/rouw)**(1 - a9))\
                 * (rouw/roup)**a10 * (roup * (v * cos(radians(theta)))**2/sigmw)**a11
        else:
            dc_1 = a1*(tw+tb * (sigmb/sigmw)**a2 + mroucb/rouw * (sigmcb/sigmw)**a2 + mroucw/rouw * (sigmcw/sigmw)**a2)\
                   * (rouw/roup)**a3 * (roup * (vls * cos(radians(theta)))**2/sigmw)**a4
            dc_2 = a5 * (tb**a6 * tw**a7 * s**(1 - a6 - a7) + a8 * (mroucb/rouw)**a9 * (mroucw/rouw)**(1 - a9))\
                * (rouw/roup)**a10 * (roup * (vhs * cos(radians(theta)))**2/sigmw)**a11
            dc = dc_1 * (6.5 - v)/3.9 + dc_2 * (v - 2.6)/3.9
        return dc

    # Stuffed Whipple Shield external
    #     def swse(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13,
    #              tb, tw, roup, roub, rouw, mroucb, mroucw, sigmw, theta, s, v):
    @staticmethod
    def swse(a, p, v, theta):
        a1 = a[0]
        a2 = a[1]
        a3 = a[2]
        a4 = a[3]
        a5 = a[4]
        a6 = a[5]
        a7 = a[6]
        a8 = a[7]
        a9 = a[8]
        a10 = a[9]
        a11 = a[10]
        a12 = a[11]
        a13 = a[12]

        # p = ['tb', 'tw', 'rhop', 'rhob', 'rhow', 'mcb', 'mcw', 'yieldb', 'yieldw', 'sigmcb', 'sigmcw', 'theta', 'S']
        tb = p[0]
        tw = p[1]
        roup = p[2]
        roub = p[3]
        rouw = p[4]
        mroucb = p[5]
        mroucw = p[6]
        sigmw = p[8]
        s = p[12]

        theta = radians(theta)
        vls = 2.6 / (cos(theta))**0.5
        vhs = 6.5 / (cos(theta))**0.75
        if v < vls:
            dc = a1 * (tw * (sigmw/40)**a2 + a3 * (roub * tb + mroucb + mroucw))/(roup**a5 * v**a6 * (cos(theta))**a4)
        elif v >= vhs:
            dc = a7 * (tw * rouw)**a8 * s ** a9 * (sigmw/40)**a10/(roup**a11 * v ** a12 * (cos(theta)) ** a13)
        else:
            dc_1 = a1*(tw * (sigmw/40)**a2 + a3 * (roub * tb + mroucb + mroucw))/(roup**a5 * vls**a6 * (cos(theta))**a4)
            dc_2 = a7 * (tw * rouw)**a8 * s**a9 * (sigmw/40)**a10/(roup**a11 * vhs**a12 * (cos(theta)) ** a13)
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
            for i in range(len(ss)):
                s = ss.pop(0)
                if s not in Shield.SHIELDS:
                    nidx = int(s) - 1
                    sn.append(node[nidx])
                else:
                    stype1 = int(ss.pop(0))
                    stype2 = int(ss.pop(0))
                    surf.append([sn, s, stype1, stype2])
                    break
            # end surface
        # end comment + node + surface
        else:
            raise Exception(line)
    # end infile
    return surf


def loadsample(infile):
    sample = []
    for line in infile:
        d, v, az, el, p = map(float, line.split())
        sample.append([d, v, az, el, p])
    return sample


# Hyper Velocity impact test
def hvitest(nvec, shieldtype, n1, n2, samples):
    sump = 0

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

    clim = cos(radians(85.))  # maximum theta
    for sample in samples:
        dia = sample[0]
        vmag = sample[1]
        vaz = radians(sample[2])
        vel = radians(sample[3])
        pop = sample[4]

        v = [vmag*cos(vel)*cos(vaz), vmag*cos(vel)*sin(vaz), vmag*sin(vel)]
        ctheta = -1 * (v[0]*nvec[0] + v[1]*nvec[1] + v[2]*nvec[2])/vmag

        if ctheta < clim:
            continue
        else:
            theta = degrees(acos(ctheta))
            if ble(a, p, vmag, theta) < dia:
                sump += pop*ctheta

    return sump


def main():
    outfile = open("RiskAssessment.txt", mode='wt')

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

    # Hyper velocity impact test
    total = 0
    for i, part in enumerate(model, start=1):
        area = apolygon(part[0])
        nvec = npolygon(part[0])
        fp_m2_year = hvitest(nvec, part[1], part[2], part[3], samples)
        np = fp_m2_year*area  # for time span of one year
        total += np
        print("S{} {:>10.5f}%".format(i, e**-np*100), file=outfile)

    print("TOTAL {:>10.5f}%".format(e**-total*100), file=outfile)
    outfile.close()
    return

if __name__ == "__main__":
    main()
