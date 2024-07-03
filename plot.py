import matplotlib.pyplot as plt
import numpy as np
import sys
import os

stepsizes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
file_name = sys.argv[1]
if file_name.endswith(".csv"):
    #with open("logs/kNearest_runtime_percentile_step_(torus.off).csv") as fp:
    #if not file_name.find("EnergyCurve"):
    #    exit(0)
    name = file_name.split(".")[0]
    with open("logs/"+file_name) as fp:
        Lines = fp.readlines()

    #Lines.split(";")
    l = [float(x.split(';')[0]) for x in Lines]
    l1 = [float(x.split(';')[1]) for x in Lines]
    l2 = [float(x.split(';')[2]) for x in Lines]

    # Data for plotting
    x = np.arange(0, len(Lines), 1)


    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(x, l)
    ax.plot(x, l1)
    ax.plot(x, l2)
    #ax.set_yscale("log")
    ax.set_yscale("log")

    ylabl = "Sum of squares energy"
    if (file_name.find("Det") != -1):
        ylabl = "Determinant energy"

    ax.set(xlabel='iterations', ylabel=ylabl,
        title='Energy curve (on cube.off)') #TODO: don't let this be hardcoded
    #ax.grid()

    fig.savefig("figures/"+name+".png")
    plt.show()

""" i = 0
for file in os.listdir("logs"):
    if file.endswith(".csv"):
        #with open("logs/kNearest_runtime_percentile_step_(torus.off).csv") as fp:
        if not file.find("EnergyCurve"):
            continue
        name = file.split(".")[0]
        with open("logs/"+file) as fp:
            Lines = fp.readlines()

        l = [float(x) for x in Lines]

        # Data for plotting
        x = np.arange(0, len(Lines), 1)


        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(x, l)

        ylabl = "Sum of squares energy"
        if (name == "quadQuadDetEnergyCurve"):
            ylabl = "Determinant energy"

        ax.set(xlabel='iterations', ylabel=ylabl,
            title='Energy curve (on quadQuad.off)')
        #ax.grid()

        fig.savefig("figures/"+name+".png")
        plt.show()
        i += 1 """