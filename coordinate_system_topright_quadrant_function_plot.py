import numpy as np                 # v 1.19.2
import matplotlib as mpl
import matplotlib.pyplot as plt    # v 3.3.2

# Enter figure size
xfigsize = 10
yfigsize = 10

# Enter x and y coordinates of points and colors
x = np.arange(-1, 5, 0.01)
y = ((1/x)*(1/x) + x*x)/2.0 # 1/2 SymDirchlet scaling energy function
# y = ((1/x)*(1/x) + x*x) # SymDirchlet scaling energy function
# y = np.array([((xe - 1) * (xe - 1)) if (xe >= 0) else ((-xe - 1) * (-xe - 1)) for xe in x]) # ARAP scaling energy function
# y = ((x - 1) * (x - 1)) if (x >= 0) else ((-x + 1) * (-x + 1)) # ARAP scaling energy function

colors = ['m', 'g', 'r', 'b']

# Select length of axes and the space between tick labels
xmin, xmax, ymin, ymax = 0, 3, 0, 4 # 1/2 SymDirichlet settings
# xmin, xmax, ymin, ymax = 0, 3, 1.5, 9 # SymDirichlet settings
# xmin, xmax, ymin, ymax = 0, 2.5, 0, 3 # ARAP settings

# Set step size of major ticks
x_tick_step = 0.5
y_tick_step = 1.0 # 0.5

# Set number of minor ticks
x_num_minor_ticks = 1
y_num_minor_ticks = 1

tick_space = xfigsize/40.0

# Calculate axis scales
xscale = (xmax - xmin) / xfigsize
yscale = (ymax - ymin) / yfigsize

xfigmin, xfigmax, yfigmin, yfigmax = xmin / xscale, xmax / xscale, ymin / yscale, ymax / yscale

# Plot points
fig, ax = plt.subplots(figsize=(xfigsize, yfigsize))
#ax.scatter(xs, ys, c=colors)
#ax.fill(x / xscale, y / yscale, facecolor='none', edgecolor='#1f77b4', ls='-', linewidth=2) # fill for closed shapes
ax.plot(x / xscale, y / yscale, color='#1f77b4', ls='-', linewidth=2)

# Set identical scales for both axes
ax.set(xlim=(xfigmin-1, xfigmax+1), ylim=(yfigmin-1, yfigmax+1), aspect='equal')

# Set bottom and left spines as x and y axes of coordinate system
# ax.spines['bottom'].set_position('zero')
ax.spines['bottom'].set_position(['data', ymin / yscale])
ax.spines['left'].set_position(['data', xmin / xscale])
#ax.spines['left'].set_position('zero')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Create 'x' and 'y' labels placed at the end of the axes
# ax.set_xlabel('x', size=20, labelpad=-24, x=1.03) # std label size was 14
# ax.set_ylabel('y', size=20, labelpad=-21, y=1.02, rotation=0)
ax.set_xlabel('Scaling factor', size=20, labelpad=10.0) # std label size was 14
ax.set_ylabel('ARAP energy', size=20, labelpad=10.0) # std label size was 14
ax.xaxis.label.set_backgroundcolor('white')
ax.yaxis.label.set_backgroundcolor('white')

# Create custom major ticks to determine position of tick labels
x_tick_labels = np.arange(0, xmax+x_tick_step, x_tick_step)
y_tick_labels = np.arange(0, ymax+y_tick_step, y_tick_step)
print(y_tick_labels)

x_ticks = x_tick_labels / xscale
y_ticks = y_tick_labels / yscale

# filter tick lists
# tick_filter = (x_ticks > (xfigmin + tick_space)) | (x_ticks < (xfigmin - tick_space)) # filter only in vicinity of axes
tick_filter = (x_ticks > (xfigmin + tick_space)) # filter all ticks below x-axis 0 level
x_ticks = x_ticks[tick_filter]
x_tick_labels = x_tick_labels[tick_filter]
# tick_filter = (y_ticks > (yfigmin + tick_space)) | (y_ticks < (yfigmin - tick_space)) # filter only in vicinity of axes
tick_filter = (y_ticks > (yfigmin + tick_space)) # filter all ticks below y-axis 0 level
y_ticks = y_ticks[tick_filter]
y_tick_labels = y_tick_labels[tick_filter]

# ax.set_xticks(x_ticks, labels=[str(x) for x in x_tick_labels])
# ax.set_yticks(y_ticks, labels=[str(y) for y in y_tick_labels])
ax.set_xticks(x_ticks[x_ticks != 0], labels=[str(x) for x in x_tick_labels[x_tick_labels != 0]])
ax.set_yticks(y_ticks[y_ticks != 0], labels=[str(y) for y in y_tick_labels[y_tick_labels != 0]])

# Create minor ticks placed at each integer to enable drawing of minor grid
# lines: note that this has no effect in this example with ticks_frequency=1
x_ticks_minor = np.arange(0, xmax, x_tick_step / (x_num_minor_ticks + 1)) / xscale
y_ticks_minor = np.arange(0, ymax, y_tick_step / (y_num_minor_ticks + 1)) / yscale

#tick_filter = (x_ticks_minor > (xfigmin + tick_space)) | (x_ticks_minor < (xfigmin - tick_space)) # filter only ticks in vicinity of x-axis 0 level
tick_filter = (x_ticks_minor > (xfigmin + tick_space)) # filter only ticks below of x-axis 0 level
x_ticks_minor = x_ticks_minor[tick_filter]
#tick_filter = (y_ticks_minor > (yfigmin + tick_space)) | (y_ticks_minor < (yfigmin - tick_space)) # filter only tikcs in vicinity of x-axis 0 level
tick_filter = (y_ticks_minor > (yfigmin + tick_space)) # filter only ticks below of x-axis 0 level
y_ticks_minor = y_ticks_minor[tick_filter]

ax.set_xticks(x_ticks_minor, minor=True)
ax.set_yticks(y_ticks_minor, minor=True)

# Set tick settings
ax.tick_params(axis='both', which='major', labelsize=15)
ax.tick_params(axis='both', which='minor', labelsize=10)
plt.setp(ax.get_xticklabels(), backgroundcolor="white")
plt.setp(ax.get_yticklabels(), backgroundcolor="white")

# Draw major and minor grid lines
ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

# Draw arrows
arrow_fmt = dict(markersize=4, color='black', clip_on=False)
ax.arrow(xfigmin, yfigmin - 1, 0, yfigsize + 2, fc='k', ec='k', lw = 0.3, 
        head_width= xfigsize/40.0, head_length=yfigsize/40.0, overhang=0.3,
        length_includes_head= True, clip_on = False)

ax.arrow(xfigmin - 1, yfigmin, xfigsize + 2, 0, fc='k', ec='k', lw = 0.3, 
        head_width= xfigsize/40.0, head_length=yfigsize/40.0, overhang=0.3,
        length_includes_head= True, clip_on = False)

plt.show()
fig.savefig("figures/coordsys_topright_function.svg", bbox_inches='tight', pad_inches=0.1)