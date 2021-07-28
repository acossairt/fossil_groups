# Find the n smallest masses in a given forest_table
# Latest updates

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
from find_halos import find_halos
from find_halos import bin_halos
from find_halos import get_list_dims

# Read in data
forest_table = pd.read_hdf('/data/a/cpac/aurora/reducedtrees0.h5', 'ft')
with open("/data/a/cpac/xinliu/reducedtrees/progenitor_idx0.txt") as f:
    progenitor_idx = [[int(p) for p in s.strip().split(" ")] if s.strip() != "" else [] for s in f]

# Set up globals
n = 5
snap = 99 
norm = np.int(2.7*10**9)
extremum = ''
my_range = [10**13, 10**14]
my_range_str = ["{:.2e}".format(my_range[0]), "{:.2e}".format(my_range[1])]
bins = [10**9.5, 10**10.5, 10**11.5, 10**12.5, 10**13.5, 10**14.5]
norm_tf = True
avg_tf = True

# Find desired halos
halo_idx = bin_halos(forest_table, snap, bins)

# Track mass evolution for each halo
def track_evol(idx, redshifts, bins='', avg = False, normalized = False):

    z_nums = []
    final_masses = []
    
    #i_range = len(bins) - 1
    i_range = 3

    # Loop over all the bins
    for i in np.arange(i_range):
        #j_range = len(idx[i])
        j_range = 10
        current_bin_masses = np.empty((j_range,101))
        
        # Loop over each halo in the bin
        for j in np.arange(j_range):
            target_idx = np.int(idx[i][j])
            main_progenitor_list = [target_idx]
            progenitors = progenitor_idx[target_idx]
 
            # Trace the halo back in time; generate list of progenitors
            while len(progenitors) > 0:
                masses = [forest_table['mass'][k] for k in progenitors]
                main_progenitor = progenitors[np.argmax(masses)]
                main_progenitor_list.append(main_progenitor)
                progenitors = progenitor_idx[main_progenitor]
                
            # Save info for this halo as a row in current_bin_masses; standardize the array of masses
            current_halo_masses = np.array([forest_table['mass'][mp] for mp in main_progenitor_list])  # Mass at each snapnum for the current halo (j)
            masses_std = np.append(current_halo_masses, np.zeros(101 - len(current_halo_masses)))  # Standardize mass array to give length 101
            
            # Normalize, if desired
            if normalized == True:
                #print("normalizing by: ", forest_table['mass'][target_idx], " which is of type: ", type(forest_table['mass'][target_idx]))
                masses_std = masses_std / np.array(forest_table['mass'][target_idx]) #masses_std[len(masses_std) - 1]
            
            current_bin_masses[j] = masses_std   # Array of arrays: contains mass arrays for all halos in this bin

        # Take the average of all the mass evolutions of all the halos in that bin        
        avg_masses = np.average(current_bin_masses, axis = 0)
        
        # Change zeros to NAN so they won't be plotted later
        avg_masses[avg_masses == 0] = np.nan
            
        # Save info for this bin
        z_nums.append(redshifts)
        final_masses.append(avg_masses)

    print("z nums is: ", z_nums)
    print("Final mass is: ", final_mass)
    return z_nums, final_masses

# Convert snapshots to redshifts
#redshifts = np.array(np.logspace(0, 1.04312639797, 101, base = 10.0) - 1)
redshifts = np.array([10.044, 9.8065, 9.5789, 9.3608, 9.1515, 8.7573, 8.5714, 8.3925, 8.0541, 7.8938, 7.7391, 7.4454, 7.3058, 7.04, 6.9134, 6.6718, 6.5564, 6.3358, 6.1277, 6.028, 5.8367, 5.6556, 5.4839, 5.3208, 5.2422, 5.0909, 4.9467, 4.7429, 4.6145, 4.4918, 4.3743, 4.2618, 4.1015, 4.00, 3.8551, 3.763, 3.6313, 3.5475, 3.4273, 3.3133, 3.205, 3.102, 3.0361, 2.9412, 2.8506, 2.7361, 2.6545, 2.5765, 2.4775, 2.4068, 2.3168, 2.2524, 2.1703, 2.0923, 2.018, 1.9472, 1.8797, 1.7994, 1.7384, 1.68, 1.6104, 1.5443, 1.4938, 1.4334, 1.3759, 1.321, 1.2584, 1.2088, 1.152, 1.1069, 1.0552, 1.006, 0.9591, 0.9143, 0.8646, 0.824, 0.7788, 0.7358, 0.6948, 0.6557, 0.6184, 0.5777, 0.5391, 0.5022, 0.4714, 0.4337, 0.4017, 0.3636, 0.3347, 0.3035, 0.2705, 0.2423, 0.2123, 0.1837, 0.1538, 0.1279, 0.1008, 0.0749, 0.0502, 0.0245, 0.00])

# Actually implement the function!
z_nums, masses = track_evol(halo_idx, redshifts, bins, avg_tf, norm_tf)
masses_norm = [np.array(masses[i])*norm for i in np.arange(len(masses))]

# Plot the results
def plot_evol(z_nums, masses, bins, avg = False, normalized = False):
    fig, ax = plt.subplots()
    color=iter(cm.jet(np.linspace(0,1,len(masses))))
    
    if avg == True:
        for m in np.arange(len(masses)):
            ax.plot(z_nums[m], masses[m], color = next(color), label = ("bin " + str(m + 1) + ": (" + "{:.2e}".format(bins[m]) + " to " + "{:.2e}".format(bins[m+1]) + ")"))
        ax.legend()
        if normalized == True:
            ax.set_title("Normalized averaged mass evolution of halos in " + str(int(len(bins) - 1)) + " bins")
        elif normalized == False:
            ax.set_title("Averaged mass evolution of halos in " + str(int(len(bins) - 1)) + " bins")

    elif avg == False:
        for m in np.arange(len(masses)):
            ax.plot(z_nums[m], masses[m], color = next(color))
        if extremum == 'max':
            ax.set_title("Mass evolution of " + str(n) + " most massive halos")
        elif extremum == 'min':
            ax.set_title("Mass evolution of " + str(n) + " least massive halos")
        elif extremum == '':
            ax.set_title("Mass evolution of halos in range " + my_range_str[0] + " to " + my_range_str[1])
       
    ax.set_yscale('log', nonpositive = 'clip')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Mass [M_sun/h]')
    plt.savefig('full_mass_evol_bins_z_norm.png')
    plt.show()

plot_evol(z_nums, masses_norm, bins, avg_tf, norm_tf)
