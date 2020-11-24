# Helper functions for running statistics on Last Journey Trees
# Functions include:
# - find_halos
# - bin_halos
# - plot_evol
# - track_evol
# - avg_bins
# - find_LMMs
# - plot_LMMs

# A few imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from itertools import groupby
from matplotlib.ticker import ScalarFormatter

# A few globals:
redshifts = np.flip(np.array([10.044, 9.8065, 9.5789, 9.3608, 9.1515, 8.7573, 8.5714, 8.3925, 8.0541, 7.8938, 7.7391, 7.4454, 7.3058, 7.04, 6.9134, 6.6718, 6.5564, 6.3358, 6.1277, 6.028, 5.8367, 5.6556, 5.4839, 5.3208, 5.2422, 5.0909, 4.9467, 4.7429, 4.6145, 4.4918, 4.3743, 4.2618, 4.1015, 4.00, 3.8551, 3.763, 3.6313, 3.5475, 3.4273, 3.3133, 3.205, 3.102, 3.0361, 2.9412, 2.8506, 2.7361, 2.6545, 2.5765, 2.4775, 2.4068, 2.3168, 2.2524, 2.1703, 2.0923, 2.018, 1.9472, 1.8797, 1.7994, 1.7384, 1.68, 1.6104, 1.5443, 1.4938, 1.4334, 1.3759, 1.321, 1.2584, 1.2088, 1.152, 1.1069, 1.0552, 1.006, 0.9591, 0.9143, 0.8646, 0.824, 0.7788, 0.7358, 0.6948, 0.6557, 0.6184, 0.5777, 0.5391, 0.5022, 0.4714, 0.4337, 0.4017, 0.3636, 0.3347, 0.3035, 0.2705, 0.2423, 0.2123, 0.1837, 0.1538, 0.1279, 0.1008, 0.0749, 0.0502, 0.0245, 0.00]))
norm = np.int(2.7*10**9)

###############################################################################
# Find Halos:                                                                 #
# return an index of halo id's whose mass values match the input requirements #
###############################################################################

def find_halos(forest_table, sn, quant = 0, extremum = '', mass_range = []):
        
    # Extract the mass values (and associated halo id's) from desired snapshot
    masses_sn = np.array(forest_table.mass.loc[forest_table.snap_num == sn]) # (2.7*10**9)*?
    halos_sn = np.array(forest_table.halo_id.loc[forest_table.snap_num == sn])

    # Sort masses_sn in order of smallest to largest values
    mass_order = np.argsort(masses_sn)
    sorted_halo_ids = halos_sn[mass_order]
    sorted_masses = masses_sn[mass_order]

    # Save only the desired values
    if extremum == 'min':
        halo_idx = sorted_halo_ids[:quant]

    elif extremum == 'max':
        halo_idx = sorted_halo_ids[-quant:]

    elif extremum == '':
        halo_idx = sorted_halo_ids[np.logical_and(sorted_masses >= mass_range[0], sorted_masses <= mass_range[1])]
        halo_values = np.array(forest_table['mass'][halo_idx])*2.7*10**9 # In case you're interested

    return halo_idx
   
#############################################################################
# Bin Halos:                                                                #
# return a 2D index of halo id's whose mass values fall into the given bins #
#############################################################################
    
def bin_halos(forest_table, sn, bins):
    
    # Extract the mass values and halo ids from the desired snapshot
    masses_sn = np.array(forest_table.mass.loc[forest_table.snap_num == sn]) #*(2.7*10**9)
    halo_ids_sn = np.array(forest_table.halo_id.loc[forest_table.snap_num == sn])
    
    # Thought this might save some compute time, but it seems to mess up my masking later... probs not worth it
    #masses_sn = forest_table.mass.loc[forest_table.snap_num == sn].values.tolist()
    #halo_ids_sn = forest_table.halo_id.loc[forest_table.snap_num == sn].values.tolist()
    
    # Assign a bin number to each halo
    bin_idx = np.digitize(masses_sn, bins)
    
    # Build list of all halos associated with each bin
    halo_idx = [halo_ids_sn[bin_idx == j] for j in range(1, len(bins))]
    
    return halo_idx

#####################################################################################################
# Track Evolution: single halo                                                                      #
# build a main progenitor branch for one halo;                                                      #
# return masses, time steps, main progenitor id's, and locations of major mergers for its mp branch #
#####################################################################################################
    
def track_evol(idx, redshifts, prog_idx, forest_tbl, mm_thresh_small = 0.1, mm_thresh_big = 0.3, normalized = False, x_axis = 'z_nums', z_threshold = 1): # Just track the evolution of one halo -- bells and whistles come later
    
    major_mergers = [np.empty(0), np.empty(0)] # Do these have to be np arrays? # Try lists too
    major_merger_times = [np.empty(0), np.empty(0)]
    #LMMs = [np.empty(0), np.empty(0)]
    LMMs = [None, None]
    LMM_times = [None, None]
    fossil_clusters = [np.empty(0), np.empty(0)]
    
    # Start tracking
    target_idx = int(idx)  # np.int(np.array(idx))
    progenitors = prog_idx[target_idx]
    z_current = redshifts[100 - forest_tbl.snap_num[target_idx]]
    main_progenitor_list = [target_idx]

    # Generate main branch: find all the progenitors of the halo (at each timestep)
    while len(progenitors) > 0:
        masses = [forest_tbl['mass'][i] for i in progenitors] # Find masses for each progenitor of the halo at this time step

        # Count major mergers
        if len(progenitors) > 1:
            order_by_mass = np.argsort(masses)[::-1] # Special syntax for reversing order
            sorted_progenitors = np.array(progenitors)[order_by_mass] # Why do I need np.array here?
            sorted_masses = np.array(masses)[order_by_mass]

            # Is this a major merger (by first definition?)
            if sorted_masses[0]*(mm_thresh_small) < sorted_masses[1]:
                
                # Is this an LMM?
                if LMM[0] is None:
                #if len(major_mergers[0]) == 0:
                    LMMs[0] = sorted_progenitors[0]
                    LMM_times[0] = forest_tbl['snap_num'][sorted_progenitors[0]]
                    
                    # Is this a potential fossil cluster?
                    if z_current >= z_threshold:
                        fossil_clusters[0] = np.append(fossil_clusters[0], target_idx)
                
                major_mergers[0] = np.append(major_mergers[0], sorted_progenitors[0])
                # Ack
                #major_merger_times[0] = np.append(major_merger_times[0], redshifts[100 - forest_tbl['snap_num'][sorted_progenitors[0]]])
                major_merger_times[0] = np.append(major_merger_times[0], forest_tbl['snap_num'][sorted_progenitors[0]])
                
            # Is this also a major merger by the second definition?
            if sorted_masses[0]*(mm_thresh_big) < sorted_masses[1]:
                # How to remove this duplication?

                # Change this
                # Is this an LMM under the second definition?
                if len(major_mergers[1]) == 0:
                    LMMs[1] = np.append(LMMs[1], sorted_progenitors[0])
                    LMM_times[1] = np.append(LMM_times[1], forest_tbl['snap_num'][sorted_progenitors[0]])
                    # Ack
                    #LMM_times[1] = np.append(LMM_times[1], redshifts[100 - forest_tbl['snap_num'][sorted_progenitors[0]]])

                    # Is this a potential fossil cluster under the second definition?
                    if z_current >= z_threshold:
                        fossil_clusters[1] = np.append(fossil_clusters[1], target_idx)

                major_mergers[1] = np.append(major_mergers[1], sorted_progenitors[0]) 
                #Ack
                #major_merger_times[1] = np.append(major_merger_times[1], redshifts[100 - forest_tbl['snap_num'][sorted_progenitors[0]]])
                major_merger_times[1] = np.append(major_merger_times[1], forest_tbl['snap_num'][sorted_progenitors[0]])
            
        # Continue
        main_progenitor = progenitors[np.argmax(masses)]
        main_progenitor_list.append(main_progenitor)
        progenitors = prog_idx[main_progenitor]
        z_current = redshifts[100 - forest_tbl.snap_num[main_progenitor]]

    # Also, do these have to be np arrays?
    raw_masses = [forest_tbl['mass'][mp] for mp in main_progenitor_list]  # Mass of each main progenitor for this halo 
    final_masses = np.append(raw_masses, np.zeros(101 - len(raw_masses))) #*norm  # Standardize mass array with length 101 and un-normalized mass

    #print("For this halo, len(major_mergers[0]): ", len(major_mergers[0]), " and len(major_mergers[1]): ", len(major_mergers[1]))
    #if len(major_mergers[0]) == 0: 
    #    print("Apparently there was nothing here")
    #    print("major_mergers: ", major_mergers)
    
    if normalized == True:
        # Probably doesn't work
        #final_masses = final_masses / forest_tbl['mass'][target_idx].values.tolist() #masses_std[len(masses_std) - 1]
        final_masses = final_masses / forest_tbl['mass'][target_idx] # Doesn't need .values because there's just one

    # Return all the things you need!
    if x_axis == 'z_nums':
        z_nums = redshifts
        return z_nums, final_masses, main_progenitor_list, major_mergers, major_merger_times, LMMs, LMM_times, fossil_clusters
    elif x_axis == 'snap_nums':
        snap_nums = np.linspace(100, 0, 101)
        return snap_nums, final_masses, main_progenitor_list, major_mergers, major_merger_times, LMMs, LMM_times, fossil_clusters
    
###############################################################################################################
# Track Evolution: multiple halos                                                                             #
# build a main progenitor branch for EACH halo in the index;                                                  #
# return LISTS of masses, time steps, main progenitor id's, and locations of major mergers for each mp branch #
###############################################################################################################

def track_evol_multiple(idx, redshifts, prog_idx, forest_tbl, mm_thresh_small = 0.1, mm_thresh_big = 0.3, normalized = False, x_axis = 'z_nums'):
    print("In track_evol_multiple")
    all_timesteps= []
    all_masses = []
    all_main_prog_list = []
    all_major_mergers = []
    all_major_merger_times = []
    #all_major_mergers = [np.empty(0), np.empty(0)]
    #all_major_merger_times = [np.empty(0), np.empty(0)]
    all_LMMs = [np.empty(0), np.empty(0)]
    all_LMM_times = [np.empty(0), np.empty(0)]
    all_fossil_clusters = [np.empty(0), np.empty(0)]

    # Loop over each halo
    for i in range(len(idx)):
        timesteps, masses, main_progenitor_list, major_mergers, major_merger_times, LMMs, LMM_times, fossil_clusters = track_evol(idx[i], redshifts, prog_idx, forest_tbl, mm_thresh_small, mm_thresh_big, normalized, x_axis)
        
        all_timesteps.append(timesteps)
        all_masses.append(masses)
        all_main_prog_list.append(main_progenitor_list)
        
        if len(major_mergers[0]) == 0:
            all_major_mergers.append(major_mergers)
            all_major_merger_times.append(major_merger_times)
            
            if len(LMMs[0]) != 0: # vs. LMMs[0] != []?
                all_LMMs[0] = np.append(all_LMMs[0], LMMs[0])
                all_LMM_times[0] = np.append(all_LMM_times[0], LMM_times[0])
            if len(LMMs[1]) != 0:# vs. LMMs[0] != []?
                all_LMMs[1] = np.append(all_LMMs[1], LMMs[1])
                all_LMM_times[1] = np.append(all_LMM_times[1], LMM_times[1])
                
            #if len(fossil_clusters[0]) != 0 or len(fossil_clusters[1] != 0): # If either dim contains anything
            if len(fossil_clusters[0]) != 0:
                all_fossil_clusters[0] = np.append(all_fossil_clusters[0], fossil_clusters[0])
            elif len(fossil_clusters[1]) != 0:
                all_fossil_clusters[1] = np.append(all_fossil_clusters[1], fossil_clusters[1])
                    
    #print("all_fossil_clusters: ", all_fossil_clusters)
    return all_timesteps, all_masses, all_main_prog_list, all_major_mergers, all_major_merger_times, all_LMMs, all_LMM_times, all_fossil_clusters

######################################################################################################################
# Track Evolution: binned halos                                                                                      #
# index of halos is binned -- build main progenitor branches for each halo in each bin;                              #
# return BINNED lists of masses, time steps, main progenitor id's, and locations of major mergers for each mp branch #
######################################################################################################################

def track_evol_binned(idx, bins, redshifts, prog_idx, forest_tbl, mm_thresh_small = 0.1, mm_thresh_big = 0.3, normalized = False, x_axis = 'z_nums'):
    all_timesteps= []
    all_masses = []
    all_main_prog_list = []
    all_major_mergers = []
    all_major_merger_times = []
    #all_LMMs = []
    #all_LMM_times = []
    all_LMMs = [[np.empty(0) for thresh in range(2)] for i in range(len(bins) - 1)]
    all_LMM_times = [[np.empty(0) for thresh in range(2)] for i in range(len(bins) - 1)]
    all_fossil_clusters = [[np.empty(0) for i in range(len(bins) - 1)] for thresh in range(2)] # 2 thresh, # bins

    # Loop over each bin
    for i in range(len(bins) - 1):
        timesteps, masses, main_progenitor_list, major_mergers, major_merger_times, LMMs, LMM_times, fossil_clusters = track_evol_multiple(idx[i], redshifts, prog_idx, forest_tbl, mm_thresh_small, mm_thresh_big, normalized, x_axis)
        if len(fossil_clusters[0]) == 0:
            print("Ack")
        
        all_timesteps.append(timesteps)
        all_masses.append(masses)
        all_main_prog_list.append(main_progenitor_list)

        #if major_mergers[0] != []: # Check len, not if empty # Do something totally different
        all_major_mergers.append(major_mergers)
        all_major_merger_times.append(major_merger_times)

        if len(LMMs[0]) != 0: # vs. LMMs[0] != []?
            all_LMMs[i][0] = np.append(all_LMMs[i][0], LMMs[0])
            all_LMM_times[i][0] = np.append(all_LMM_times[i][0], LMM_times[0])
        if len(LMMs[1]) != 0:# vs. LMMs[0] != []?
            all_LMMs[i][1] = np.append(all_LMMs[i][1], LMMs[1])
            all_LMM_times[i][1] = np.append(all_LMM_times[i][1], LMM_times[1])

        if len(fossil_clusters[0]) != 0:
            all_fossil_clusters[0][i] = np.append(all_fossil_clusters[0][i], fossil_clusters[0])
        if len(fossil_clusters[1]) != 0:
            all_fossil_clusters[1][i] = np.append(all_fossil_clusters[1][i], fossil_clusters[1])
            
    #print("len of all_major_mergers inside track_evol_binned is: ", len(all_major_mergers))
    return all_timesteps, all_masses, all_main_prog_list, all_major_mergers, all_major_merger_times, all_LMMs, all_LMM_times, all_fossil_clusters
    
###############################################################################
# Average Bins:                                                               #
# track the evolution of halos in bins, return the average masses of each bin #
###############################################################################

def avg_bins(idx, bins, redshifts, prog_idx, forest_tbl, mm_thresh_small = 0.1, mm_thresh_big = 0.3, normalized = False, x_axis = 'z_nums'):
    #Note that here, idx comes from bin_halos, so it's a 2D matrix (ie. five bins with some # halos per bin)
    final_masses = []
    final_timesteps = []
    
    # Track the evolution of halos in bins
    timesteps, masses, main_prog_list, maj_mergers, mm_times, LMMs, LMM_times, fossil_clusters = track_evol_binned(idx, bins, redshifts, prog_idx, forest_tbl, mm_thresh_small, mm_thresh_big, normalized, x_axis)
    
    # Take the average over all the masses in each bin
    for i in np.arange(len(bins) - 1): # Same as len(masses)
        avg_masses = np.average(masses[i], axis = 0) # do I still need to specify axis = 0?
        final_masses.append(avg_masses)
        
    final_timesteps = [redshifts for j in range(len(timesteps))]
        
    return final_timesteps, final_masses

########################################################
# Plot evolution:                                      #        
# Display M(z) for halos that we tracked in track_evol #
########################################################
    
def plot_evol(timesteps, masses, forest_tbl, mm_times = [[], []], filename = "new_plot", bins = [], avg = False, normalized = False, extremum = '', quant = 0, mass_range = [], x_axis = "z_nums"):
    
    fig, ax = plt.subplots()
    color = iter(cm.jet(np.linspace(0,1,len(masses)))) # Bin colors
    
    # Change zeros to nans so they don't get plotted
    for m in range(len(masses)):
        masses[m][masses[m] == 0] = np.nan
    
    # Pick your plot style
    if avg == True:
        for n in range(len(masses)): # loop over all bins
            ax.plot(timesteps[n], masses[n], color = next(color), label = ("bin " + str(n + 1) + ": (" + "{:.2e}".format(bins[n]) + " to " + "{:.2e}".format(bins[n+1]) + ")"))   
        ax.legend()
        
        if normalized == True:
            ax.set_title("Normalized averaged mass evolution of halos in " + str(int(len(bins) - 1)) + " bins")
        elif normalized == False:
            ax.set_title("Averaged mass evolution of halos in " + str(int(len(bins) - 1)) + " bins") # Why are there only 9?

    elif avg == False:
        for p in range(len(masses)): # One for each bin
            ax.plot(timesteps[p], masses[p], color = next(color))
            
        if extremum == 'max':
            ax.set_title("Mass evolution of " + str(quant) + " most massive halos")
        elif extremum == 'min':
            ax.set_title("Mass evolution of " + str(quant) + " least massive halos")
        elif extremum == '':
            if mass_range != []:
                ax.set_title("Mass evolution of halos in range " + "{:.2e}".format(mass_range[0]) + " to " + "{:.2e}".format(mass_range[1]))
            else:
                ax.set_title("Mass evolution of halo(s)")
    
    # Display major mergers (if desired)
    # Only works for non_binned results, really...
    if mm_times != [[]]: # Does this handle both positions? Or only one, really?
        color2 = iter(cm.jet(np.linspace(0,1,len(mm_times)))) # Bin colors again
        
        # Loop over each halo
        for this_halo in range(len(mm_times)):
            current_color = next(color2)
            #print("just changed color, halo num is: ", this_halo) # Remember the problem this represents
            linestyles = iter([':', '--'])
            
            # Loop over the two threshholds
            for this_thresh in range(len(mm_times[this_halo])):
                this_linestyle = next(linestyles)
                
                # Loop over each major merger (for this halo)
                for this_mm in range(len(mm_times[this_halo][this_thresh])):
                    
                    # Why do I need these "ints" in here?
                    if x_axis == "z_nums":
                        merg = redshifts[int(100 - mm_times[this_halo][this_thresh][this_mm])]
                    elif x_axis == "snap_nums":
                        merg = mm_times[this_halo][this_thresh][this_mm]
                    ax.axvline(merg, color = current_color, linestyle = this_linestyle)
                
                # Old way, before I had mm_times
                # Loop over each major merger (for this halo)
                #for s in np.arange(len(major_mergers[q][r])):
                    #if x_axis == "z_nums":
                    #    merg = redshifts[100 - forest_tbl['snap_num'][major_mergers[q][r][s]]]
                    #elif x_axis == "snap_nums":
                    #    merg = forest_tbl['snap_num'][major_mergers[q][r][s]]
                    #ax.axvline(merg, color = current_color, linestyle = this_linestyle)

    if x_axis == "z_nums":
        ax.set_xlim(10.044, 0)
        ax.set_xlabel("Redshift [z]")
    elif x_axis == "snap_nums":
        ax.set_xlabel("Snapnumber (SN = 99 -> z = 0)")
    ax.set_yscale('log', nonpositive = 'clip')
    ax.set_ylabel('Mass [M_sun/h]')
    plt.savefig(filename + ".png")
    plt.show()
    
    
###############################################################
# Find Last Major Merger:                                     #
# return halo id's of major mergers and associated time steps #
###############################################################

def find_LMMs(major_mergers, major_merger_times, forest_tbl, x_axis = 'snap_nums', redshifts = redshifts):
    
    # Mask up! (For second threshhold only)
    mask = [len(major_merger_times[i][1]) != 0 for i in range(len(major_merger_times))]
    #mask = []
    #for i in range(len(major_merger_times)):
    #    print("i is: ", i)
    #    mask.append(len(major_merger_times[i][1]) != 0)
    masked_mergers = np.array(major_mergers, dtype = object)[mask]
    masked_merger_times = np.array(major_merger_times, dtype = object)[mask]
    
    # Make lists of LMM (one for each threshhold)
    # Major mergers[i] is already ordered from most recent to most distantly in the past, so just peel off the first entry
    LMM_list_small = [major_mergers[i][0][0] for i in np.arange(len(major_mergers))]
    LMM_list_big = [masked_mergers[i][1][0] for i in np.arange(len(masked_mergers))]
    
    LMM_times_small = [major_merger_times[i][0][0] for i in np.arange(len(major_merger_times))]
    LMM_times_big = [masked_merger_times[i][1][0] for i in np.arange(len(masked_merger_times))]
    
    # Find the time steps associated with the last major mergers
    #if x_axis == 'z_nums':
    #    LMM_times_small = np.array(redshifts[99 - forest_tbl['snap_num'][LMM_list_small]])
    #    LMM_times_big = np.array(redshifts[99 - forest_tbl['snap_num'][LMM_list_big]])
    #elif x_axis == 'snap_nums':  
    #    LMM_times_small = np.array(forest_tbl['snap_num'][LMM_list_small])
    #    LMM_times_big = np.array(forest_tbl['snap_num'][LMM_list_big])
     
    LMM_list = [LMM_list_small, LMM_list_big]
    LMM_times = [LMM_times_small, LMM_times_big]
    
    return LMM_list, LMM_times

######################################################################################
# Find Last Major Merger for Bins:                                                   #
# return list of binned halo id's of major mergers, and associated binned time steps #
######################################################################################

def find_LMMs_binned(binned_major_mergers, binned_major_merger_times, forest_tbl, x_axis = 'snap_nums', redshifts = redshifts):
    
    all_LMM_list = []
    all_LMM_times = []
    for mms, mm_times in zip(binned_major_mergers, binned_major_merger_times):
        this_LMM_list, this_LMM_times = find_LMMs(mms, mm_times, forest_tbl, x_axis, redshifts)
        all_LMM_list.append(this_LMM_list)
        all_LMM_times.append(this_LMM_times)
    
    return all_LMM_list, all_LMM_times

####################################################
# Plot Last Major Mergers:                         #
# PDF of last major mergers at different timesteps #
####################################################

def plot_LMMs(LMM_times, mass_bins = [], mass_range = [], x_axis = 'z_nums'): # Note: assume LMM_times are binned
    # How can I make this more efficient?
    
    # Fix the dims of LMM_times
    LMM_times_small_thresh = [[LMM_times[i][j][0] for j in range(len(LMM_times[i]))] for i in range(len(LMM_times))]
    LMM_times_big_thresh = [[LMM_times[i][j][1] for j in range(len(LMM_times[i])) if len(LMM_times[i][j][1]) != 0] for i in range(len(LMM_times))]
    re_LMM_times = [[LMM_times_small_thresh[i], LMM_times_big_thresh[i]] for i in range(len(LMM_times))]
    
    # Change everything to redshifts
    LMM_z_times = [[[redshifts[int(100 - re_LMM_times[this_bin][this_thresh][this_mm])] for this_mm in range(len(re_LMM_times[this_bin][this_thresh]))] for this_thresh in range(len(re_LMM_times[this_bin]))] for this_bin in range(len(re_LMM_times))]
    
    # Calculate histogram of LMM_times, plot as a distribution
    fig, ax = plt.subplots()
    color = iter(cm.jet(np.linspace(0,1,len(LMM_z_times))))
    
    # Plot once for each bin
    for i in np.arange(len(LMM_z_times)):
        current_color = next(color)
        
        # Plot once for each threshold
        for j in np.arange(len(LMM_z_times[i])): # length is 2
            norm_factor = len(LMM_z_times[i][j])
            hist = np.histogram(LMM_z_times[i][j])
            bin_centers = (hist[1][:-1] + hist[1][1:]) / 2
            
            if j == 0:
                ax.plot(bin_centers, hist[0]/norm_factor, linestyle = '--', color = current_color)
            elif j == 1:
                if mass_bins != []:
                    ax.plot(bin_centers, hist[0]/norm_factor, linestyle = '-', color = current_color, label = ("bin " + str(i + 1) + ": (" + "{:.2e}".format(mass_bins[i]) + " to " + "{:.2e}".format(mass_bins[i+1]) + ")"))
                else:
                    ax.plot(bin_centers, hist[0]/norm_factor, linestyle = '-', color = current_color)
    
    # Set customized labels and titles
    if x_axis == 'z_nums':
        ax.set_xlabel("Redshift of LMM")
    elif x_axis == 'snap_nums':
        ax.set_xlabel("snapnum of LMM")
    if mass_range != []:
        ax.set_title("PDF of Last Major Mergers in range: " + "{:.1e}".format(mass_range[0]) + " to " + "{:.1e}".format(mass_range[1]))
    elif mass_range == []:
        ax.set_title("PDF of Last Major Mergers (binned)")
    ax.set_ylabel("Probability")
    ax.legend(loc = 'upper right')
    
    # Fancy legend
    leg1 = ax.legend(loc='upper right')
    # Add second legend for the maxes and mins.
    # leg1 will be removed from figure
    dashed_line = Line2D([0,1],[0,1],linestyle='--', color="black")
    solid_line = Line2D([0,1],[0,1],linestyle='-', color="black")
    eg2 = ax.legend((dashed_line, solid_line), ('ε > 0.1', 'ε > 0.3'), loc = 'center right')
    # Manually add the first legend back
    ax.add_artist(leg1)
    
    plt.savefig("pdf_lmms.png")
    plt.show()

################################################################################################
# Plot Cumulative Distribution Function:                                                       #
# Plot the probability that some data (X) will take on a value less than/equal to quantity (x) #
################################################################################################

def plot_CDF(data_raw, comparison_data, bins = [], redshifts = redshifts, x_axis = 'z_nums'): # data = LMM_times, comp_data = binned_masses (because that gives us the total number of trees, including those without major mergers)
    
    # Change everything to redshifts
    data = [[[redshifts[int(100 - data_raw[this_bin][this_thresh][this_mm])] for this_mm in range(len(data_raw[this_bin][this_thresh]))] for this_thresh in range(len(data_raw[this_bin]))] for this_bin in range(len(data_raw))]
    
    fig, ax = plt.subplots()
    color = iter(cm.jet(np.linspace(0,1,len(data))))
    #hist_bins = np.concatenate([redshifts[0]], [(redshifts[i] + redshifts[i + 1]) / 2 for i in np.arange(len(redshifts) - 1)], redshifts[-1])
    
    # Loop over each bin
    for i in range(len(data)):
        current_color = next(color)
        print("\nfor bin: ", i, " # of halos is: ", len(comparison_data[i]))
        
        # Loop over each threshold
        for j in range(len(data[i])):
            
            data_sorted = np.sort(data[i][j]) # Should automatically sort along the last axis
            #print("for j: ", j, " data_sorted is: ", data_sorted)
            hist_keys = [key for key, group in groupby(data_sorted)] # Redshift values
            print("for j: ", j, " hist_keys is: ", hist_keys)
            hist_values = [len(list(group)) for key, group in groupby(data_sorted)] # Count of each redshift value
            print("for j ", j,  " # halos with mergers is... ", sum(hist_values)) 
            print("hist values is: ", hist_values)
            #cum_probs = [sum(hist_values[0:k]) / len(comparison_data[i]) for k in range(len(hist_values))]
            cum_probs = np.cumsum(hist_values) / len(comparison_data[i])
            print("cum probs is: ", cum_probs)
            #for l in range(len(hist_values)):
            #    print("hist_values[", 0, ":", l, "]: ", hist_values[0:l])
            #    print("cum_probs[", l, "]: ", cum_probs[l])
            if j == 0:
                ax.plot(hist_keys, cum_probs, color = current_color, linestyle = '--')
            elif j == 1:
                ax.plot(hist_keys, cum_probs, color = current_color, linestyle = '-', label = ("bin " + str(i + 1) + ": (" + "{:.2e}".format(bins[i]) + " to " + "{:.2e}".format(bins[i+1]) + ")"))
    
    if x_axis == 'z_nums':
        ax.set_xlabel("Redshift of LMM")
    elif x_axis == 'snap_nums':
        ax.set_xlabel("Snapnum of LMM")
        
    ax.set_ylabel("Probability")
    ax.set_title("Cumulative Probability Distribution of LMMs")
    ax.set_xscale("symlog", linthresh = 1, linscale = 0.4)
    
    # Unnecessarily Complicated Tick Marks
    stepsize = 1
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end, stepsize))
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    
    # Unnecessarily Complicated Legend
    leg1 = ax.legend(loc='lower right')
    dashed_line = Line2D([0,1],[0,1],linestyle='--', color="black")
    solid_line = Line2D([0,1],[0,1],linestyle='-', color="black")
    leg2 = ax.legend((dashed_line, solid_line), ('ε > 0.1', 'ε > 0.3'), loc = 'center right')
    # Manually add the first legend back
    ax.add_artist(leg1)
    plt.savefig("cdf_large_range.png")
    plt.show()

#######################################################################################
# Calculate Mass Growth Rate:                                                         #
# Find alphas (Srisawat eq. 7) for each halo in halo_idx, for each pair of time steps #
#######################################################################################

def calc_mass_growth_rate(timesteps, masses, main_prog_list, prog_idx, forest_tbl):    
    alpha_list = []
    
    # Change z_values to lookback times
    cosmo = FlatLambdaCDM(H0=67.66, Om0=0.310)
    lookback_times = np.array(cosmo.lookback_time(timesteps))
    age = float(cosmo.age(0) / u.Gyr)
    ages_list = [age for i in range(len(lookback_times[0]))]
    times = [ages_list - lt for lt in lookback_times]
    
    # Loop through each halo
    for i in range(len(masses)):
        
        # Mask up, friends!
        mask = (masses[i] > 10**12)
        masses_masked = masses[i][mask]
        times_masked = times[i][mask]
        main_prog_list[i] = np.append(main_prog_list[i], np.zeros(101 - len(main_prog_list[i])))[mask]
        
        # Calculate alphas
        t_B = times_masked[:-1]
        M_B = masses_masked[:-1]
        t_A = times_masked[1:]
        M_A = masses_masked[1:]
        alpha = (t_B + t_A) * (M_B - M_A) / ((t_B - t_A)*(M_B + M_A))
        alpha_std = alpha[np.isfinite(alpha)]
        #alpha_std = alpha[np.isnan(alpha) == False]
        alpha_list.append(alpha_std)

    return alpha_list

#########################################################
# Calculate mass growth rate for binned lists of halos: #
# Run calc_mass_growth_rate once for each bin;          #
# return a binned list of alphas                        #  
#########################################################

def calc_BINNED_mass_growth_rate(timesteps, masses, main_prog_list, prog_idx, forest_tbl):

    total_alpha_list = [calc_mass_growth_rate(timesteps[i], masses[i], main_prog_list[i], prog_idx, forest_tbl) for i in np.arange(len(timesteps))]
    #print("len(total_alpha_list): ", len(total_alpha_list))
    #print("len(total_alpha_list[0]): ", len(total_alpha_list[0]))
    #print("len(total_alpha_list[0][0]): ", len(total_alpha_list[0][0]))
    return total_alpha_list

################################################################
# Plot Distribution:                                           #
# Display PDF distribution of some metric using np.histogram() #
################################################################

def plot_distrib(values, metric_name, values_name, bins = [], zoom = False, n_hist_bins = 10, log = False): # For mass growth rates, "values" is list of alphas 
    # Note: assume values is binned
    fig, ax = plt.subplots()
    color = iter(cm.jet(np.linspace(0,1,len(values))))
    all_values = []

    # For all bins
    for h in range(len(values)):
        current_color = next(color)
        
        # For all halo trees
        this_halo_values = np.concatenate([values[h][i] for i in range(len(values[h]))])
        
        #this_halo_values = []
        #for i in range(len(values[h])):
        #    this_halo_values = np.append(this_halo_values, values[h][i])
            #print("len(this_halo_values): ", len(this_halo_values))
        
        all_values.append(this_halo_values)

        if zoom == True:
            values_in_range = np.array([all_values[h][i] for i in range(len(all_values[h])) if (all_values[h][i] < 10 and all_values[h][i] > -10)])
            hist = np.histogram(values_in_range, bins = n_hist_bins)
        else:
            hist = np.histogram(all_values[h], bins = n_hist_bins)

        bin_centers = ((hist[1][:-1] + hist[1][1:]) / 2)
        
        if bins == []:
            ax.plot(bin_centers, hist[0]+1, color = current_color)
            binned_tf = []
        else:
            ax.plot(bin_centers, hist[0]+1, color = current_color,  label = ("bin " + str(h + 1) + ": (" + "{:.2e}".format(bins[h]) + " to " + "{:.2e}".format(bins[h+1]) + ")"))

    # Accesorize the plots
    if log == True:
        ax.set_xscale('symlog')
    
    name = "Distribution of " + metric_name
    if bins != []:
        name = "Binned " + name
    
    # Unnecessarily Complicated Tick Marks
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    
    ax.set_title(name)
    ax.set_yscale('log')
    ax.set_xlabel(values_name)
    ax.set_ylabel("N + 1")
    ax.legend()
    plt.savefig(name + ".png")
    plt.show()    
    
##################################################
# Plot Length of Main Branch:                    #
# Find main branch length from mp list, plot PDF #
##################################################

def plot_main_branch_length(mp_list, n_bins = 32, hist_bins = [], zoom = False, log = True, dist_or_hist = 'dist'):
    # Note: assume mp_list is binned
    fig, ax = plt.subplots()
    color=iter(cm.jet(np.linspace(0,1,len(mp_list))))
    my_bins = [i*(101/n_bins) for i in np.arange(n_bins + 1)]
    
    # Loop over each bin
    for i in np.arange(len(mp_list)):
        current_color = next(color)
        mp_lengths = []
        
        # Loop over each halo root
        for j in np.arange(len(mp_list[i])):
            mp_lengths.append(len(mp_list[i][j]))
            
        if dist_or_hist == 'dist':
            hist = np.histogram(mp_lengths, bins = my_bins)
            bin_centers = ((hist[1][:-1] + hist[1][1:]) / 2)

            if hist_bins != []:
                ax.plot(bin_centers, hist[0]+1, color = current_color, label = ("bin " + str(i + 1) + ": (" + "{:.2e}".format(hist_bins[i]) + " to " + "{:.2e}".format(hist_bins[i+1]) + ")"))
            else:
                ax.plot(bin_centers, hist[0]+1, color = current_color)

        elif dist_or_hist == 'hist':
            ax.hist(mp_lengths, bins = n_bins)
            
    if log == True:
        ax.set_yscale('log')
    ax.set_xlabel("Lengths")
    ax.set_ylabel("N + 1")
    ax.set_title("Distribution of Main Branch Lengths")
    
    # Mess around with legend
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.04,0.5))
    text = ax.text(-0.2,1.05, "Count", transform=ax.transAxes)
    plt.savefig("main_branch_lengths_new.png", bbox_extra_artists=(lgd, text), bbox_inches='tight')
    plt.show()
    
#################################################################
# Calculate Cumulative Number of Major Mergers: (just one halo) #
# Find redshifts associated with each major merger;             #
# Count and average number of major mergers at each redshift    #
#################################################################

def calc_cum_maj_mergers(major_merger_times, redshifts, forest_tbl): # Note: assume major_mergers is for just one halo
    
    # For all redshifts z, count # of mms that took place between 0 and z
    mask = [[[(major_merger_times[i][j] <= z) for j in range(len(major_merger_times[i]))] for z in redshifts] for i in range(len(major_merger_times))]
    cum_mms = [[mask[i][k].count(True) for k in range(len(mask[i]))] for i in range(len(major_merger_times))] # mask[k] is a list, count the True values in the list
    
    return cum_mms # dim: 2 thresholds, 101 z's, some # MMs (T/F values)
    
###########################################################################
# Average Cumulative Number of Major Mergers: (multiple halos)            #
# Run calc_cum_maj_mergers once for each halo, find mean for each z value #
###########################################################################

def avg_cum_maj_mergers(major_merger_times, redshifts, forest_tbl): # Note: assume major_mergers contains multiple halos
    
    cum_mms = [calc_cum_maj_mergers(major_merger_times[i], redshifts, forest_tbl) for i in range(len(major_merger_times))] # One for each halo # Is this (list comprehension + a function call) actually faster than a for loop?
    avg = [np.average([cum_mms[i][j] for i in np.arange(len(cum_mms))], axis = 0) for j in range(len(cum_mms[0]))] # Kind of cheating... hard-coded to get the right length for the loop (2)
    #print("avg is: ", avg)
    return avg

    # For comparison
    #all_avg = []
    #for j in [0,1]:
    #    rearrange = [cum_mms[i][j] for i in np.arange(len(cum_mms))]
    #    avg1 = np.average(rearrange, axis = 0)
    #    print("avg1: ", avg1)
    #    all_avg.append(avg1)
    
#################################################################
# Binned Avg Cumulative Number of Major Mergers: (binned halos) #
# Run avg_cum_maj_mergers once for each bin                     #
#################################################################

def binned_avg_cum_maj_mergers(major_merger_times, redshifts, forest_tbl): # Note: assume major_mergers is binned
    
    binned_avg = [avg_cum_maj_mergers(major_merger_times[i], redshifts, forest_tbl) for i in range(len(major_merger_times))]
    return binned_avg

##############################################################################
# Plot Mean Number of Major Mergers (Fakhouri & Ma, 2011, Fig. 7):           #
# Mean # of major mergers experienced (by a halo at z0 = 0) between z0 and z #
##############################################################################

def plot_cum_mms(binned_averages, bins, redshifts):
    fig, ax = plt.subplots()
    color = iter(cm.jet(np.linspace(0,1,len(binned_averages))))
    
    for i in np.arange(len(binned_averages)): # Number of bins
        current_color = next(color)
        
        for j in np.arange(len(binned_averages[i])): # 2 thresholds
            if j == 0:
                ax.plot(redshifts, binned_averages[i][j], color = current_color, linestyle = '--')
            elif j == 1:
                ax.plot(redshifts, binned_averages[i][j], color = current_color, linestyle = '-', label = ("bin " + str(i + 1) + ": (" + "{:.2e}".format(bins[i]) + " to " + "{:.2e}".format(bins[i+1]) + ")"))
    
    # Unnecessarily Complicated Legend
    leg1 = ax.legend(loc='lower right')
    dashed_line = Line2D([0,1],[0,1],linestyle='--', color="black")
    solid_line = Line2D([0,1],[0,1],linestyle='-', color="black")
    leg2 = ax.legend((dashed_line, solid_line), ('ε > 0.1', 'ε > 0.3'), loc = 'center right')
    ax.add_artist(leg1)
        
    # More accessories
    ax.set_xscale("symlog", linthresh = 1, linscale = 0.4)
    ax.set_xlabel("Redshift (z)")
    ax.set_yscale('log')
    ax.set_ylabel("Mean # Mergers between z0 and z")
    ax.set_title("Mean Number of Major Mergers between z0 and z")
    
    # Unnecessarily Complicated Tick Marks
    stepsize = 1
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end, stepsize))
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    
    # Finish
    plt.savefig("mean_num_mergers.png")
    plt.show()
    
###########################
# Find halos with LMM > 1 #
###########################

def find_large_LMM_halos(binned_LMM_list, binned_LMM_times, forest_tbl):
    
    # Find LMM_list entries for which their corresponding LMM_times has z > 1 (aka for which Mask = True)
    masked_list = [[[binned_LMM_list[i][j][k] for k in range(len(binned_LMM_list[i][j])) if binned_LMM_times[i][j][k] > 1] for j in range(len(binned_LMM_list[i]))] for i in range(len(binned_LMM_list))]
    
    # Extract the core_ids of those halos
    core_ids = [[[forest_tbl['core_id'][halo_id] for halo_id in this_thresh] for this_thresh in this_bin] for this_bin in masked_list]
    
    # Find the "root" (the latest halo_id with the same core_id)    
    # Only save the roots which are between snapshots 90 and 99
    roots = [[[max(np.array(forest_tbl.halo_id.loc[forest_tbl.core_id == core])) for core in core_ids[i][j] if np.array(forest_tbl.snap_num.loc[forest_tbl.halo_id == max(np.array(forest_tbl.halo_id.loc[forest_tbl.core_id == core]))]) > 90] for j in range(len(core_ids[i]))] for i in range(len(core_ids))]
    
    # Save the halo_id's of the roots in an index (one for each threshold)
    roots_idx = [[roots[i][j] for i in range(len(roots))] for j in range(2)] # Is there a way to do this without hard-coding?

    # Bonus!
    count = sum([sum([len(roots_idx[i][j]) for j in range(len(roots_idx[i]))]) for i in range(len(roots_idx))]) # Total # of halos in this category
    
    return roots_idx, core_ids

# *********************
# **  Old functions  **
# *********************

def OLD_track_evol(idx, redshifts, prog_idx, forest_tbl, mm_thresh_small = 0.1, mm_thresh_big = 0.3, normalized = False, x_axis = 'z_nums'): # Just track the evolution of one halo -- bells and whistles come later
    
    target_idx = idx  # np.int(np.array(idx))
    progenitors = prog_idx[target_idx]
    main_progenitor_list = [target_idx]
    major_mergers = [np.empty(0), np.empty(0)] # Do these have to be np arrays?
    major_merger_times = [np.empty(0), np.empty(0)]

    # Generate main branch: find all the progenitors of the halo (at each timestep)
    while len(progenitors) > 0:
        masses = [forest_tbl['mass'][i] for i in progenitors] # Find masses for each progenitor of the halo at this time step
        main_progenitor = progenitors[np.argmax(masses)]
        main_progenitor_list.append(main_progenitor)

        # Count major mergers
        if len(progenitors) > 1:
            order_by_mass = np.argsort(masses)[::-1] # Special syntax for reversing order
            sorted_progenitors = np.array(progenitors)[order_by_mass]
            sorted_masses = np.array(masses)[order_by_mass]

            if sorted_masses[0]*(mm_thresh_small) < sorted_masses[1]:
        # sorted_progenitors[0] vs sorted_progenitors[1]
                major_mergers[0] = np.append(major_mergers[0], sorted_progenitors[0])
                major_merger_times[0] = np.append(major_merger_times[0], redshifts[99 - forest_tbl['snap_num'][sorted_progenitors[0]]])
                
                if sorted_masses[0]*(mm_thresh_big) < sorted_masses[1]:
                    major_mergers[1] = np.append(major_mergers[1], sorted_progenitors[0]) # This just said progenitors before? Is that right?
                    major_merger_times[1] = np.append(major_merger_times[1], redshifts[99 - forest_tbl['snap_num'][sorted_progenitors[0]]])
            
        # Continue
        progenitors = prog_idx[main_progenitor]

    # Also, do these have to be np arrays?
    raw_masses = [forest_tbl['mass'][mp] for mp in main_progenitor_list]  # Mass of each main progenitor for this halo 
    final_masses = np.append(raw_masses, np.zeros(101 - len(raw_masses)))*norm  # Standardize mass array with length 101 and un-normalized mass

    # Normalize by final mass, if desired
    if normalized == True:
        final_masses = final_masses / np.array(forest_tbl['mass'][target_idx]) #masses_std[len(masses_std) - 1]
    
    # Return all the things you need!
    if x_axis == 'z_nums':
        z_nums = redshifts
        return z_nums, final_masses, main_progenitor_list, major_mergers, major_merger_times
    elif x_axis == 'snap_nums':
        snap_nums = np.linspace(99, -1, 101)
        return snap_nums, final_masses, main_progenitor_list, major_mergers, major_merger_times

def OLD_find_large_LMM_halos(binned_LMM_list, binned_LMM_times, forest_tbl):
    
    # Make a mask: find LMM_times for which z > 1
    #mask = [[[(z > 1) for z in this_thresh] for this_thresh in this_bin] for this_bin in binned_LMM_times]
    #masked_list = [[[binned_LMM_list[i][j][k] for k in range(len(binned_LMM_list[i][j])) if mask[i][j][k] == True] for j in range(len(binned_LMM_list[i]))] for i in range(len(binned_LMM_list))]
    
    # Find LMM_list entries for which their corresponding LMM_times has z > 1 (aka for which Mask = True)
    masked_list = [[[binned_LMM_list[i][j][k] for k in range(len(binned_LMM_list[i][j])) if binned_LMM_times[i][j][k] > 1] for j in range(len(binned_LMM_list[i]))] for i in range(len(binned_LMM_list))]
    
    # Extract the core_ids of those halos
    core_ids = [[[forest_tbl['core_id'][halo_id] for halo_id in this_thresh] for this_thresh in this_bin] for this_bin in masked_list]
    
    # Find the "root" (the latest halo_id with the same core_id)
    #roots = [[[max(np.array(forest_tbl.halo_id.loc[forest_tbl.core_id == core])) for core in core_ids[i][j]] for j in range(len(core_ids[i]))] for i in range(len(core_ids))]
    
    # Only save the roots which are between snapshots 90 and 99
    roots = [[[max(np.array(forest_tbl.halo_id.loc[forest_tbl.core_id == core])) for core in core_ids[i][j] if np.array(forest_tbl.snap_num.loc[forest_tbl.halo_id == max(np.array(forest_tbl.halo_id.loc[forest_tbl.core_id == core]))]) > 90] for j in range(len(core_ids[i]))] for i in range(len(core_ids))]
    
    # Save the halo_id's of the roots in an index (one for each threshold)
    roots_idx = [[roots[i][j] for i in range(len(roots))] for j in range(2)] # Is there a way to do this without hard-coding?

    # Bonus!
    count = sum([sum([len(roots_idx[i][j]) for j in range(len(roots_idx[i]))]) for i in range(len(roots_idx))]) # Total # of halos in this category
    
    return roots_idx

def OLD_find_LMMs(major_mergers, forest_tbl, x_axis = 'snap_nums', redshifts = redshifts):
    
    # Mask up! (For second threshhold only)
    mask = [len(major_mergers[i][1]) != 0 for i in np.arange(len(major_mergers))]
    masked_mergers = np.array(major_mergers, dtype = object)[mask]
    
    # Make lists of LMM (one for each threshhold)
    # Major mergers[i] is already ordered from most recent to most distantly in the past, so just peel off the first entry
    LMM_list_small = [major_mergers[i][0][0] for i in np.arange(len(major_mergers))]
    LMM_list_big = [masked_mergers[i][1][0] for i in np.arange(len(masked_mergers))]
    
    # Find the time steps associated with the last major mergers
    if x_axis == 'z_nums':
        LMM_times_small = np.array(redshifts[99 - forest_tbl['snap_num'][LMM_list_small]])
        LMM_times_big = np.array(redshifts[99 - forest_tbl['snap_num'][LMM_list_big]])
    elif x_axis == 'snap_nums':  
        LMM_times_small = np.array(forest_tbl['snap_num'][LMM_list_small])
        LMM_times_big = np.array(forest_tbl['snap_num'][LMM_list_big])
     
    LMM_list = [LMM_list_small, LMM_list_big]
    LMM_times = [LMM_times_small, LMM_times_big]
    
    return LMM_list, LMM_times

def OLD_plot_main_branch_length(mp_list, n_bins = 50, hist_bins = [], zoom = False, log = True, dist_or_hist = 'dist'):
    # Note: assume mp_list is binned
    fig, ax = plt.subplots()
    color=iter(cm.jet(np.linspace(0,1,len(mp_list))))
    
    # Loop over each bin
    for i in np.arange(len(mp_list)):
        current_color = next(color)
        mp_lengths = []
        
        # Loop over each halo root
        for j in np.arange(len(mp_list[i])):
            mp_lengths.append(len(mp_list[i][j]))
        print("len(mp_lengths): ", len(mp_lengths))
        if dist_or_hist == 'dist':
            print("current i is: ", i)
            min_length = np.min(mp_lengths)
            max_length = np.max(mp_lengths)
            # Changes are a-comin'!
            # Want all the same n_bins (~32)
            # So all empty values must be replaced with zeros so the whole array is 101 entries long
            n_bins = max_length - min_length # Should this have a + 1, or not? I think not...
            hist = np.histogram(mp_lengths, bins = n_bins)
            
            #pre_gap = np.arange(hist[1][0] - 1)
            #post_gap = np.arange(hist[1][-1] + 1, 101 + 1)
            #std_hist_keys = np.concatenate((pre_gap, hist[1], post_gap), axis = 0)
            #std_hist_values = np.concatenate((np.zeros(len(pre_gap)), hist[0], np.zeros(len(post_gap))), axis = 0)
            #bin_centers = ((std_hist_keys[:-1] + std_hist_keys[1:]) / 2)
            bin_centers = ((hist[1][:-1] + hist[1][1:]) / 2)

            if hist_bins != []:
                ax.plot(bin_centers, hist[0]+1, color = current_color, label = ("bin " + str(i + 1) + ": (" + "{:.2e}".format(hist_bins[i]) + " to " + "{:.2e}".format(hist_bins[i+1]) + ")"))
            else:
                ax.plot(bin_centers, hist[0]+1, color = current_color)

        elif dist_or_hist == 'hist':
            ax.hist(mp_lengths, bins = n_bins)
            
    if log == True:
        ax.set_yscale('log')
    ax.set_xlabel("Lengths")
    ax.set_ylabel("N + 1")
    ax.set_title("Distribution of Main Branch Lengths")
    
    # Mess around with legend
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.04,0.5))
    text = ax.text(-0.2,1.05, "Count", transform=ax.transAxes)
    plt.savefig("main_branch_lengths_new.png", bbox_extra_artists=(lgd, text), bbox_inches='tight')
    plt.show()

def OLD_plot_CDF(data, bins = [], x_axis = 'z_nums'): # data = LMM_times
    fig, ax = plt.subplots()
    color = iter(cm.jet(np.linspace(0,1,len(data))))
    
    # Loop over each bin
    for i in np.arange(len(data)):
        current_color = next(color)
        
        # Loop over each threshold
        for j in np.arange(len(data[i])):
            data_sorted = np.sort(data[i][j]) # Should automatically sort along the last axis
            probs = np.linspace(0,1, len(data[i][j]))
            if j == 0:
                ax.plot(data_sorted, probs, color = current_color, linestyle = '--', label = ("bin " + str(i + 1) + ": (" + "{:.2e}".format(bins[i]) + " to " + "{:.2e}".format(bins[i+1]) + ")"))
            elif j == 1:
                ax.plot(data_sorted, probs, color = current_color, linestyle = '-', label = ("bin " + str(i + 1) + ": (" + "{:.2e}".format(bins[i]) + " to " + "{:.2e}".format(bins[i+1]) + ")"))
    
    if x_axis == 'z_nums':
        ax.set_xlabel("Redshift of LMM")
    elif x_axis == 'snap_nums':
        ax.set_xlabel("snapnum of LMM")
        
    ax.set_ylabel("Probability")
    ax.set_title("Cumulative Probability Distribution of LMMs")
    ax.set_xscale("symlog", linthresh = 1, linscale = 0.4)
    #ax.set_xscale("symlog")
    
    # Unnecessarily Complicated Legend
    dashed_line = Line2D([0,1],[0,1],linestyle='--', color="black")
    solid_line = Line2D([0,1],[0,1],linestyle='-', color="black")
    ax.legend((dashed_line, solid_line), ('ε > 0.1', 'ε > 0.3'))
    # Maybe import Legend so you can create a separate legend and then use add_artist()?
    
    plt.show()

def calc_mass_growth_rate_WITH_COMMENTS(timesteps, masses, main_prog_list, prog_idx, forest_tbl):

    alpha_list = []
    count = 0
    count_close = 0
    total_count = 0
    unnorm = 1/(2.7*10**9)
    print("Length timesteps: ", len(timesteps))
    
    # Change z_values to lookback times
    cosmo = FlatLambdaCDM(H0=67.66, Om0=0.310)
    lookback_times = np.array(cosmo.lookback_time(timesteps))
    age = float(cosmo.age(0) / u.Gyr)
    ages_list = [age for i in np.arange(len(lookback_times[0]))]
    times = [ages_list - lt for lt in lookback_times]
    #print("times: ", times)
    
    # Loop through each halo
    for i in np.arange(len(masses)):
        
        # Mask up, friends!
        mask = (masses[i] > 10**12)
        masses_masked = np.array(masses)[i][mask]
        times_masked = times[i][mask]
        main_prog_list[i] = np.append(main_prog_list[i], np.zeros(101 - len(main_prog_list[i])))[mask]
        
        # Calculate alphas
        t_B = times_masked[0:(len(masses_masked) - 1)]
        M_B = masses_masked[0:(len(masses_masked) - 1)]
        t_A = times_masked[1:]
        M_A = masses_masked[1:]
        alpha = (t_B + t_A) * (M_B - M_A) / ((t_B - t_A)*(M_B + M_A))
        alpha_std = alpha[np.isnan(alpha) == False]
        alpha_list.append(alpha_std)
        
        # Check out weird values
        #for j in np.arange(len(alpha)):
            #print("j is: ", j, " and len(alpha) is: ", len(alpha))
        #    if alpha[j] > 10:
        #        count = count + 1
                #print("\nCount: ", count, " for i: ", i, " and j: ", j, " and alpha: ", alpha[j])
        #        dist_from_end = (len(alpha) -1) - j
                #print("Alpha distance from end: ", dist_from_end)
        #        if dist_from_end < 10:
        #            count_close = count_close + 1
                #print("M_C masses[i][j-1]: ", masses[i][j-1]*unnorm)
                #print("*M_B masses[i][j]*: ", masses[i][j]*unnorm)
                #print("M_A is masses[i][j+1]: ", masses[i][j+1]*unnorm)
                #print("M_Z is masses[i][j+2]: ", masses[i][j+2]*unnorm)
                    
    #print("count: ", count)
    #print("# dist_from_end < 10: ", count_close)
    return alpha_list
    
def OLD_calc_BINNED_mass_growth_rate(timesteps, masses, main_prog_list, prog_idx, forest_tbl):

    total_alpha_list = []
    
    #Loop over each bin
    for i in np.arange(len(timesteps)): 
        alpha_list = calc_mass_growth_rate(timesteps[i], masses[i], main_prog_list[i], prog_idx, forest_tbl)
        total_alpha_list.append(alpha_list)
        
    return total_alpha_list
    
def OLD_calc_mass_growth_rate(halo_idx, redshifts, prog_idx, forest_tbl):
    z_nums, masses_list, main_proj_list, maj_mergers = track_evol(halo_idx, redshifts, prog_idx, forest_tbl, x_axis = 'z_nums')
    alpha_list = []
    masses = np.array(masses_list)
    
    # Change z_values to lookback times
    cosmo = FlatLambdaCDM(H0=67.66, Om0=0.310)
    times = np.array(cosmo.lookback_time(z_nums))
    
    # Loop through each halo
    for i in np.arange(len(masses)):
        this_halo = np.empty(0)
        
        #alpha = (times_masked[0:(len(masses_masked) - 1)] + times_masked[1:]) * (masses_masked[0:(len(masses_masked) - 1)] - masses_masked[1:]) / ((times_masked[1:] - times_masked[0:(len(masses_masked) - 1)]) * (masses_masked[0:(len(masses_masked) - 1)] + masses_masked[1:])) # alpha is a vector
        
        for j in np.arange(len(masses[i]) - 1):
            # M_B = masses[i][j], M_A = masses[i][j+1]
            # t_B = times[i][j+1], t_A = times[i][j]
            if masses[i][j] >= 10**12 and masses[i][j+1] >= 10**12: # Threshold
                alpha = (times[i][j+1] + times[i][j]) * (masses[i][j] - masses[i][j+1]) / ((times[i][j+1] - times[i][j]) * (masses[i][j] + masses[i][j+1]))
            #alpha_std = alpha[np.isnan(alpha) == False]
                this_halo = np.append(this_halo, alpha)
                
            #if alpha < -2:
            #    print("Alpha is: ", alpha)
            #    #print("Alpha > 5 at: ", main_proj_list[i][j], " ( main proj is ", main_proj_list[i][0], ") at index: ", i, " ", j)
            #    print("Main proj is ", main_proj_list[i][0], " at index: ", i, " ", j)
            #    print("Masses are MB: ", masses[i][j], " and MA: ", masses[i][j+1], " with difference: ", (masses[i][j] - masses[i][j+1]))
            #    print("Times are tB: ", times[i][j+1], " and tA: ", times[i][j], " with difference: ", (times[i][j+1] - times[i][j]), "\n")

        alpha_list.append(this_halo)
        
    return alpha_list

def OLD_avg_bins(idx, bins, redshifts, prog_idx, forest_tbl, normalized = False):
    
    #Note that here, idx comes from bins, so it's a 2D matrix (five bins, some # halos per bin)
    final_masses = []
    final_z_nums = []
    
    # Loop over all the bins in idx
    for i in np.arange(len(bins) - 1):
        z_nums, masses, main_prog_list, maj_mergers = track_evol(idx[i], redshifts, prog_idx, forest_tbl, normalized)
        
        # Take the average over all the masses in this bin
        avg_masses = np.average(masses, axis = 0) # do I still need to specify axis = 0?

        # Save info for this bin
        final_z_nums.append(redshifts)
        final_masses.append(avg_masses)
        
    return final_z_nums, final_masses, main_prog_list, maj_mergers

    
def OLD_track_evol(idx, redshifts, prog_idx, forest_tbl, normalized = False, x_axis = 'z_nums'): # Just track the evolution -- bells and whistles come later
    z_nums = []
    final_masses = []
    final_main_prog_list = []
    major_mergers = []
    snap_nums = []

    # Loop over each halo
    for i in np.arange(len(idx)):
        target_idx = np.int(idx[i])
        progenitors = prog_idx[target_idx]
        main_progenitor_list = [target_idx]
        mergers = []
        
        # Generate main branch: find all the progenitors of each halo
        while len(progenitors) > 0:
            masses = [forest_tbl['mass'][i] for i in progenitors]
            main_progenitor = progenitors[np.argmax(masses)]
            main_progenitor_list.append(main_progenitor)
            
            # Count major mergers
            if len(progenitors) > 1:
                mass_order = np.flip(np.argsort(masses))
                sorted_progenitors = np.array(progenitors)[mass_order]
                sorted_masses = np.array(masses)[mass_order]
        
                if sorted_masses[0]*(1/4) < sorted_masses[1]:
                    mergers.append(progenitors[1])
            
            # Continue
            progenitors = prog_idx[main_progenitor]
            
        current_halo_masses = np.array([forest_tbl['mass'][mp] for mp in main_progenitor_list])  # Mass at each snapnum for the current halo (j)
        masses_std = np.append(current_halo_masses, np.zeros(101 - len(current_halo_masses)))*norm  # Standardize mass array to give length 101 and convert mass
        
        # Normalize, if desired
        if normalized == True:
            masses_std = masses_std / np.array(forest_tbl['mass'][target_idx]) #masses_std[len(masses_std) - 1]
        
        z_nums.append(redshifts)
        snap_nums.append(np.linspace(99, -1, 101))
        final_masses.append(masses_std)
        final_main_prog_list.append(main_progenitor_list)
        if mergers != []:
            major_mergers.append(mergers)
    
    if x_axis == 'z_nums':
        return z_nums, final_masses, final_main_prog_list, major_mergers
    elif x_axis == 'snap_nums':
        return snap_nums, final_masses, final_main_prog_list, major_mergers
    
def OLD_find_LMMs_binned(binned_major_mergers, forest_tbl, x_axis = 'snap_nums', redshifts = redshifts):
    
    # Mask up! (For second threshhold only)
    mask = [[len(this_bin[i][1]) != 0 for i in np.arange(len(this_bin))] for this_bin in binned_major_mergers]
    masked_mergers =[np.array(binned_major_mergers[i], dtype = object)[mask[i]] for i in np.arange(len(binned_major_mergers))]
    
    # Major mergers is already ordered from most recent to most distantly in the past, so just peel off the first entry
    LMM_list_small = [[this_bin[i][0][0] for i in np.arange(len(this_bin))] for this_bin in binned_major_mergers]
    LMM_list_big = [[this_bin[i][0][0] for i in np.arange(len(this_bin))] for this_bin in masked_mergers]
    
    # Find the time steps associated with the last major mergers
    if x_axis == 'z_nums':
        LMM_times_small = [redshifts[99 - forest_tbl['snap_num'][LMM_bin]] for LMM_bin in LMM_list_small]
        LMM_times_big = [redshifts[99 - forest_tbl['snap_num'][LMM_bin]] for LMM_bin in LMM_list_big]
        
    elif x_axis == 'snap_nums':  
        LMM_times_small = [forest_tbl['snap_num'][LMM_bin] for LMM_bin in LMM_list_small]
        LMM_times_big = [forest_tbl['snap_num'][LMM_bin] for LMM_bin in LMM_list_big]
    
    all_LMM_list = [LMM_list_small, LMM_list_big]
    all_LMM_times = [LMM_times_small, LMM_times_big]
    
    return all_LMM_list, all_LMM_times
    
# Example of old implementation

## 1b
## Average mass evolutions (in bins)
# Find and bin the halos
#bin_halo_idx = help_func.bin_halos(forest_table, snap, bins)
# Track them (and take their averages in the process)
#bin_z_nums, bin_masses, bin_main_prog_list, major_mergers = help_func.avg_bins(bin_halo_idx, bins, redshifts, progenitor_idx, forest_table)
# Plot them
#help_func.plot_evol(bin_z_nums, bin_masses, "mass_evol_binned", bins, avg_tf)

## 1c
## Normalized average mass evolutions (binned)
# Track them (and take their averages in the process)
#norm_bin_z_nums, norm_bin_masses, norm_bin_main_prog_list, norm_maj_mergers = help_func.avg_bins(bin_halo_idx, bins, redshifts, progenitor_idx, forest_table, norm_tf)
# Plot them
#help_func.plot_evol(norm_bin_z_nums, norm_bin_masses, "mass_evol_binned", bins, avg_tf, norm_tf)

## 1d
## Tracking major mergers
# Choose just one halo, and choose x-axis (z_nums or snap_nums)
#halo_id = [6207440]
#xaxis = 'snap_nums'
# Track the halo
#snap_nums, masses, main_prog_list, maj_mergers = help_func.track_evol_multiple(halo_id, redshifts, prog_idx = progenitor_idx, forest_tbl = forest_table, x_axis = xaxis)
# Plot them
#help_func.plot_evol(snap_nums, masses, major_mergers = maj_mergers, x_axis = xaxis)

#################
# Legend Things #
#################

#dashed_line = Line2D([0,1],[0,1],linestyle='--', color="black")
#solid_line = Line2D([0,1],[0,1],linestyle='-', color="black")
#ax.legend((dashed_line, solid_line), ('ε > 0.1', 'ε > 0.3'))
# Maybe import Legend so you can create a separate legend and then use add_artist()?
# Mess around with legend
#handles, labels = ax.get_legend_handles_labels()
#lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.04,0.5))
#lgd = ax.legend(handles, labels, loc='lower right')
#text = ax.text(-0.2,1.05, "Prob", transform=ax.transAxes)