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
import haccytools.mergertrees.visualization
import pickle
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
from astropy.cosmology import FlatLambdaCDM
from itertools import groupby
from matplotlib.ticker import ScalarFormatter

# A few globals:
redshifts = np.flip(np.array([10.044, 9.8065, 9.5789, 9.3608, 9.1515, 8.7573, 8.5714, 8.3925, 8.0541, 7.8938, 7.7391, 7.4454, 7.3058, 7.04, 6.9134, 6.6718, 6.5564, 6.3358, 6.1277, 6.028, 5.8367, 5.6556, 5.4839, 5.3208, 5.2422, 5.0909, 4.9467, 4.7429, 4.6145, 4.4918, 4.3743, 4.2618, 4.1015, 4.00, 3.8551, 3.763, 3.6313, 3.5475, 3.4273, 3.3133, 3.205, 3.102, 3.0361, 2.9412, 2.8506, 2.7361, 2.6545, 2.5765, 2.4775, 2.4068, 2.3168, 2.2524, 2.1703, 2.0923, 2.018, 1.9472, 1.8797, 1.7994, 1.7384, 1.68, 1.6104, 1.5443, 1.4938, 1.4334, 1.3759, 1.321, 1.2584, 1.2088, 1.152, 1.1069, 1.0552, 1.006, 0.9591, 0.9143, 0.8646, 0.824, 0.7788, 0.7358, 0.6948, 0.6557, 0.6184, 0.5777, 0.5391, 0.5022, 0.4714, 0.4337, 0.4017, 0.3636, 0.3347, 0.3035, 0.2705, 0.2423, 0.2123, 0.1837, 0.1538, 0.1279, 0.1008, 0.0749, 0.0502, 0.0245, 0.00]))

###############################################################################
# Find Halos:                                                                 #
# return an index of halo id's whose mass values match the input requirements #
###############################################################################

def find_halos(forest_table, sn, quant = 0, extremum = '', mass_range = []):
        
    # Extract the mass values (and associated halo id's) from desired snapshot
    masses_sn = np.array(forest_table.mass.loc[forest_table.snap_num == sn]) # (2.7*10**9)*?
    halos_sn = np.array(forest_table.halo_id.loc[forest_table.snap_num == sn])
    
    print(len(masses_sn))

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
    
def track_evol(idx, redshifts, prog_idx, forest_masses, forest_snap_nums, thresholds, thresholds_are_absolute = False, normalized = False, x_axis = 'z_nums', z_threshold = 1): # Just track the evolution of one halo -- bells and whistles come later
    
    major_mergers = [[] for thresh in range(len(thresholds))]
    major_merger_times = [[] for thresh in range(len(thresholds))]
    LMMs = [None for thresh in range(len(thresholds))]
    LMM_times = [None for thresh in range(len(thresholds))]
    fossil_groups = [None for thresh in range(len(thresholds))] # It's either a yes or a no for each halo, just need to know for each thresh
    
    # Start tracking
    target_idx = np.int(np.array(idx)) #int(idx)
    progenitors = prog_idx[target_idx]
    main_progenitor_list = [target_idx]

    # Generate main branch: find all the progenitors of the halo (at each timestep)
    while len(progenitors) > 0:
        masses = [forest_masses[i] for i in progenitors] # Find masses for each progenitor of the halo at this time step
        main_progenitor = progenitors[np.argmax(masses)]
        z_current = redshifts[100 - forest_snap_nums[main_progenitor]]
        main_progenitor_list.append(main_progenitor)

        # Count major mergers
        if len(progenitors) > 1:
            order_by_mass = np.argsort(masses)[::-1] # Special syntax for reversing order
            sorted_progenitors = np.array(progenitors)[order_by_mass] # Why do I need np.array here?
            sorted_masses = np.array(masses)[order_by_mass]

            # Loop over all thresholds (definitions of major mergers)
            for i, threshold in enumerate(thresholds):
                absolute_threshold = threshold if thresholds_are_absolute else threshold * sorted_masses[0]
                
                # Is this a major merger? (under the current definition)
                if sorted_masses[1] >= absolute_threshold:
                    # yay, it's a major merger!
                    major_mergers[i].append(sorted_progenitors[0]) # 1? 0? i?
                    major_merger_times[i].append(forest_snap_nums[sorted_progenitors[0]])
                    
                    # Is this an LMM? # Might not even be necessary...
                    if LMMs[i] is None:
                        LMMs[i] = sorted_progenitors[0]
                        LMM_times[i] = forest_snap_nums[sorted_progenitors[0]]

                        # Is this a potential fossil group?
                        if z_current >= z_threshold:
                            fossil_groups[i] = target_idx
                
        # Continue
        progenitors = prog_idx[main_progenitor]

    raw_masses = forest_masses[main_progenitor_list]
    final_masses = np.append(raw_masses, np.zeros(101 - len(raw_masses))) #*norm  # Standardize mass array with length 101 and un-normalized mass

    # Include fossil group candidates with no LMMs
    for this_thresh in range(len(thresholds)):
        if LMMs[this_thresh] is None:
            fossil_groups[this_thresh] = target_idx
    
    if normalized == True:
        final_masses = final_masses / forest_masses[target_idx] # Doesn't need .values because there's just one

    if x_axis == 'z_nums':
        z_nums = redshifts
        return z_nums, final_masses, main_progenitor_list, major_mergers, major_merger_times, LMMs, LMM_times, fossil_groups
    elif x_axis == 'snap_nums':
        snap_nums = np.linspace(100, 0, 101)
        return snap_nums, final_masses, main_progenitor_list, major_mergers, major_merger_times, LMMs, LMM_times, fossil_groups
    
###############################################################################################################
# Track Evolution: multiple halos                                                                             #
# build a main progenitor branch for EACH halo in the index;                                                  #
# return LISTS of masses, time steps, main progenitor id's, and locations of major mergers for each mp branch #
###############################################################################################################

def track_evol_multiple(idx, redshifts, prog_idx, forest_masses, forest_snap_nums, thresholds, thresholds_are_absolute = False, normalized = False, x_axis = 'z_nums', z_threshold = 1):
    #print("In track_evol_multiple")
    all_timesteps= []
    all_masses = []
    all_main_prog_list = []
    all_major_mergers = []
    all_major_merger_times = []
    all_LMMs = []
    all_LMM_times = []
    all_fossil_groups = []

    # Loop over each halo
    for this_halo_id in idx:
        timesteps, masses, main_progenitor_list, major_mergers, major_merger_times, LMMs, LMM_times, fossil_groups = track_evol(this_halo_id, redshifts, prog_idx, forest_masses, forest_snap_nums, thresholds, thresholds_are_absolute, normalized, x_axis, z_threshold)
        
        all_timesteps.append(timesteps)
        all_masses.append(masses)
        all_main_prog_list.append(main_progenitor_list)
        all_major_mergers.append(major_mergers)
        all_major_merger_times.append(major_merger_times)
        all_LMMs.append(LMMs)
        all_LMM_times.append(LMM_times)
        all_fossil_groups.append(fossil_groups)
                    
    return all_timesteps, all_masses, all_main_prog_list, all_major_mergers, all_major_merger_times, all_LMMs, all_LMM_times, all_fossil_groups

######################################################################################################################
# Track Evolution: binned halos                                                                                      #
# index of halos is binned -- build main progenitor branches for each halo in each bin;                              #
# return BINNED lists of masses, time steps, main progenitor id's, and locations of major mergers for each mp branch #
######################################################################################################################

def track_evol_binned(idx, bins, redshifts, prog_idx, forest_masses, forest_snap_nums, thresholds, thresholds_are_absolute = False, normalized = False, x_axis = 'z_nums', z_threshold = 1):
    all_timesteps= []
    all_masses = []
    all_main_prog_list = []
    all_major_mergers = []
    all_major_merger_times = []
    all_LMMs = []
    all_LMM_times = []
    all_fossil_groups = []

    # Loop over each bin
    for this_bin_idx in idx:
        timesteps, masses, main_progenitor_list, major_mergers, major_merger_times, LMMs, LMM_times, fossil_groups = track_evol_multiple(this_bin_idx, redshifts, prog_idx, forest_masses, forest_snap_nums, thresholds, thresholds_are_absolute, normalized, x_axis, z_threshold)
        
        all_timesteps.append(timesteps)
        all_masses.append(masses)
        all_main_prog_list.append(main_progenitor_list)
        all_major_mergers.append(major_mergers)
        all_major_merger_times.append(major_merger_times)
        all_LMMs.append(LMMs)
        all_LMM_times.append(LMM_times)
        all_fossil_groups.append(fossil_groups)
 
    return all_timesteps, all_masses, all_main_prog_list, all_major_mergers, all_major_merger_times, all_LMMs, all_LMM_times, all_fossil_groups
    
###############################################################################
# Average Bins:                                                               #
# track the evolution of halos in bins, return the average masses of each bin #
###############################################################################

def avg_bins(idx, bins, redshifts, prog_idx, forest_masses, forest_snap_nums, thresholds, thresholds_are_absolute = False, normalized = False, x_axis = 'z_nums', z_threshold = 1):
    
    # Track the evolution of halos in bins
    timesteps, masses, main_prog_list, maj_mergers, mm_times, LMMs, LMM_times, fossil_clusters = track_evol_binned(idx, bins, redshifts, prog_idx, forest_masses, forest_snap_nums, thresholds, thresholds_are_absolute, normalized, x_axis, z_threshold)
    
    # Take the average over all the masses in each bin
    final_masses = []
    final_timesteps = []
    for i in np.arange(len(bins) - 1): # Same as len(masses)
        avg_masses = np.average(masses[i], axis = 0) # do I still need to specify axis = 0?
        final_masses.append(avg_masses)
    
    if x_axis == 'z_nums':
        final_timesteps = [redshifts for j in range(len(timesteps))]
    else:
        final_timesteps = timesteps
        
    return final_timesteps, final_masses

########################################################
# Plot evolution:                                      #        
# Display M(z) for halos that we tracked in track_evol #
########################################################
    
def plot_evol(timesteps, masses, mm_times = [], thresholds = [], filename = "new_plot", bins = [], redshifts = redshifts, avg = False, normalized = False, extremum = '', quant = 0, mass_range = [], x_axis = "z_nums", fig = None, ax = None, auto_legend = True, cust_legend = [], extra_legend = [], cust_color = None, **kwargs):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = fig
        ax = ax
    color = iter(cm.jet(np.linspace(0,1,len(masses)))) # Bin colors
    bin_legend_handles = []
    
    # Change zeros to nans so they don't get plotted
    for m in range(len(masses)):
        masses[m][masses[m] == 0] = np.nan
    
    # Plot
    for n in range(len(masses)): # loop over all bins
        if cust_color is None:
            current_color = next(color)
        else:
            current_color = cust_color
        ax.plot(timesteps[n], masses[n], color = current_color, **kwargs)

        # Pick your plot style
        if avg == True:
            if cust_legend != []:
                bin_legend_handles.append(mpatches.Patch(color=current_color, label=cust_legend))
            else:
                bin_legend_handles.append(mpatches.Patch(color=current_color, label="bin " + str(n + 1) + ": (" + "{:.2e}".format(bins[n]) + " to " + "{:.2e}".format(bins[n+1]) + ")"))
            if normalized == True:
                title = "Normalized averaged mass evolution of halos in " + str(int(len(bins) - 1)) + " bins"
                filename = "norm_" + filename
            elif normalized == False:
                title = "Averaged mass evolution of halos in " + str(int(len(bins) - 1)) + " bins"
        else:
            if extremum == 'max':
                title = "Mass evolution of " + str(quant) + " most massive halos"
            elif extremum == 'min':
                title = "Mass evolution of " + str(quant) + " least massive halos"
            elif extremum == '':
                if mass_range != []:
                    title = "Mass evolution of halos in range " + "{:.2e}".format(mass_range[0]) + " to " + "{:.2e}".format(mass_range[1])
                else:
                    title = "Mass evolution of halo(s)"
    
    # Display major mergers (if desired)
    # Only works for non_binned results, really...
    if mm_times != []: # Does this handle both positions? Or only one, really?
        color2 = iter(cm.jet(np.linspace(0,1,len(mm_times)))) # Bin colors again
        
        # Loop over each halo
        for i, this_halo in enumerate(mm_times):
            current_color = next(color2)
            linestyles = iter([':', '--'])
            
            # Loop over the two threshholds
            for j, this_thresh in enumerate(thresholds):
                this_linestyle = next(linestyles)
                
                # Loop over each major merger (for this halo)
                for this_mm in this_halo[j]:
                    # Why do I need these "ints" in here?
                    if x_axis == "z_nums":
                        merg = redshifts[int(100 - this_mm)]
                    elif x_axis == "snap_nums":
                        merg = this_mm
                    ax.axvline(merg, color = current_color, linestyle = this_linestyle)
            
    ax.set_yscale('log', nonpositive = 'clip')
    #ax.set_title(title)
    if x_axis == "z_nums":
        ax.set_xlim(10.044, 0)
        ax.set_xlabel(r"Redshift $z$")
    #elif x_axis == "snap_nums":
    #    ax.set_xlabel(r"Snapnumber $(SN = 100 \rarrow z = 0)$")
    #if normalized == True:
    #    ax.set_ylabel(r'Normalized Mass $[h^{-1}M_\odot]$')
    #else:
    #    ax.set_ylabel(r'Mass $[h^{-1}M_\odot]$')
        
    if auto_legend == True:
        # Fancy legend
        leg1 = ax.legend(handles = bin_legend_handles, loc='lower right')

        # If relevant, add second legend for the linestyles.
        # leg1 will be removed from figure
        if extra_legend != []:
            legs = extra_legend[0]
            labels = extra_legend[1]
            leg2 = ax.legend((legs), (labels), loc = 'center right')
        ax.add_artist(leg1)

    return fig, ax

##################################################################
# NEW Plot Last Major Mergers:                                   #
# PDF of last major mergers at different timesteps               #
# Now customized for one threshold, compare fgs to whole dataset #
##################################################################

def plot_LLMs(LLM_times_raw, thresholds, bins, bin_labels = [], cust_legend = [], mass_range = [], x_axis = 'z_nums', fig = None, ax = None, **kwargs): # Note: assume LMM_times are binned

    # Rearrange, get rid of any "nones", and convert to z's if needed:
    if x_axis == 'z_nums':
        LLM_times = [[[redshifts[int(100 - this_halo[i])] for this_halo in this_bin if this_halo[i] is not None] for this_bin in LLM_times_raw] for i in range(len(thresholds))]
    elif x_axis == 'snap_nums':
        LLM_times = [[[this_halo[i] for this_halo in this_bin if this_halo[i] is not None] for this_bin in LLM_times_raw] for i in range(len(thresholds))]
        
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    # Calculate histogram of LMM_times, plot as a distribution
    color = iter(cm.jet(np.linspace(0,1,len(bins) - 1)))
    bin_legend_handles = []
    
    # Plot once for each bin
    for bin_n in range(len(bins) - 1):
        current_color = next(color)
        
        # Plot once for each threshold
        for i, this_threshold in enumerate(LLM_times): # length is 2
            norm_factor = len(this_threshold[bin_n])
            hist = np.histogram(this_threshold[bin_n])
            bin_centers = (hist[1][:-1] + hist[1][1:]) / 2
            ax.plot(bin_centers, hist[0]/norm_factor, color = current_color, **kwargs)
            if i == 0:
                if bin_labels == []:
                    bin_legend_handles.append(mpatches.Patch(color=current_color, label="bin " + str(bin_n + 1) + ": (" + "{:.2e}".format(bins[bin_n]) + " to " + "{:.2e}".format(bins[bin_n+1]) + ")"))
                else:
                    bin_legend_handles.append(mpatches.Patch(color=current_color, label=bin_labels[bin_n]))
    
    # Set customized labels and titles
    if x_axis == 'z_nums':
        ax.set_xlabel("Redshift of LLM")
        ax.set_xscale('symlog')
    elif x_axis == 'snap_nums':
        ax.set_xlabel("snapnum of LLM")
    if mass_range != []:
        ax.set_title("PDF of Last Luminous Mergers in range: " + "{:.1e}".format(mass_range[0]) + " to " + "{:.1e}".format(mass_range[1]))
    elif mass_range == []:
        ax.set_title("PDF of Last Luminous Mergers")
    ax.set_ylabel("Probability")
    
    # Unnecessarily Complicated Tick Marks
    stepsize = 1
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end, stepsize))
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    
    # Fancy legend
    leg1 = ax.legend(handles = bin_legend_handles, loc='upper right')
    if cust_legend != []:
        legs = cust_legend[0]
        labels = cust_legend[1]
        leg2 = ax.legend((legs), (labels), loc = 'center right', bbox_to_anchor = (1, 0.65))
    ax.add_artist(leg1)
    
    # Save
    return fig, ax

################################################################################################
# NEW Plot Cumulative Distribution Function:                                                   #
# Plot the probability that some data (X) will take on a value less than/equal to quantity (x) #
# Now optimized for one threshold, comparing fgs to whole dataset                              #
################################################################################################

def plot_CDF(data_raw, comparison_data, thresholds, bins = [], bin_labels = [], cust_legend = [], redshifts = redshifts, x_axis = 'z_nums', fig = None, ax = None, z_end = None, **kwargs): # data = LLM_times, comp_data = binned_masses (because that gives us the total number of trees, including those without major mergers) -- isn't there a better way to do that part?

    # Rearrange and get rid of any "nones":
    if x_axis == 'z_nums':
        data = [[[redshifts[int(100 - this_halo[i])] for this_halo in this_bin if this_halo[i] is not None] for this_bin in data_raw] for i in range(len(thresholds))]
    elif x_axis == 'snap_nums':
        data = [[[this_halo[i] for this_halo in this_bin if this_halo[i] is not None] for this_bin in data_raw] for i in range(len(thresholds))]
    
    # Get ready to plot
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    color = iter(cm.jet(np.linspace(0,1,len(comparison_data))))
    bin_legend_handles = []
    
    # Loop over each bin
    for bin_n in range(len(bins) - 1):
        current_color = next(color)
        
        # Loop over each threshold
        for i, this_thresh in enumerate(data):
            
            # Plot
            data_sorted = np.sort(this_thresh[bin_n]) # Should automatically sort along the last axis
            hist_keys = [key for key, group in groupby(data_sorted)] # Redshift values
            hist_values = [len(list(group)) for key, group in groupby(data_sorted)] # Count of each redshift value
            cum_probs = np.cumsum(hist_values) / len(comparison_data[bin_n])
            
            if z_end is not None:
                hist_keys = np.append(hist_keys, z_end)
                cum_probs = np.append(cum_probs, cum_probs[-1])
                
            ax.plot(hist_keys, cum_probs, color = current_color, **kwargs)
            
            if i == 0:
                if bin_labels == []:
                    bin_legend_handles.append(mpatches.Patch(color=current_color, label="bin " + str(bin_n + 1) + ": (" + "{:.2e}".format(bins[bin_n]) + " to " + "{:.2e}".format(bins[bin_n+1]) + ")"))
                else:
                    bin_legend_handles.append(mpatches.Patch(color=current_color, label = bin_labels[bin_n]))
    
    # Accessorize
    ax.set_ylabel("Probability")
    
    if x_axis == 'z_nums':
        ax.set_xlabel("Redshift of LLM")
        ax.set_xscale("symlog", linthresh = 1, linscale = 0.4)
        
        # Unnecessarily Complicated Tick Marks
        stepsize = 1
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(0, end, stepsize))
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ScalarFormatter())
            
    elif x_axis == 'snap_nums':
        ax.set_xlabel("Snapnum of LLM")
    
    # Fancy legend
    leg1 = ax.legend(handles = bin_legend_handles, loc='lower right')
    if cust_legend != []:
        legs = cust_legend[0]
        labels = cust_legend[1]
        leg2 = ax.legend((legs), (labels), loc = 'center right', bbox_to_anchor = (1, 0.35))
    ax.add_artist(leg1)
    
    # Save
    return fig, ax
 
#######################################################################################
# Calculate Mass Growth Rate:                                                         #
# Find alphas (Srisawat eq. 7) for each halo in halo_idx, for each pair of time steps #
#######################################################################################

def calc_mass_growth_rate(timesteps, masses, main_prog_list):    
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

def calc_BINNED_mass_growth_rate(timesteps, masses, main_prog_list):

    total_alpha_list = [calc_mass_growth_rate(timesteps[i], masses[i], main_prog_list[i]) for i in np.arange(len(timesteps))]
    return total_alpha_list

################################################################
# Plot Distribution:                                           #
# Display PDF distribution of some metric using np.histogram() #
################################################################

def plot_mass_growth_rates(alphas, metric_name, bins = [], bin_labels = [], zoom = False, n_hist_bins = 10, log = False, fig = None, ax = None, **kwargs): # bin labels is an optional way to supply custom bin labels

    color = iter(cm.jet(np.linspace(0,1,len(alphas))))
    all_alphas = []
    hist_bins = np.linspace(-10, 10, n_hist_bins)
    bin_legend_handles = []
    if ax == None:
        fig, ax = plt.subplots()
    else:
        fig = fig
        ax = ax

    # For all bins
    for bin_n, this_alpha in enumerate(alphas):
        current_color = next(color)
        
        # For all halo trees
        this_halo_alphas = np.concatenate([this_alpha[i] for i in range(len(this_alpha))])
        all_alphas.append(this_halo_alphas)

        if zoom == True:
            alphas_in_range = np.array([all_alphas[bin_n][i] for i in range(len(all_alphas[bin_n])) if (all_alphas[bin_n][i] < 10 and all_alphas[bin_n][i] > -10)])
            hist = np.histogram(alphas_in_range, bins = hist_bins)
        else:
            hist = np.histogram(all_alphas[bin_n], bins = hist_bins)

        bin_centers = ((hist[1][:-1] + hist[1][1:]) / 2)
        ax.plot(bin_centers, hist[0]+1, color = current_color, **kwargs)

        if bin_labels == []:
            if bins == []:
                binned_tf = []
            else:
                bin_legend_handles.append(mpatches.Patch(color=current_color, linestyle = '-', label=("bin " + str(bin_n + 1) + ": (" + "{:.2e}".format(bins[bin_n]) + " to " + "{:.2e}".format(bins[bin_n+1]) + ")")))
        else:
            bin_legend_handles.append(mpatches.Patch(color=current_color, linestyle = '-', label=("bin " + str(bin_n + 1) + ": " + bin_labels[bin_n])))
    
    # Accesorize the plots
    if log == True:
        ax.set_xscale('symlog')
        
    # Fancy Legend
    leg1 = ax.legend(handles = bin_legend_handles, loc='lower center', bbox_to_anchor=(0.6, 0))
    ax.add_artist(leg1)
    
    # Unnecessarily Complicated Tick Marks
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    
    ax.set_yscale('log')
    ax.set_xlabel(r"$\alpha_M$ $[d\logM/d\logt]$")
    ax.set_ylabel("N + 1")
    return fig, ax
    
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
    
    # Mess around with legend
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.04,0.5))
    text = ax.text(-0.2,1.05, "Count", transform=ax.transAxes)
    plt.savefig("main_branch_lengths_new.png", bbox_extra_artists=(lgd, text), bbox_inches='tight')
    plt.show()

def plot_main_branch_length_new(mp_list, n_bins = 32, hist_bins = [], zoom = False, log = True, dist_or_hist = 'dist'):
    # Note: assume mp_list is binned
    fig, (ax, cax) = plt.subplots(1, 2, figsize=(5, 3), gridspec_kw=dict(wspace=0.03, width_ratios=[1, 0.03]))
    cmap = plt.cm.jet
    colors = cmap(np.linspace(0,1,len(mp_list)))
    color_edges = np.log10(hist_bins)
    tick_edges = [10, 11, 12, 13, 14] # currently hardcoded
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('blablabla', colors, cmap.N)
    norm = plt.matplotlib.colors.BoundaryNorm(color_edges, cmap.N)

    length_bins = [i*(101/n_bins) for i in np.arange(n_bins + 1)]

    # Loop over each bin
    for i in range(len(mp_list)):
        current_color = colors[i]
        mp_lengths = []

        # Loop over each halo root
        for j in np.arange(len(mp_list[i])):
            mp_lengths.append(len(mp_list[i][j]))

        if dist_or_hist == 'dist':
            hist, edges = np.histogram(mp_lengths, bins = length_bins)
            bin_centers = ((edges[:-1] + edges[1:]) / 2)
            ax.plot(bin_centers, hist+1, color = current_color)

        elif dist_or_hist == 'hist':
            ax.hist(mp_lengths, bins = n_bins)

    if log:
        ax.set_yscale('log')
    ax.set_xlabel("length of the main progenitor branch")
    ax.set_ylabel("halo count $N + 1$")

    cb = plt.matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, ticks=tick_edges, boundaries=color_edges)
    cb.set_ticklabels([f'$10^{{{m}}}$' for m in tick_edges])
    cb.ax.tick_params(labelsize=10)
    cb.set_label('mass bin [$h^{-1}M_\odot$]')

    fig.tight_layout()
    return fig, ax
    
# Michael, here's some code for you!
#bin_halo_idx = help_func.bin_halos(forest_table, snap, rangel_bins)
#xaxis = 'z_nums'
#%time binned_timesteps, binned_masses, binned_main_prog_list, binned_maj_mergers, binned_mm_times, binned_LMMs, binned_LMM_times, binned_fossil_groups = help_func.track_evol_binned(bin_halo_idx, rangel_bins, redshifts, progenitor_idx, forest_masses, forest_snap_nums, thresholds, x_axis = xaxis)
#help_func.plot_main_branch_length(binned_main_prog_list, hist_bins = rangel_bins, log = True)

#################################################################
# Calculate Cumulative Number of Major Mergers: (just one halo) #
# Find redshifts associated with each major merger;             #
# Count and average number of major mergers at each redshift    #
#################################################################

def calc_cum_lum_mergers(lum_merger_times, redshifts): # Note: assume major_mergers is for just one halo
    
    # For all redshifts z, count # of mms that took place between 0 and z
    mask = [[[(redshifts[100 - lum_merger_times[halo_n][time_n]] <= z) for time_n in range(len(lum_merger_times[halo_n]))] for z in redshifts] for halo_n in range(len(lum_merger_times))]
    # Count the true values in the mask
    cum_lms = [[mask[i][k].count(True) for k in range(len(mask[i]))] for i in range(len(lum_merger_times))] 
    
    return cum_lms # dim: 2 thresholds, 101 z's, some # MMs (T/F values)
    
###########################################################################
# Average Cumulative Number of Major Mergers: (multiple halos)            #
# Run calc_cum_maj_mergers once for each halo, find mean for each z value #
###########################################################################

def avg_cum_lum_mergers(lum_merger_times, redshifts): # Note: assume major_mergers contains multiple halos
    
    cum_lms = [calc_cum_lum_mergers(lum_merger_times[i], redshifts) for i in range(len(lum_merger_times))] # One for each halo
    avg = [np.average([cum_lms[i][j] for i in np.arange(len(cum_lms))], axis = 0) for j in range(len(cum_lms[0]))] # Kind of cheating... hard-coded to get the right length for the loop (2)
    
    return avg

#################################################################
# Binned Avg Cumulative Number of Major Mergers: (binned halos) #
# Run avg_cum_maj_mergers once for each bin                     #
#################################################################

def binned_avg_cum_lum_mergers(lum_merger_times, redshifts,): # Note: assume major_mergers is binned
    
    binned_avg = [avg_cum_lum_mergers(lum_merger_times[i], redshifts) for i in range(len(lum_merger_times))]
    return binned_avg 

##############################################################################
# NEW Plot Mean Number of Major Mergers (Fakhouri & Ma, 2011, Fig. 7):       #
# Mean # of major mergers experienced (by a halo at z0 = 0) between z0 and z #
# Now with fancy linestyles!                                                 #
##############################################################################

def plot_cum_lms(binned_averages, bins, thresholds, bin_labels = [], cust_legend = [], redshifts = redshifts, linestyle_labels = [], fig = None, ax = None, **kwargs):
    # If you provide a linestyle, make sure it's in a list!
    # Current stuff
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
        
    color = iter(cm.jet(np.linspace(0,1,len(bins) - 1)))
    bin_legend_handles = []
    
    for bin_n in range(len(bins) - 1): # Loop over bins
        current_color = next(color)
        
        for thresh_n in range(len(thresholds)): # Loop over thresholds
            ax.plot(redshifts, binned_averages[bin_n][thresh_n], color = current_color, **kwargs)
            
            if thresh_n == 0:
                if bin_labels == []:
                    bin_legend_handles.append(mpatches.Patch(color=current_color, label="bin " + str(bin_n + 1) + ": (" + "{:.2e}".format(bins[bin_n]) + " to " + "{:.2e}".format(bins[bin_n+1]) + ")"))
                else:
                    bin_legend_handles.append(mpatches.Patch(color=current_color, label = bin_labels[bin_n]))

     # Fancy legend
    leg1 = ax.legend(handles = bin_legend_handles, loc='lower right')
    if cust_legend != []:
        legs = cust_legend[0]
        labels = cust_legend[1]
        leg2 = ax.legend((legs), (labels), loc = 'center right', bbox_to_anchor = (1, 0.35))
    ax.add_artist(leg1)
    
    # More accessories
    ax.set_xscale("symlog", linthresh = 1, linscale = 0.4)
    ax.set_xlabel("Redshift (z)")
    ax.set_yscale('log')
    ax.set_ylabel("Mean # Luminous Mergers between z0 and z")
    
    # Unnecessarily Complicated Tick Marks
    stepsize = 1
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end, stepsize))
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    
    return fig, ax

#############################
# Plot PDF of Major Mergers #
#############################

def pdf_lms(mm_times_raw, thresholds, bins, bin_labels = [], cust_legend = [], mass_range = [], x_axis = 'z_nums', fig = None, ax = None, linestyle_labels = [], **kwargs): # Note: assume mm_times_raw are binned

    # Rearrange, get rid of any empty lists, flatten all mms for all halos, and convert to z's if needed:
    if x_axis == 'z_nums':
        mm_times = [[[redshifts[int(100 - this_mm)] for this_halo in this_bin if len(this_halo[i]) != 0 for this_mm in this_halo[i]] for this_bin in mm_times_raw] for i in range(len(thresholds))]
    elif x_axis == 'snap_nums':
        mm_times = [[[this_mm for this_halo in this_bin if len(this_halo[i]) != 0 for this_mm in this_halo[i]] for this_bin in mm_times_raw] for i in range(len(thresholds))]
    
    # Calculate histogram of mm_times, plot as a distribution
    # Make a new fig, ax if needed, or read one in
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    color = iter(cm.jet(np.linspace(0,1,len(bins) - 1)))
    bin_legend_handles = []
    
    # Plot once for each bin
    for bin_n in range(len(bins) - 1):
        current_color = next(color)
        
        # Plot once for each threshold
        for i, mms_in_this_threshold in enumerate(mm_times): # length is 2
            norm_factor = len(mms_in_this_threshold[bin_n])
            hist = np.histogram(mms_in_this_threshold[bin_n], bins = 10)
            bin_centers = (hist[1][:-1] + hist[1][1:]) / 2
            ax.plot(bin_centers, hist[0]/norm_factor, color = current_color, **kwargs)
            
            # Add a legend handle for the first threshold
            if i == 0:
                if bin_labels == []:
                    bin_legend_handles.append(mpatches.Patch(color=current_color, label="bin " + str(bin_n + 1) + ": (" + "{:.2e}".format(bins[bin_n]) + " to " + "{:.2e}".format(bins[bin_n+1]) + ")"))
                else:
                    bin_legend_handles.append(mpatches.Patch(color=current_color, label= bin_labels[bin_n]))
    
    # Set customized labels and titles
    if x_axis == 'z_nums':
        ax.set_xlabel("Redshift of LLM")
        ax.set_xscale("symlog", linthresh = 1, linscale = 0.4)
    elif x_axis == 'snap_nums':
        ax.set_xlabel("snapnum of LLM")
    if mass_range != []:
        ax.set_title("PDF of Luminous Merger Times in range: " + "{:.1e}".format(mass_range[0]) + " to " + "{:.1e}".format(mass_range[1]))
    elif mass_range == []:
        ax.set_title("PDF of Luminous Merger Times")
    ax.set_ylabel("Probability")
    
    # Fancy legend
    leg1 = ax.legend(handles = bin_legend_handles, loc='upper right')
    if cust_legend != []:
        legs = cust_legend[0]
        labels = cust_legend[1]
        leg2 = ax.legend((legs), (labels), loc = 'center right', bbox_to_anchor = (1, 0.65))
    ax.add_artist(leg1)
    
    # Unnecessarily Complicated Tick Marks
    stepsize = 1
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end, stepsize))
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())

    return fig, ax
    
####################
# Plot comparisons #
####################

def plot_compare(binned_averages, bins, thresholds, filename = "cum_mms", redshifts = redshifts):
    # Binned averages goes: group 1, bins, thresholds, then it varies (in this case, timesteps)
    
    # Current stuff
    fig, ax = plt.subplots()
    color = iter(cm.jet(np.linspace(0,1,len(bins) - 1)))
    linestyle_options = ['-', '--', ':', '-.'][:len(thresholds)]
    bin_legend_handles = []
    
    for bin_n in range(len(bins) - 1): # Loop over bins
        current_color = next(color)
        linestyles = iter(linestyle_options)
        
        for thresh_n in range(len(thresholds)): # Loop over thresholds
            this_linestyle = next(linestyles)
            ax.plot(redshifts, binned_averages[bin_n][thresh_n], color = current_color, linestyle = this_linestyle)
            
            if thresh_n == 0:
                bin_legend_handles.append(mpatches.Patch(color=current_color, label="bin " + str(bin_n + 1) + ": (" + "{:.2e}".format(bins[bin_n]) + " to " + "{:.2e}".format(bins[bin_n+1]) + ")"))

     # Fancy legend
    leg1 = ax.legend(handles = bin_legend_handles, loc='lower right')
    
    # Add second legend for the linestyles.
    # leg1 will be removed from figure
    thresh_legs = []
    thresh_labels = []
    linestyles = iter(linestyle_options)
    for thresh_n in range(len(thresholds)):
        this_linestyle = next(linestyles)
        thresh_legs.append(Line2D([0,1],[0,1], linestyle = this_linestyle, color="black"))
        thresh_labels.append("ε > " + "{:.2e}".format(thresholds[thresh_n]))
    leg2 = ax.legend((thresh_legs), (thresh_labels), loc = 'center right')
    
    # Manually add the first legend back
    ax.add_artist(leg1)
    
    # More accessories
    ax.set_xscale("symlog", linthresh = 1, linscale = 0.4)
    ax.set_xlabel("Redshift (z)")
    ax.set_yscale('log')
    ax.set_ylabel("Mean # Mergers between z0 and z")
    ax.set_title("Cumulative Number of Major Mergers between z0 and z")
    
    # Unnecessarily Complicated Tick Marks
    stepsize = 1
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end, stepsize))
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    
    # Finish
    plt.savefig(filename + ".png")
    plt.show()
    
    
############################
# Prepare mms for plotting #
############################

def pdf_mms(mm_times_raw, thresholds, x_axis = 'z_nums'): # Note: assume mm_times_raw are binned

    # Rearrange, get rid of any empty lists, flatten all mms for all halos, and convert to z's if needed:
    if x_axis == 'z_nums':
        mm_times = [[[redshifts[int(100 - this_mm)] for this_halo in this_bin if len(this_halo[i]) != 0 for this_mm in this_halo[i]] for this_bin in mm_times_raw] for i in range(len(thresholds))]
    elif x_axis == 'snap_nums':
        mm_times = [[[this_mm for this_halo in this_bin if len(this_halo[i]) != 0 for this_mm in this_halo[i]] for this_bin in mm_times_raw] for i in range(len(thresholds))]
    
    return mm_times
    
############
# Plot PDF #
############

def plot_binned_pdf(values, thresholds, bins = [0, 1], x_axis = 'z_nums'):
    
    # Calculate histogram of values, plot as a distribution
    fig, ax = plt.subplots()
    color = iter(cm.jet(np.linspace(0,1,len(bins) - 1)))
    linestyle_options = ['-', '--', ':', '-.'][:len(thresholds)]
    bin_legend_handles = []
    
    # Plot once for each bin
    for bin_n in range(len(bins) - 1):
        current_color = next(color)
        linestyles = iter(linestyle_options)
        
        # Plot once for each threshold
        for i, values_in_this_threshold in enumerate(values): # length is 2
            this_linestyle = next(linestyles)
            norm_factor = len(values_in_this_threshold[bin_n])
            hist = np.histogram(values_in_this_threshold[bin_n])
            bin_centers = (hist[1][:-1] + hist[1][1:]) / 2
            ax.plot(bin_centers, hist[0]/norm_factor, linestyle = this_linestyle, color = current_color)
            
            # Add a legend handle for the first threshold
            if i == 0:
                bin_legend_handles.append(mpatches.Patch(color=current_color, label="bin " + str(bin_n + 1) + ": (" + "{:.2e}".format(bins[bin_n]) + " to " + "{:.2e}".format(bins[bin_n+1]) + ")"))
    
    # Set customized labels and titles
    if x_axis == 'z_nums':
        ax.set_xlabel("Redshift of LMM")
        ax.set_xscale("symlog", linthresh = 1, linscale = 0.4)
    elif x_axis == 'snap_nums':
        ax.set_xlabel("snapnum of LMM")
    if bins != []:
        ax.set_title("PDF of Major Merger Times in range: " + "{:.1e}".format(bins[0]) + " to " + "{:.1e}".format(bins[-1]))
    elif bins == [0, 1]:
        ax.set_title("PDF of Major Merger Times (binned)")
    ax.set_ylabel("Probability")
    
    # Fancy legend
    leg1 = ax.legend(handles = bin_legend_handles, loc='upper right')
    
    # Add second legend for the linestyles.
    # leg1 will be removed from figure
    thresh_legs = []
    thresh_labels = []
    linestyles = iter(linestyle_options)
    for thresh_n in range(len(thresholds)):
        this_linestyle = next(linestyles)
        thresh_legs.append(Line2D([0,1],[0,1], linestyle = this_linestyle, color="black"))
        thresh_labels.append("ε > " + "{:.2e}".format(thresholds[thresh_n]))
    leg2 = ax.legend((thresh_legs), (thresh_labels), loc = 'center right')
    
    # Manually add the first legend back
    ax.add_artist(leg1)
    
    # Unnecessarily Complicated Tick Marks
    stepsize = 1
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end, stepsize))
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
        
    return fig
    
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

def plot_hist_dist(data, data_names, filename = "new_plot", normalized = False):
    # Note: assume data is two dimensional (with some number of rows)
    
    fig, ax = plt.subplots()
    linestyles = iter(['-', '--', ':', '-.'])
    
    # Loop through each row of data
    for i, this_subset in enumerate(data):
        color = iter(cm.jet(np.linspace(0,1, len(data[i]))))
        this_linestyle = next(linestyles)
        
        # Loop through each statistic in that row
        for j, this_stat in enumerate(this_subset):
            this_color = next(color)
            hist = np.histogram(this_stat)
            bin_centers = ((hist[1][:-1] + hist[1][1:]) / 2)
            #if i == 0:
            #    norm_factor = max(hist[0])
            norm_factor = hist[0][0]
            if normalized == True:
                ax.plot(bin_centers, hist[0]/norm_factor, color = this_color, linestyle = this_linestyle, label = data_names[i][j])
            else:
                ax.plot(bin_centers, hist[0], color = this_color, linestyle = this_linestyle, label = data_names[i][j])

    ax.legend()
    ax.set_xlabel("Value")
    if normalized == True:
        ax.set_ylabel("Probability")
    else:
        ax.set_ylabel("Count")
    return fig, ax
    
############################
# Candidates vs. Threshold #
############################

def calc_candidates_vs_threshold(fossil_group_idx, thresholds, nbins, normalized = False):
    # Note: assume fossil_group_idx is binned_fossil_groups
    
    rearranged_fg_idx = [[[this_halo[i] for this_halo in this_bin if this_halo[i] is not None] for this_bin in fossil_group_idx] for i in range(len(thresholds))]
    
    #for each bin
    if normalized == True:
        tot_norm_factor = sum([len(fossil_group_idx[bin_n]) for bin_n in range(nbins)])
        tot_num_candidates = [(len(this_thresh[bin_n]) + len(this_thresh[bin_n + 1])) / tot_norm_factor for bin_n in range(nbins - 1) for this_thresh in rearranged_fg_idx]
        binned_norm_factors = [len(this_bin) for this_bin in fossil_group_idx]
        binned_num_candidates = [[len(this_thresh[bin_n]) / binned_norm_factors[bin_n] for this_thresh in rearranged_fg_idx] for bin_n in range(nbins)]
    else:
        tot_num_candidates = [len(this_thresh[bin_n]) + len(this_thresh[bin_n + 1]) for bin_n in range(nbins - 1) for this_thresh in rearranged_fg_idx]
        binned_num_candidates = [len(this_thresh[bin_n]) for this_thresh in rearranged_fg_idx for bin_n in range(nbins)]
    
    return tot_num_candidates, binned_num_candidates

#####################################
# Candidates vs. Threshold: plot it #
#####################################

def plot_candidates_vs(thresholds, tot_num_candidates, binned_num_candidates, bins = [], bin_labels = [], normalized = False, plot_tots = True, fig = None, ax = None, cust_legend = [], vert_line = None, hor_line = None, **kwargs):
    # Finally, plot them
    if ax == None:
        fig, ax = plt.subplots()
    else:
        fig = fig
        ax = ax
        
    color=iter(cm.jet(np.linspace(0,1,len(binned_num_candidates))))
    bin_legend_handles = []
    
    # Plot the totals
    if plot_tots == True:
        ax.plot(thresholds, tot_num_candidates, color = "lime", label = "all bins", **kwargs)
        bin_legend_handles.append(mpatches.Patch(color= "lime", label= "All candidates (bin 1 + bin 2)"))
    
    # Plot the individual bins
    for i, this_bin in enumerate(binned_num_candidates):
        current_color = next(color)
        ax.plot(thresholds, this_bin, color = current_color, **kwargs)
        if bin_labels != []:
            bin_legend_handles.append(mpatches.Patch(color=current_color, label = bin_labels[i]))
        else:
            bin_legend_handles.append(mpatches.Patch(color=current_color, label="bin " + str(i + 1)  + ": (" + "{:.1e}".format(bins[i]) + " to " + "{:.1e}".format(bins[i+1]) + ")"))
    
    # If desired, plot vertical and horizontal lines
    if vert_line is not None:
        ax.axvline(vert_line, color = 'black', linestyle = '-.', linewidth = 1.5)
    if hor_line is not None:
        ax.axhline(hor_line, color = 'black', linestyle = ':')
    
    # Accesorize
    ax.set_xscale('log') # log or linear
    ax.set_xlabel(r"Threshold ($h^{-1}M_\odot$)")
    if normalized == True:
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of Fossil Group Candidates")
    else:
        ax.set_ylabel("Number of Fossil Group Candidates")
    
    # Fancy legend
    leg1 = ax.legend(handles = bin_legend_handles, loc='upper left')
    if cust_legend != []:
        legs = cust_legend[0]
        labels = cust_legend[1]
        leg2 = ax.legend((legs), (labels), loc = 'center left')
    ax.add_artist(leg1)
    
    return fig, ax

##################
# Michael's Code #
##################

#thresholds = [10**12]#[6*10**11]
#thresholds_are_absolute = True
## Find and bin the halos
#my_bins = [10**13, 10**13.4, 10**15]
#bin_halo_idx = help_func.bin_halos(forest_table, snap, my_bins)
## Build main progenitor branches over all bins
#binned_timesteps, binned_masses, binned_main_prog_list, binned_maj_mergers, binned_mm_times, binned_LMMs, binned_LMM_times, binned_fossil_groups = help_func.track_evol_binned(bin_halo_idx, my_bins, redshifts, progenitor_idx, forest_masses, forest_snap_nums, thresholds, thresholds_are_absolute, x_axis = xaxis)
#print(len(binned_mm_times))
#print(len(binned_mm_times[0]))
#print(len(binned_mm_times[0][0]))

## Now, count them!
#z_threshold = 1
#nbins = 2
#nthresholds = 1
#candidate_counts = np.zeros((nbins, nthresholds), dtype=np.uint32)
#for i in range(nbins):
#    for j in range(nthresholds):
#        print("for bin ", i, " and threshold ", j, " binned_mm_times is: ", binned_mm_times[i][j])
#        mm_times = binned_mm_times[i][j][0]
#        print("mm_times: ", mm_times)
#        for k in range(len(mm_times)):
#            #print("for mm_times number ", k, " len(mm_times[", k, "]): ", len(mm_times[k]), " and mm_times[", k, "][0]: ", mm_times[k][0])
#            is_candidate = (len(mm_times[k]) == 0) or (mm_times[k][0] > z_threshold)
#            if is_candidate:
#                candidate_counts[i][j] += 1
#            print("End of the line")
#print(candidate_counts)
