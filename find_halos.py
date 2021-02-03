# Find the n smallest masses in a given forest_table

import numpy as np

def find_halos(forest_table, sn, quant = 0, mass_range = [], extremum = ''):
        
        # Extract the mass values (and associated halo id's) from desired snapshot
        masses_sn = np.array((2.7*10**9)*(forest_table.mass.loc[forest_table.snap_num == sn]))
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

def bin_halos(forest_table, sn, bins):
    # Extract the mass values from the desired snapshot
    masses_sn = np.array((2.7*10**9)*(forest_table.mass.loc[forest_table.snap_num == sn]))
    halos_sn = np.array(forest_table.halo_id.loc[forest_table.snap_num == sn])
    
    # Assign a bin number to each halo
    bin_idx = np.digitize(masses_sn, bins)
    
    # Build list of all halos associated with each bin
    halo_idx = []
    for j in range(1, len(bins)): # Why start at 1? Because the bin numbers go from 1 to len(bins)
        halo_idx.append(halos_sn[bin_idx == j]) 
   
    return halo_idx

def get_list_dims(my_list):
    print("The list is: ", my_list, "\nIt has length: ", len(my_list))
    for i in np.arange(len(my_list)):
        print("Length of row ", i, " is ", len(my_list[i]))

def michaels_bin_halos(forest_table, sn, bins):
        # Extract the mass values from desired snapshot
        masses_sn = np.array((2.7*10**9)*(forest_table.mass.loc[forest_table.snap_num == sn]))
        halos_sn = np.array(forest_table.halo_id.loc[forest_table.snap_num == sn])
        bin_idx = np.digitize(masses_sn, bins)   # note that: bins[i-1] <= x < bins[i]
        halo_idx = []
        for i in range(1, len(bins)):
            halo_idx.append(halos_sn[bin_idx == i])
        return halo_idx

def old_bin_halos(forest_table, sn, bins):
        
        # Extract the mass values (and associated halo id's) from desired snapshot
        masses_sn = np.array((2.7*10**9)*(forest_table.mass.loc[forest_table.snap_num == sn]))
        halos_sn = np.array(forest_table.halo_id.loc[forest_table.snap_num == sn])
        
        # Sort masses_sn in order of smallest to largest values
        mass_order = np.argsort(masses_sn)
        sorted_halo_ids = halos_sn[mass_order]
        sorted_masses = masses_sn[mass_order]
     
        # Put the halo_ids and masses in bins
        halo_idx = []
        
        for i in np.arange(len(bins) - 1):  
            halo_idx.append(np.array(sorted_halo_ids[np.logical_and(sorted_masses >= bins[i], sorted_masses <= bins[i+1])]))  
        
        return halo_idx

def old_find_halos(quant, sn, extremum, forest_table):
        
        ft_masses_sn = (2.7*10**9)*np.array(forest_table.mass.loc[forest_table.snap_num == sn]) #proxy for masses, changes throughout loop
        shift = len(np.array(forest_table['mass'])) - len(ft_masses_sn)
        halo_idx = np.empty(0)
        halo_values = np.empty(0)

        #Find the minimum, add it to lists, rm from ft_masses_x, repeat
        for i in np.arange(quant):
            if extremum == 'max':
                current_idx = np.argmax(ft_masses_sn)
            elif extremum == 'min':
                current_idx = np.argmin(ft_masses_sn)
            halo_idx = np.append(halo_idx, np.int(current_idx + shift))
            halo_values = np.append(halo_values, ft_masses_sn[current_idx])
            ft_masses_sn = np.delete(ft_masses_sn, current_idx, None)
            i = i + 1
        
        #print("Halo index is: ", halo_idx)
        #print("Halo values are: ", halo_values)
        #print("Halo id's are: ", forest_table['halo_id'][halo_idx])
        return halo_idx, halo_values
