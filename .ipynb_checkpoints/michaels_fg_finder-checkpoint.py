def fg_finder(step):
    f = f"{base}.{step:03d}.hdf5"
    forest, progenitor_array = haccytrees.mergertrees.read_forest(
        f,
        simulation="LastJourney",
    )
    target_mask = forest["snapnum"] == last_snap
    target_mask &= forest["tree_node_mass"] > z0_masslim[0]
    target_mask &= forest["tree_node_mass"] < z0_masslim[1]
    target_idx = forest["halo_index"][target_mask]

    mainbranch_index = haccytrees.mergertrees.get_mainbranch_indices(
        forest, simulation="LastJourney", target_index=target_idx
    )
    active_mask = mainbranch_index > 0 # I used != -1

    # get indices to secondary progenitors (main mergers)
    main_merger_index = haccytrees.mergertrees.get_nth_progenitor_indices(
        forest, progenitor_array, target_index=mainbranch_index[active_mask], n=2
    )

    # the index will be negative if there's no merger, mask those out
    merger_mask = main_merger_index >= 0
    
    ######################
    # Identical up to here
    ######################
    
    _merger_mass = forest["tree_node_mass"][main_merger_index[merger_mask]] # ???
    #########################################################
    # Comparable to 
    # mainbranch_mass[active_mask] = forest['tree_node_mass'][mainbranch_index[active_mask]]
    # where active_mask = mainbranch_index != -1 (Is that the same?)
    #########################################################
    
    ##################################################################
    # This becomes what I call `mainbranch_merger` (why am I so bad at naming things??)
    # Also what's with the underscore, "_" in "_merger_mass"?
    ##################################################################
    
    merger_mass = np.zeros_like(main_merger_index, dtype=np.float32) 
    ############################################################
    # I did `mainbranch_merger = np.zeros_like(mainbranch_index, dtype=np.float32)` # (Is that the same shape?)
    ############################################################
    
    merger_mass[merger_mask] = _merger_mass # Same
    ##############################################
    # So this is like
    # merger_mass[main_merger_index >= 0] = forest["tree_node_mass"][main_merger_index[merger_mask]]
    ##############################################
    
    merger_mass_matrix = np.zeros_like(mainbranch_index, dtype=np.float32)
    ######################################################################
    # Aha! So actually it's this one that is like my `mainbranch_merger = np.zeros_like(mainbranch_index, dtype=np.float32)` 
    ######################################################################
    
    merger_mass_matrix[active_mask] = merger_mass
    #############################################
    # This is like my
    # `mainbranch_merger[active_mask] = forest['tree_node_mass'][main_merger_index]`
    #############################################

    # major merger mask with a relative threshold
    mm_mask = merger_mass_matrix > mm_thresh
    ########################################
    # Same as `mm_mask = mainbranch_mergers > threshold`
    # Or maybe `mm_mask = major_mergers > threshold`?
    ########################################

    # finding the last index
    last_mm_index = last_snap - np.argmax(mm_mask[:, ::-1], axis=1)
    ########################################################
    # Same except I use `major_mergers` instead of `mm_mask`
    ########################################################

    last_mm_redshift = 1 / scale_factors[last_mm_index] - 1

    # mark all halos without any major merger with a last_mm_redshift of -1
    last_mm_redshift[~np.any(mm_mask, axis=1)] = -1

    fg_candidate_idx = target_idx[(last_mm_redshift >= 1.0) | (last_mm_redshift == -1)]
    ###########################################################
    # Interesting... I use
    # fg_merging_idx = target_idx[last_mm_redshifts > z_thresh]
    # So I guess the other is an or?
    # So the outcome is a full mask, and not just a list of fg candidates?
    # Or it's a list of both fgs and failed groups?
    # Cause I do `failures_mask = last_mm_redshifts == -1`
    ###########################################################

    fg_data_list = []
    for idx in fg_candidate_idx:
        fg_data_list.append(
            {
                k: forest[k][idx : idx + forest["branch_size"][idx]]
                for k in forest.keys()
            }
        )
    ######################################################
    # Instead of a list of fgs, Michael makes a dictionary
    # (which is actually pretty smart -- then you don't have
    # to rerun the whole sequence just to get the fgs info)
    ######################################################
    return fg_data_list