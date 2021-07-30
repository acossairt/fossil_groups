# fossil_groups
Argonne National Laboratory - Cosmological Physics and Advanced Computing group (CPAC)

Aurora Cossairt

July 30th, 2021

Welcome! This repo holds notebooks, python scripts, and images relevant to our exploration of fossil group candidates in the Last Journey. 
Our paper, "Cosmo-Paleontology: Statistics of Fossil Groups in a Gravity-Only Simulation" is in preparation.

If you'd like to see the source code for any of the figures in the paper, here are the relevant notebooks
- Figure 2: `lj_abundances.ipynb`
- Table 1: `lj_abundances.ipynb`
- Figure 3: `lj_luminous_mergers.ipynb`
- Figure 4: `lj_mass_evolution.ipynb`
- Figure 5: `lj_zfracs.ipynb`
- Figure 6: `lj_diff_MAHs.ipynb` (which references materials in the diffmah subdirectory)
- Figure 7: `lj_concentration_relaxation.ipynb`
- Figure 8: `lj_concentration_relaxation.ipynb`
- Table 2: `lj_substructures_stats.ipynb`
- Figure 9: not in this repo -- reach out to @michaelbuehlmann
- Figure 10: set up is in `maps.ipynb` and `locate_FG_Candidates.ipynb`,
             images associated with the cosmic web map construction are in the `map-visualizations` subdirectory,
             for final figures, reach out to @michaelbuehlmann
- Figure 11: set up is in `maps.ipynb` and `locate_FG_Candidates.ipynb`,
             images associated with the cosmic web map construction are in the `map-visualizations` subdirectory,
             for final figures, reach out to @michaelbuehlmann
- Figure 12: `lj_haccytrees_validation.ipynb`
- Figure 13: `lj_haccytrees_validation.ipynb`
All of these notebooks rely on the helper functions hosted in `help_func_haccytrees.py`. All of the resulting figures (and some images that didn't
make it into the paper) are stored in the `full_lj_plots` subdirectory.

## About our sample
We primarily examined fossil group candidates hosted in three narrow mass bins:
- 10^13.0 - 10^13.05 h^{-1} M_\odot: 269,358 FG candidates
- 10^13.3 - 10^13.35 h^{-1} M_\odot: 36,181 FG candidates
- 10^13.6 - 10^13.65 h^{-1} M_\odot: 2454 FG candidates

In order to compare the features of these candidates to the full sample of halos, we created "random samples" in the same three narrow mass bins.
Each "random sample" is the same size (i.e. contains the same total number of halos) as the FG sample for the same mass bin. However, this "random sample" 
is a random mix of FG candidates and non-candidates. To see how we chose the halos to include in each random sample, check out `howto_create_random_sample.ipynb`.
@michaelbuehlmann then created a new forest just for halos in this sample
- 10^13.0 - 10^13.05 h^{-1} M_\odot: 269,358 halos; 231,618 non-candidates + 37,740 FG candidates (4,171 of which are QH candidates, 33,569 are just FGs)
- 10^13.3 - 10^13.35 h^{-1} M_\odot: 36,181 halos; 34,808 non-candidates + 1,373 FG candidates (9 of which are QH candidates, 1,364 are just FGs)
- 10^13.6 - 10^13.65 h^{-1} M_\odot: 2454 halos; 2441 non-candidates + 13 FG candidates (0 of which are QH candidates)

## How-to notebooks
This repo includes two short "how-to" tutorials: `howto_create_random_sample.ipynb` (discussed above) and `howto_find_FG_candidates.ipynb`. 
The latter shows how to find FG candidates for any target index of halos. This is a routine used in several other notebooks, but is explained here more thoroughly.

## Other statistics in preparation
The following notebooks (and some associated subdirectories) are for statistics which have not yet been finalized:
- `ljsv_two-point-correlations.ipynb` and the `two-point-correlations` subdirectory. These stats have so far only been run on the smaller version of LJ.
- `lj_gap_statistics.ipynb` (in collaboration with @evevkovacs)
