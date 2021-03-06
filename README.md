# `photo_db`
## A database and basic data wrangling of photosynthesis-related traits.
### What's in the database?
I'm currently populating the database with traits that have traditionally been linked to photosynthesis in general and CAM in particular. I am also adding the hypothesized photosynthetic pathway for each taxon when/as noted by the author(s). Because the study of the evolution of CAM requires us to expand sampling outside just our focal CAM groups, this database will not be exclusive to CAM taxa. The traits included should be (relatively) easily observed/measured across most taxa. While this may be a bit of a judgement call, there are certainly a suite of traits (e.g. leaf thickness, carbon isotope ratios, titratable acidity) that the community has decided to record when studying CAM. Furthermore, these traits have been recorded in *healthy* or *wild* plants; not those subject to experimentation. I do not have immediate plans for incorporating the every growing of transcriptomic work here because I believe that every experiment is too unique to be generally comparable. However, presence/absence of CAM related loci (e.g. ppc-1E1c) may be added in the future. 

### Access
The data can be accessed in a two main ways:
  1. Downloaded in the raw csv from this repository
  2. Importing through the python package `photo_db` that I created for doing some wrangling, imputation, and exploration of the data.

### Curation
This database is currently curated by me but I'm always looking for help. CAM is found in many species rich clades that are constantly undergoing taxonomic revisions (I'm looking at you orchids), so please report issues when you find them so I can provide the most up to date information. With regard to the data themselves, I have been collecting them by a combination of automated PDF table reading with [Tabula](http://tabula.technology) and manual entry and correction. I try to double check them all, but any discrepancies between the database and original publications should be posted in issues. Finally, I strongly believe that **this database should contain open data**, and so will only include publicly available data with their sources. If you would like to contribute data I only ask that the source is included, and if the data are not published that you provide some sort of qualification. If you are interested in contributing your data or can point me in the direction of existing published data please (get in touch with me at ian.gilman@yale.edu.

### Legend
All features ending in 'sd' are standard deviations of their respective variables. Units are included when applicable. 
*	`Family`
* `Subfamily`
* `Tribe`
* `Subtribe`
* `Genus`
* `Species`
* `Subspecies`
* `BS_area_um2`: bundle sheath area in square micrometers
* `FM_DM`
* `FM_DM_sd`:
* `H+_ev_mumol_g-1_FW`: Evening hydrogen ion concentration per gram fresh mass
* `H+_ev_mumol_g-1_FW_sd`
* `H+_mo_mumol_g-1_FW`: Morning hydrogen ion concentration per gram fresh mass
* `H+_mo_mumol_g-1_FW_sd`
* `Habit`: Overall plant growth form
* `IAS`: Intercellular air space as a percentage
* `IAS_sd`
* `Pathway`: Type of photosynthesis
* `SLA_cm2_g-1`: Specific leaf area
* `SLA_cm2_g-1_sd`
* `air_channel_area_um2`:
* `air_channel_area_um2_sd`
* `chlorenchyma_diameter_um`
* `chlorenchyma_diameter_um_sd`
* `chlorenchyma_hydrenchyma_ratio`: Chlorenchyma diameter:hydrenchyma diameter
* `chlorenchyma_hydrenchyma_ratio_sd`
* `chlorenchyma_vertical_thickness_um`
* `chlorenchyma_vertical_thickness_um_sd`
* `dC13`: delta carbon 13 isotope ratio
* `dH+`: Change in hydrogen ion concentration from evening to morning (`H+_ev_mumol_g-1_FW` - `H+_mo_mumol_g-1_FW`)
* `dH+_significance`: Significance of hydrogen ion concentration from evening to morning
* `distance_between_BS_um`
* `hydrenchyma_vertical_thickness_um`
* `hydrenchyma_vertical_thickness_um_sd`
* `leaf_thickness_mm	leaf_thickness_mm_sd`
* `leaf_thickness_um	leaf_thickness_um_sd`
* `mesophyll_abaxial_area_um2`
* `mesophyll_abaxial_length_um`
* `mesophyll_adaxial_area_um2`
* `mesophyll_adaxial_length_um`
* `mesophyll_cell_area_um2`
* `plant_part`: Type of tissue measurement was taking from
* `Source`: Source of data (usually peer reviewed journal article)
* `transpiration_ratio` : Ratio of dark to light transpiration rate

Note that there is currently `plant_part`, `Tissue`, and `tissue` that need to be merged.
