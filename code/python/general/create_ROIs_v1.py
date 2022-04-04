"""
Download and visualise Schaefer's (2018) 2mm functional atlas with 200 areas (in MNI space but in xyz coordinates, has to be because MNI has negative indices). These areas have been labelled to belong to a particular network based on Yeo et al. (2011; Abbreviations, including the names of Yeo et al.'s (2011) networks, can be found here: https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering/1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels/Yeo2011_7networks_N1000.split_components.glossary.csv; e.g. "Cont" network stands for "control").
Import Long et al.'s (2020) coordinates (in MNI space and in MNI coordinates) in which a significant difference in resting-state functional connecivity was found between healthy and MDD patients (tsv file created by myself).
See in which areas of Schaefer's atlas these coordinates lie, and visualise these specific areas.
"""

## Import packages

from processing.conversion_functions import get_continuous_cmap, find_atlas_ROI_label_of_coord

# Create folders that don't exist yet
for folder in [vars.Schaefer_200_2mm_figs_filepath, vars.Schaefer_200_2mm_ROIs_filepath, vars.Schaefer_200_2mm_ROIs_figs_filepath]:
    if not folder.exists():
        folder.mkdir(exist_ok=False)  # Setting exist_ok=False makes sure pathlib throws a warning if the directory already exists

## Parameters


## Download Schaefer (2018) atlas using nilearn

# Download atlas
# nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2, data_dir=data_reference_path, base_url=None, resume=True, verbose=1) # n_rois=400 and resolution_mm=1 is the default
# Load data
Schaefer_200_2mm_img = nib.load(vars.Schaefer_200_2mm_data_filepath.as_posix())  # Load the image
Schaefer_200_2mm_img.shape
Schaefer_200_2mm_img.affine
Schaefer_200_2mm_data = Schaefer_200_2mm_img.get_fdata()

## Turn Schaefer et al.'s (2018) atlas with 200 regions into a dictionary with ROI numbers as keys and ROI labels as values
# Open file
with open(vars.Schaefer_200_2mm_order_filepath, 'r') as file:
    Schaefer_200_2mm_txt = file.read()

# Create dictionary with ROI numbers as keys and ROI labels as values
Schaefer_200_labels = {}
for line in Schaefer_200_2mm_txt.split("\n"):  # Split file into lines of each network
    split_line = line.split("\t")
    Schaefer_200_labels[int(split_line[0])] = split_line[
        1]  # Add to dictionary; last four elements in each line are x,y,z,t I think
    # print("split_line: {}".format(split_line))
Schaefer_200_labels[0] = float("nan")  # add one entry for 0 -> no ROI found

## Long (2020) ROIs
# Import tsv with Long (2020) coordinates which differed significantly in terms of dynamic RSFC
Long_2020_data = pd.read_csv(vars.Long_2020_filename, sep='\t')

## Find the coordinates which belong to the DMN
print("There are {} coordinates in total that are significantly different between controls and MDD patients.".format(
    Long_2020_data.shape[0]))

## Find the corresponding ROI label in the Schaefer2018 atlas for each of Long et al.'s (2020) identified coordinates
ROI_labels_Long_2020_coords_all = pd.DataFrame(list(
    map(functools.partial(find_atlas_ROI_label_of_coord, atlas_data=Schaefer_200_2mm_data,
                          atlas_labels=Schaefer_200_labels, study_coords_xyz=Long_2020_data),
        list(range(Long_2020_data.shape[0])))), columns=["ROI_nr", "ROI_label", "nr_voxels", "MNI x", "MNI y", "MNI z"])
# Get ROIs belonging to the DMN (contain "Default" in their name) and only extract unique labels - some coordinates may belong to the same ROI
ROI_labels_Long_2020_coords_DMN = ROI_labels_Long_2020_coords_all.loc[
    ROI_labels_Long_2020_coords_all[ROI_labels_Long_2020_coords_all["ROI_label"].str.contains("Default", na=False)][
        "ROI_label"].drop_duplicates().index]  # na = False makes sure that NaN values are filled by False so that no error is thrown during indexing; more elaborate way of indexing because the x,y,z coordinates are not be the same for all of Long et al.'s (2020) coordinates though they do belong to the same ROI; we thus want to drop duplicate ROIs

## Save dictionary with ROI labels to txt file
pd.DataFrame.from_dict(ROI_labels_Long_2020_coords_DMN).to_csv(vars.ROI_output_filepath, sep='\t', encoding='ascii', index=False, header=True)


## Plot complete atlas
Schaefer_200_2mm_plot = nilearn.plotting.plot_roi(Schaefer_200_2mm_img, title="Schaefer (2018) 200parc atlas", cmap = vars.cmap_Schaefer_200_2mm, colorbar = False, output_file=pathlib.Path.joinpath(
    vars.Schaefer_200_2mm_figs_filepath, "Schaefer2018_200Parcels_7Networks_2mm_complete_atlas.png").as_posix())

## Creating nifti mask and plotting it for each ROI, as well as plotting all ROIs together in labelled plot
# Only keep ROI regions that are part of the DMN & identified by Long et al. (2020)
Schaefer_200_2mm_DMN = np.zeros(Schaefer_200_2mm_data.shape)
for ROI_nr in ROI_labels_Long_2020_coords_DMN["ROI_nr"]:
    Schaefer_200_2mm_DMN += np.where((Schaefer_200_2mm_data == ROI_nr), Schaefer_200_2mm_data, 0)

plt.close('all')

## Load and plot MNI152 template
MNI152_brain = nib.load(vars.MNI152_brain_filename)  # Load MNI152 template
# MNI152_brain.get_fdata().shape
# MNI152_brain = nilearn.datasets.load_mni152_template(resolution=2) # Don't use standard template because it doesn't have the same dimensions as the ROI masks (91, 109, 91)

# Plot template to initialize extracted ROIs plot
nr_cuts = (3, 3, 3)  # Number of cuts to perform if display_mode = "mosaic"
Schaefer_200_2mm_DMN_plot = nilearn.plotting.plot_roi(MNI152_brain, title="Schaefer (2018) 200parc DMN", cmap="Greys",
                                                      colorbar=False, draw_cross=False, cut_coords=nr_cuts,
                                                      display_mode='mosaic')

## Create and save ROI masks as nifti images; plot them as overlays on top of the MNI152 template.
ROI_masks = {}
ROI_masks_filenames = {}
ROI_mask_centre = {}

# Loop through ROIs and create mask, nifti file from mask, png image of mask, and add to extracted ROIs plot
for ROI in range(len(ROI_labels_Long_2020_coords_DMN)):
    # Get ROI number
    ROI_nr, ROI_label, nr_voxels = ROI_labels_Long_2020_coords_DMN.loc[
        ROI_labels_Long_2020_coords_DMN.iloc[ROI].name, ["ROI_nr", "ROI_label", "nr_voxels"]]
    print("ROI number: {}".format(ROI_nr))
    mask = np.where(Schaefer_200_2mm_data == ROI_nr, 1, 0)  # Create binary mask

    ## Save as nifti image with affine transformation from original Schaefer atlas
    filename_mask = "Schaefer2018_200_2mm_mask_{}".format(ROI_label)

    # Save to dictionary
    ROI_masks[ROI_nr] = mask  # Save mask in dictionary
    ROI_masks_filenames[ROI_nr] = pathlib.Path.joinpath(vars.Schaefer_200_2mm_ROIs_filepath, filename_mask).with_suffix(".nii").as_posix()

    # Nifti image
    nib.Nifti1Image(mask, affine=Schaefer_200_2mm_img.affine).to_filename(ROI_masks_filenames[ROI_nr])

    # Png image of mask only
    Schaefer_200_2mm_mask_plot = nilearn.plotting.plot_roi(ROI_masks_filenames[ROI_nr], title="Schaefer (2018) {}".format(ROI_label), cmap=get_continuous_cmap(np.repeat(
        vars.ROI_colours[ROI], 10)),
                                                           colorbar=False, draw_cross=False, output_file = pathlib.Path.joinpath(
            vars.Schaefer_200_2mm_ROIs_figs_filepath, filename_mask).with_suffix(".png").as_posix())

    # Plot mask as overlay
    Schaefer_200_2mm_DMN_plot.add_overlay(ROI_masks_filenames[ROI_nr], cmap=get_continuous_cmap(
        np.repeat(vars.ROI_colours[ROI],
                  10)))  # Create colour map from single colour to ensure that ROI label colours match the ROI mask; It doesn't matter what the second argument to get_continuous_cmap is as it simply determines the number of ticks, but it cannot be 1.

# Change size of plot
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 6)

# Save final plot to png
Schaefer_200_2mm_DMN_plot.savefig(
    pathlib.Path.joinpath(vars.Schaefer_200_2mm_ROIs_figs_filepath, "Schaefer_200_2mm_all_ROIs.png"), dpi="figure")

## Create labels for ROIs in figure
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=np.repeat(0, len(ROI_labels_Long_2020_coords_DMN)),
    y=np.linspace(1, 0, len(ROI_labels_Long_2020_coords_DMN)),
    mode="markers+text",
    name="",
    marker_symbol="square",
    marker={
        "color": vars.ROI_colours,
        "size": 10
    },
    text=["<b>{}</b>".format(label) for label in ROI_labels_Long_2020_coords_DMN["ROI_label"]],  # Make bold <b>87</b>
    textposition="middle right",
    textfont=dict(
        family=vars.font_family,
        size=18,
        color=vars.ROI_colours
    )
))
fig.update_layout(height=500, width=500,
                  # showlegend=False,
                  plot_bgcolor="white",  # Change background colour
                  )

fig.update_xaxes(range=[-.2, 1.5], visible=False)
fig.update_yaxes(range=[-.2, 1.1], visible=False)  # title_text="Hz",

# Save to .png
fig.write_image(pathlib.Path.joinpath(vars.Schaefer_200_2mm_ROIs_figs_filepath,
                                      "Schaefer2018_200_2mm_DMN_ROI_labels.png").as_posix())


