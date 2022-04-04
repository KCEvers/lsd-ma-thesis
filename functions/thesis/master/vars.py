## Variables
import plotly.express as px
from processing.conversion_functions import rgb_to_hex
import pathlib

# File patterns for building file paths
# pattern_output_ROI = "{pipeline}\sub-{subject}[\ses-{session}]\[{datatype<func|beh>|func}\][\{timeseries_or_figs}\][{type_of_timeseries}\]sub-{subject}[_ses-{session}]_task-{task}[_run-{run}]_{suffix}[{extension}]"  # Pattern of output files containing extracted timeseries
pattern_output_image = "{pipeline}\sub-{subject}[\ses-{session}]\[{datatype<func|beh>|func}\]sub-{subject}[_ses-{session}]_task-{task}[_run-{run}]_{suffix}[{extension}]"  # Pattern of output files containing extracted timeseries
# pattern_folder = "sub-{subject}[\ses-{session}]\[{datatype<func|beh>|func}\]"  # Pattern of folder names, used to create output file paths
# pattern_derivatives_anat = "{pipeline}\sub-{subject}[\{datatype<anat>|anat}]"  # Pattern of folder names, used to create output file paths
pattern_rawdata = "sub-{subject}[\ses-{session}][\{datatype<anat|func>|anat}]"  # Pattern of folder names, used to create output file paths
# pattern_derivatives_func = "{pipeline}\{timeseries_or_figs}[\{type_of_fig}\]\{mask_unstand_or_stand}\{raw_or_PC_unstand_or_stand}\sub-{subject}\ses-{session}\{datatype<func|beh|anat>|func}"  # Pattern of folder names, used to create output file paths
# pattern_derivatives = "{pipeline}\{timeseries_or_figs}[\{type_of_fig}\]\{mask_unstand_or_stand}\{raw_or_PC_unstand_or_stand}\sub-{subject}[\{datatype<func|anat>|func}][\ses-{session}]"  # Pattern of folder names, used to create output file paths
# pattern_derivatives_func = "{pipeline}\sub-{subject}[\ses-{session}]\[{datatype<func|beh>|func}\][\{timeseries_or_figs}\][{type_of_timeseries}\][{type_of_fig}\]"  # Pattern of folder names, used to create output file paths
pattern_derivatives = "{pipeline}\{timeseries_or_figs}[\{type_of_fig}]\{mask_unstand_or_stand}\{raw_or_PC_unstand_or_stand}\sub-{subject}[\ses-{session}][\{datatype<func|anat>|func}]"  # Pattern of folder names, used to create output file paths
pattern_derivatives_output = "{pipeline}\{timeseries_or_figs}[\{type_of_fig}]\{mask_unstand_or_stand}\{raw_or_PC_unstand_or_stand}\sub-{subject}[\ses-{session}][\{datatype}][\sub-{subject}_ses-{session}_task-{task}_run-{run}][_{concat}][_{suffix}{extension}]"  # Pattern of output files containing extracted timeseries

pipelines = ["preproc-rois", "PLRNN_analysis", "timeseries_analysis"]
type_of_figs = ["timeseries", "brain", "phase-space", "RP", "space-time-traj", "table_PCs", "stationarity", "d2"]

# File paths
path_to_datafolder = pathlib.Path("D:\\Data\\thesis\\rawdata")
path_to_derivatives = path_to_datafolder.with_name("derivatives")
project_dir = pathlib.Path().parent.resolve()  # Project directory; The command "pathlib.Path(__file__).parent.resolve()" only works if you are running a script, not in the console
data_reference_path = pathlib.Path.joinpath(project_dir, "data", "data_reference")
path_to_sim = pathlib.Path(path_to_derivatives, "simulations")
path_to_sim_entr = pathlib.Path(path_to_sim, "entropies")
path_to_sim_entr_lor = pathlib.Path(path_to_sim_entr, "Lorenz")
tiseanpath = "C:/Users/KyraEvers/Documents/Tisean_3.0.0/bin"

# Schaefer2018
Schaefer_2018_foldername = "schaefer2018"
Schaefer_200_2mm_data_filepath = pathlib.Path.joinpath(data_reference_path, Schaefer_2018_foldername,
                                                       "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")
Schaefer_200_2mm_order_filepath = pathlib.Path.joinpath(data_reference_path, Schaefer_2018_foldername,
                                                        "Schaefer2018_200Parcels_7Networks_order.txt")

Schaefer_200_2mm_figs_filepath = pathlib.Path.joinpath(data_reference_path, Schaefer_2018_foldername, "figs")
Schaefer_200_2mm_ROIs_filepath = pathlib.Path.joinpath(data_reference_path, Schaefer_2018_foldername, "ROIs")
Schaefer_200_2mm_ROIs_figs_filepath = pathlib.Path.joinpath(Schaefer_200_2mm_ROIs_filepath, "figs")
# Path to save figures of Schaefer2018 to
Schaefer_ROIs_df_filename = "Schaefer_200_2mm_ROIs_df.txt"  # Filename of dataframe with ROI numbers, labels, and coordinates
ROI_output_filepath = pathlib.Path.joinpath(Schaefer_200_2mm_ROIs_filepath,
                                            Schaefer_ROIs_df_filename)  # Path to save ROI dataframe (incl. nr, label, nr_voxels, and MNI coordinates) to

# Long2020
Long_2020_filename = pathlib.Path.joinpath(data_reference_path, "long2020", "Long_2020_full.tsv")

# MNI152
MNI152_brain_filename = pathlib.Path.joinpath(data_reference_path, "MNI152_T1_2mm",
                                              "MNI152_T1_2mm_Brain.nii.gz").as_posix()  # Brain-extracted MNI152 template (2mm)

# File names
prefix_mask = "Schaefer2018_200_2mm_mask"  # The prefix that the filename of each ROI mask starts with
# pattern_output_ROI = "sub-{subject}[\ses-{session}]\[{datatype<func|beh>|func}\]sub-{subject}[_ses-{session}]_task-{task}[_run-{run}]_{suffix}[{extension}]"  # Pattern of output files containing extracted timeseries

# Parameters
nr_PCs = 3
nr_timepoints = 217

# Figures
cmap_Schaefer_200_2mm = "Dark2"  # "terrain"
# ROI_colours = list(map(rgb_to_hex, px.colors.sequential.deep[4:12]))
# ROI_colours = px.colors.cyclical.Twilight[2:10] # Already in hex format
# ROI_colours = [px.colors.cyclical.Twilight[i] for i in [1, 2, 3, 5, 6, 7, 8, 9]] # Already in hex format
# ROI_colours = list(map(rgb_to_hex, px.colors.diverging.curl[4:12]))
# ROI_colours = list(map(rgb_to_hex, [px.colors.diverging.curl[i] for i in [1, 2, 3, 5, 7, 8, 9, 11]]))
# ROI_colours = [px.colors.sequential.Electric[i] for i in [1, 2, 3, 4, 5, 7, 8, 9]] # Already in hex format
# ROI_colours = list(map(rgb_to_hex, [px.colors.qualitative.Pastel[i] for i in [0, 1, 2, 3, 5, 6, 7, 8]]))
ROI_colours = list(map(rgb_to_hex, [px.colors.qualitative.Vivid[i] for i in [0, 1, 2, 3, 5, 6, 7, 8]]))

# Colours for recurrence plot
RP_colours = {"LSD": "amp", # "Burgyl" "Peach" # "Teal"
                "PLCB": "Teal" } # Blugrn

space_time_trajs = list(map(rgb_to_hex, [px.colors.qualitative.Prism[i] for i in [1, 2, 3, 4]]))

alpha_val_timeseries_error_bands = .2
line_width_timeseries_mean = 2.4

fig_timeseries_height = 1000
fig_timeseries_width = 1800
fig_phase_space_height = 1000
fig_phase_space_width = 1400
fig_phase_space_height_lorenz = 600
fig_phase_space_width_lorenz = 1400

fig_stationary_height = 1000
fig_stationary_width = 1800

fig_RP_height = 1000
fig_RP_width = 1000
fig_d2_height = 800
fig_d2_width=1800
margins_fig_stationary = dict(t=150, l=80, b=100, r=80)
margins_fig_stationary_subplot_fig = dict(t=110, l=80, b=100, r=110)
margins_fig_stationary_title = dict(t=0, l=0, b=100, r=0)

margins_fig_d2 = dict(t=100, l=0, b=100, r=0)
margins_fig_timeseries = dict(t=110, l=80, b=100, r=80)
margins_fig_timeseries_subplot_fig = dict(t=110, l=80, b=100, r=110)
margins_fig_timeseries_title = dict(t=0, l=0, b=100, r=0)
margins_fig_phase_space =  dict(t=50, r=10, l=10, b=100)# dict(t=10, r=300, l=300, b=300)
margins_fig_phase_space_title = dict(t=0, l=0, b=0, r=0)
margins_fig_phase_space_lorenz =  dict(t=10, r=0, l=0, b=0)# dict(t=10, r=300, l=300, b=300)
margins_fig_phase_space_title_lorenz = dict(t=10, l=0, b=0, r=0)
margins_fig_RP = dict(t=0, l=0, b=0, r=0)

font_family = "Raleway"
fig_timeseries_title_font_size = 28
fig_timeseries_subplot_title_font_size = 20
fig_timeseries_subplot_axis_title_font_size = 26
fig_timeseries_subplot_tick_font_size = 20
fig_timeseries_subplot_nr_ticks = 5
fig_timeseries_subplot_legend_font_size = 20

fig_complexities_suptitle_font_size = 16
fig_complexities_title_font_size = 12
fig_complexities_axis_title_font_size = 14

fig_stationary_title_font_size = 28
fig_stationary_subplot_title_font_size = 25
fig_stationary_subplot_axis_title_font_size = 22
fig_stationary_subplot_tick_font_size = 20
fig_stationary_subplot_nr_ticks = 5
fig_stationary_subplot_legend_font_size = 20
fig_stationarity_alpha_ci = .3


fig_RP_colorbar_ticks_font_size=14


fig_phase_space_title_font_size = 36
fig_phase_space_subplot_axis_title_font_size = 20
fig_phase_space_subplot_tick_font_size = 16
fig_phase_space_subplot_nr_ticks = 5
fig_phase_space_alpha_background = .45

fig_d2_suptitle_font_size = 16
fig_d2_title_font_size = 12
fig_d2_axis_title_font_size = 14
fig_d2_color_outside_scaling_region = "darkgray"
fig_d2_color_inside_scaling_region = "darkorange"
fig_d2_colour_d2_estimate = "LightSeaGreen"

grid_width = 1.5  # For timeseries plots
grid_color = px.colors.sequential.Greys[2]  # For timeseries plots

fig_PC_label_font_size = 15
# fig_PC_shade_factor = 0.35
fig_PC_shade_tint_factors = [.35, .05, .35]
fig_PC_shade_or_tint = ["shade", "shade", "tint"]
# fig_PC_tint_factor = 0.35
fig_PC_lwd = [2.4, 2.2, 2]
fig_unst_time_y_range_min = -15
fig_unst_time_y_range_max = 15
fig_st_time_y_range_min = -3
fig_st_time_y_range_max = 3

fig_unst_PC_y_range_min = -100
fig_unst_PC_y_range_max = 100
fig_st_PC_y_range_min = -4
fig_st_PC_y_range_max = 4

fig_phase_space_marker_size = 5
fig_phase_space_line_width = 4

# Dictionary for phase space plot to specify which angles need to be shown of the 3D plot
# The eye vector determines the position of the camera. The default is $(x=1.25, y=1.25, z=1.25)$.
# The up vector determines the up direction on the page. The default is $(x=0, y=0, z=1)$, that is, the z-axis points up.
# The projection of the center point lies at the center of the view. By default it is $(x=0, y=0, z=0)$.

fig_phase_space_dict_general = dict(  # General dictionary
    aspectmode='cube',  # 'manual',
    aspectratio=dict(x=1, y=1, z=1),
    xaxis=dict(
        title=dict(
            font=dict(
                size=fig_phase_space_subplot_axis_title_font_size),
            text=r'PC 1'),
        nticks=fig_phase_space_subplot_nr_ticks,
        range=[fig_unst_PC_y_range_min, fig_unst_PC_y_range_max],  # Specify axis range
        tickfont=dict(
            size=fig_phase_space_subplot_tick_font_size),
        backgroundcolor="rgb(200, 200, 230)",
        gridcolor=grid_color,
        showbackground=True,
        zerolinecolor=grid_color,
    ),
    yaxis=dict(
        title=dict(
            font=dict(
                size=fig_phase_space_subplot_axis_title_font_size),
            text=r'PC 2'),
        nticks=fig_phase_space_subplot_nr_ticks,
               range=[fig_unst_PC_y_range_min, fig_unst_PC_y_range_max],  # Specify axis range
               tickfont=dict(
                   size=fig_phase_space_subplot_tick_font_size),
        backgroundcolor="rgb(230, 200, 230)",
        gridcolor=grid_color,
        showbackground=True,
        zerolinecolor=grid_color,
               ),
    zaxis=dict(
        title=dict(
            font=dict(
                size=fig_phase_space_subplot_axis_title_font_size),
            text=r'PC 3'),
               nticks=fig_phase_space_subplot_nr_ticks,
               range=[fig_unst_PC_y_range_min, fig_unst_PC_y_range_max], # Specify axis range
               tickfont=dict(
                   size=fig_phase_space_subplot_tick_font_size),
        backgroundcolor="rgb(230, 230, 200)",
        gridcolor=grid_color,
        showbackground=True,
        zerolinecolor=grid_color,
               )
)

fig_phase_space_camera_dict_general = dict(  # General dictionary for camera
    center=dict(
        x=0, y=0, z=0),
    eye=dict(
        x=2, y=2, z=0.1),
    up=dict(
        x=0, y=0, z=1)
)


#     # ticktext=['TICKS', 'MESH', 'PLOTLY', 'PYTHON'],
#     # tickvals=[0, 50, 75, -50]),
#     # yaxis=dict(
#     #     nticks=5, tickfont=dict(
#     #         color='green',
#     #         size=12,
#     #         family='Old Standard TT, serif', ),
#     #     ticksuffix='#'),
#     # zaxis=dict(
#     #     nticks=4, ticks='outside',
#     #     tick0=0, tickwidth=4), ),


fig_phase_space_camera_dict = dict(
    xz={  # y-z plane
        **fig_phase_space_camera_dict_general,
        # center= dict(
        #     x=0, y=0, z=0),
        "eye": dict(
            x=.1, y=2.75, z=.1),
    },
    xy={  # y-z plane
        **fig_phase_space_camera_dict_general,
        # center= dict(
        #     x=0, y=0, z=0),
        "eye": dict(
            x=.1, y=.1, z=2.75),
    },
    yz={  # y-z plane
        **fig_phase_space_camera_dict_general,
        # center= dict(
        #     x=0, y=0, z=0),
        "eye": dict(
            x=2.75, y=.1, z=.1),
    },
    left={
        **fig_phase_space_camera_dict_general,
        # center= dict(
        #     x=0, y=0, z=0),
        "eye": dict(
            x=3, y=2, z=2),
    },
    middle={  # Lower the view point
        **fig_phase_space_camera_dict_general,
        # center= dict(
        #     x=0, y=0, z=0),
        "eye": dict(
            x=2.75,y=2.75,z=.8),
    },
    right={
        **fig_phase_space_camera_dict_general,
        # center= dict(
        #     x=0, y=0, z=0),
        "eye": dict(
            x=2, y=3, z=2),
    }
)


fig_phase_space_lorenz_camera_dict = dict(
    left={
        **fig_phase_space_camera_dict_general,
        # center= dict(
        #     x=0, y=0, z=0),
        "eye": dict(
            # x=2.25, y=1.25, z=1.25),
        x = 1.25, y = -1.25, z = 1.25),
},
    middle={  # Lower the view point
        **fig_phase_space_camera_dict_general,
        # center= dict(
        #     x=0, y=0, z=0),
        "eye": dict(
            x=1.75,y=1.75,z=.1),
    },
    right={
        **fig_phase_space_camera_dict_general,
        # center= dict(
        #     x=0, y=0, z=0),
        "eye": dict(
            # x=1.25, y=2.25, z=1.25),
        # x = 4.25, y = 2.25, z = 1.25),
        # x = -1.25, y = 1.25, z = 1.25),
            x=.1, y=.1, z=2.75),
    }
)


legend_x_coord = 1.05
legend_y_coord = .5
