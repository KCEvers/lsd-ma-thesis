
# Import packages
import vars
from python.master.globalimports import *
from python.processing import hex_to_rgb, rgb_to_rgba, tint_or_shade_rgb, rgb_to_rgba_255, acf_wrapper


def recurrence_plot(dict_ent, layout_der, pattern_derivatives_output,
                mask_unstand_or_stand,
                raw_or_PC_unstand_or_stand,
                method='frr', # Fixed recurrence rate
                thresh=.05, # Fixed recurrence rate of .05
                    theiler_window=0,
                    lmin=2 # Minimum line length
    ):

    # Build file path to output data from RQA
    dict_ent['pipeline'] = 'preproc-rois'
    dict_ent['extension'] = '.json'
    dict_ent['suffix'] = 'PCs'
    dict_ent['timeseries_or_figs'] = 'timeseries'
    dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
    dict_ent['raw_or_PC_unstand_or_stand'] = raw_or_PC_unstand_or_stand
    filepath_data = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True)).as_posix()

    # Build RQA output file path
    filepath_output = pathlib.Path(
        layout_der.build_path({**dict_ent, 'pipeline': 'timeseries_analysis',
                               'suffix': "{}_{}_{}{}_theiler_{}".format(dict_ent['suffix'], 'RQAoutput', method, thresh, theiler_window)},
        pattern_derivatives_output, validate=False,
                              absolute_paths=True)).as_posix()

    # Build plot output file path
    filepath_fig_RP_unthresh = pathlib.Path(
        layout_der.build_path({**dict_ent, 'pipeline': 'timeseries_analysis',
                               'timeseries_or_figs': 'figs', 'type_of_fig': 'RP', 'extension': '.png',
                               'suffix': "{}_{}_{}{}_theiler_{}_unthresh".format(dict_ent['suffix'], 'RQAoutput', method, thresh, theiler_window)},
        pattern_derivatives_output, validate=False,
                              absolute_paths=True)).as_posix()
    filepath_fig_RP_thresh = pathlib.Path(
        layout_der.build_path({**dict_ent, 'pipeline': 'timeseries_analysis',
                               'timeseries_or_figs': 'figs', 'type_of_fig': 'RP', 'extension': '.png',
                               'suffix': "{}_{}_{}{}_theiler_{}_thresh".format(dict_ent['suffix'], 'RQAoutput', method, thresh, theiler_window)},
        pattern_derivatives_output, validate=False,
                              absolute_paths=True)).as_posix()

    # Read data
    if dict_ent["extension"] == ".json":
        with open(filepath_data, 'r') as file:
            data_json = json.load(file)  # Read .json file


    # Read output
    if dict_ent["extension"] == ".json":
        with open(filepath_output, 'r') as file:
            output_json = json.load(file)  # Read .json file

    # Plot unthresholded recurrence matrix
    colourbar = dict(
        title="Distance",
        title_font_family=vars.font_family,
        titleside="top",
        tickmode="array",
        tickfont=dict(size=vars.fig_RP_colorbar_ticks_font_size),
        # tickvals=[0, .25, .5, .75, 1],
        # ticktext=["no event", "stimulus", "response, correct: n/a", "incorrect response", "correct response"],
        ticks="outside",
        len= .8, # len = Length of colour bar; set to half so it doesn't extend to the balance plot
        yanchor= "middle" # Anchors colour bar to ... of the plot
    )

    # Create plotly subplot figure for UNTHRESHOLDED RP
    fig_RP_unthresh = make_subplots(
        rows=2, cols=2,
        # subplot_titles=subplot_titles,
        # shared_xaxes=True,
        # x_title=x_title,
        # y_title=y_title,
        vertical_spacing=.01,  # Vertical spacing indicates the distance between the rows of the subplots
        horizontal_spacing=.01,  # Horizontal spacing indicates the distance between the columns of the subplots
        row_heights=[.8, .2],
        column_widths=[.2, .8]
    )

    fig_RP_unthresh.add_trace(go.Heatmap(
                        z=output_json["RP_unthresh"],
        #[[1, 20, 30],
                          # [20, 1, 60],
                          # [30, 60, 1]]
        colorbar=colourbar,
        colorscale=vars.RP_colours[dict_ent["session"]],
        reversescale=True # Reverse colour scale so that closest neighbours are coloured darkest
    ), row = 1, col = 2)

    # Update general layout
    fig_RP_unthresh.update_layout(
        # title={  # Master title
        #     'text': master_title,
        #     'font': dict(size=vars.fig_stationary_title_font_size),
        #     'pad': vars.margins_fig_stationary_title,  # Add padding between master title and plot
        #     'x': 0.5,  # Centre title
        #     'xanchor': 'center',
        #     'y': .97,  # Move title up a bit
        #     'yanchor': 'middle',
        #     # 'yref': 'paper'
        # },
        height=vars.fig_RP_height,
        width=vars.fig_RP_width,
        # showlegend=False,
        # legend=dict(
        #     # yanchor="bottom",
        #     # y=vars.legend_y_coord,
        #     # xanchor="left",
        #     # x=vars.legend_x_coord,
        #     font=dict(
        #         size=vars.fig_PC_label_font_size,
        #         # color="black"
        #     ),
        # ),
        # legend_tracegroupgap=5,  # Distance between legend groups in pixels
        plot_bgcolor="white",  # Change background colour
        paper_bgcolor='white',
        font_family=vars.font_family,  # Change font family
        margin=vars.margins_fig_RP,  # Adjust margins figure
    )
    fig_RP_unthresh.show()

    # TO DO
    # # Plot timeseries
    # ts_plot = go.Scatter
    # # Flip timeseries plot
    # fig_RP_unthresh.add_trace( , row=1, col = 1)
    # fig_RP_unthresh.add_trace( , row=2, col = 2)

    fig_RP_unthresh.show()

    # Create .png file of plot
    fig_RP_unthresh.write_image(filepath_fig_RP_unthresh)

    # Create plotly subplot figure for THRESHOLDED RP
    fig_RP_thresh = make_subplots(
        rows=2, cols=2,
        # subplot_titles=subplot_titles,
        # shared_xaxes=True,
        # x_title=x_title,
        # y_title=y_title,
        vertical_spacing=.01,  # Vertical spacing indicates the distance between the rows of the subplots
        horizontal_spacing=.01,  # Horizontal spacing indicates the distance between the columns of the subplots
        row_heights=[.8, .2],
        column_widths=[.2, .8]
    )

    fig_RP_thresh.add_trace(go.Heatmap(
                        z=output_json["RP_thresh"],
        #[[1, 20, 30],
                          # [20, 1, 60],
                          # [30, 60, 1]]
        # colorbar=colourbar, # No colour bar
        showscale=False,
        colorscale=vars.RP_colours[dict_ent["session"]],
        # reversescale=True # Reverse colour scale so that closest neighbours are coloured darkest
    ), row = 1, col = 2)

    # Update general layout
    fig_RP_thresh.update_layout(
        # title={  # Master title
        #     'text': master_title,
        #     'font': dict(size=vars.fig_stationary_title_font_size),
        #     'pad': vars.margins_fig_stationary_title,  # Add padding between master title and plot
        #     'x': 0.5,  # Centre title
        #     'xanchor': 'center',
        #     'y': .97,  # Move title up a bit
        #     'yanchor': 'middle',
        #     # 'yref': 'paper'
        # },
        height=vars.fig_RP_height,
        width=vars.fig_RP_width,
        # showlegend=False,
        # legend=dict(
        #     # yanchor="bottom",
        #     # y=vars.legend_y_coord,
        #     # xanchor="left",
        #     # x=vars.legend_x_coord,
        #     font=dict(
        #         size=vars.fig_PC_label_font_size,
        #         # color="black"
        #     ),
        # ),
        # legend_tracegroupgap=5,  # Distance between legend groups in pixels
        plot_bgcolor="white",  # Change background colour
        paper_bgcolor='white',
        font_family=vars.font_family,  # Change font family
        margin=vars.margins_fig_RP,  # Adjust margins figure
    )
    fig_RP_thresh.show()

    # Create .png file of plot
    fig_RP_thresh.write_image(filepath_fig_RP_thresh)

    return filepath_fig_RP_unthresh

