"""
Functions for data visualisation: plot timeseries, plot phase space, and plot summary figures and figures per ROI.
"""

# Import packages
from python.master.globalimports import *
from python.processing import hex_to_rgb, rgb_to_rgba, tint_or_shade_rgb, rgb_to_rgba_255, acf_wrapper

def create_fig_stationary_wrapper(Schaefer_ROIs_df, nr_PCs, TR, dict_ent, layout_der, pattern_derivatives_output,
                    mask_unstand_or_stand = "mask_stand", raw_or_PC_unstand_or_stand = "PC_unstand"):

    # Add entities to dictionary
    dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
    dict_ent['raw_or_PC_unstand_or_stand'] = raw_or_PC_unstand_or_stand

    # Define subplot titles and number of rows and columns
    subplot_titles = []
    for PC_nr in range(1, nr_PCs + 1):
        subplot_titles = subplot_titles + ["PC{}".format(PC_nr)]

    for PC_nr in range(1, nr_PCs + 1):
        subplot_titles = subplot_titles + ["Autocorrelation"]

    for PC_nr in range(1, nr_PCs + 1):
        subplot_titles = subplot_titles + ["Comparison first (darker) and second (lighter) half"]

    Schaefer_ROIs_df.apply(
        lambda x: create_fig_stationary(df_row=x,
                                            TR=TR,
                                            subplot_titles=subplot_titles,
                                            nr_PCs=nr_PCs,
                                            Schaefer_ROIs_df=Schaefer_ROIs_df,
                                            dict_ent=dict_ent,
                                            layout_der=layout_der,
                                            pattern_derivatives_output=pattern_derivatives_output,
                                            ),
        axis=1)

    return None


def create_fig_stationary(df_row, TR, subplot_titles, nr_PCs, Schaefer_ROIs_df, dict_ent, layout_der,
                              pattern_derivatives_output, nbins=20
                              ):
    # Find label number, row number, and column number
    mask_label = df_row["ROI_label"]
    label_nr = np.where(Schaefer_ROIs_df["ROI_label"].values == mask_label)[0][0]
    nr_rows = 3
    nr_cols = nr_PCs

    # Define file path output figure
    # dict_ent['suffix'] = "PCs" if "PC" in raw_or_PC_unstand_or_stand else "mask"
    dict_ent['suffix'] = "PCs_{}".format(mask_label) if "PC" in dict_ent["raw_or_PC_unstand_or_stand"] else "mask_{}".format(mask_label)
    dict_ent['extension'] = ".csv" if dict_ent["raw_or_PC_unstand_or_stand"] == "raw" else ".json"
    dict_ent['timeseries_or_figs'] = "timeseries"
    if "type_of_fig" in dict_ent.keys(): # Make sure the type_of_fig key is not in the dictionary for reading the data
        del dict_ent["type_of_fig"]

    # Build path for output figure
    filepath_fig_stationary = pathlib.Path(
        layout_der.build_path(
            {**dict_ent, 'extension': ".html", 'timeseries_or_figs': 'figs',
             'type_of_fig': 'stationarity'},
            pattern_derivatives_output, validate=False,
            absolute_paths=True))

    # Set up figure
    # row_nr = np.repeat(np.arange(1, nr_rows + 1), nr_cols)[label_nr]
    # col_nr = (list(range(1, nr_cols + 1)) * nr_rows)[label_nr]

    ## Load data ((standardized) timeseries of all ROIs or PCs of all ROIs)
    # Build path by filling in dict_ent (making sure to include the appropriate suffix and extension) in pattern_derivatives_output to save timeseries file in derivatives directory
    filepath_masked_data = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True))

    # Read data
    # if dict_ent['extension'] == ".csv":
    #     masked_data = pd.read_csv(filepath_masked_data, sep='\t')  # Read csv file
    #
    #     # Demean if unstandardized version; before we compute the mean timeseries across voxels, demean each voxel's timeseries using its own mean
    #     if dict_ent["mask_unstand_or_stand"] == "mask_unstand":
    #         masked_data = (masked_data - masked_data.mean(axis=0))
    #
    #         # Same result when substracting mean of mean timeseries and when demeaning before, i.e.
    #         # mean_timeseries - np.array(mean_timeseries).mean()
    #
    #     # Prepare data for plotting (time vector, mean, and error bands which show standard error of the mean)
    #     # Define time vector with respect to image acquisition
    #     time_vec = list(np.arange(0, TR * masked_data.shape[0], TR))  # Time vector in seconds, steps of TR
    #     mean_timeseries = list(masked_data.mean(axis=1))  # Mean time series
    #     std_lower_timeseries = list(mean_timeseries - masked_data.std(axis=1))  # Line for lower error band
    #     std_upper_timeseries = list(mean_timeseries + masked_data.std(axis=1))  # Line for upper error band
    #     sem_lower_timeseries = list(mean_timeseries - (masked_data.std(axis=1, ddof=1) / np.sqrt(
    #         masked_data.shape[1])))  # Line for lower error band: standard error of the mean
    #     sem_upper_timeseries = list(mean_timeseries + (masked_data.std(axis=1, ddof=1) / np.sqrt(
    #         masked_data.shape[1])))  # Line for upper error band: standard error of the mean
    #
    #     # Mean time series
    #     timeseries_scatter = go.Scatter(
    #         x=time_vec,
    #         y=mean_timeseries,
    #         # line=dict(color=vars.ROI_colours[label_nr], width=vars.line_width_timeseries_mean),
    #         line=dict(color=vars.ROI_colours[label_nr], width=vars.line_width_timeseries_mean),
    #         # name="",
    #         mode='lines'
    #     )
    #     # Error bands
    #     timeseries_error_scatter = go.Scatter(
    #         x=time_vec + time_vec[::-1],  # Time vector, then time vector reversed
    #         y=std_upper_timeseries + std_lower_timeseries[::-1],  # Upper, then lower reversed
    #         fill='toself',
    #         fillcolor=rgb_to_rgba(hex_to_rgb(vars.ROI_colours[label_nr]), vars.alpha_val_timeseries_error_bands),
    #         # Convert to rgba
    #         # line=dict(color=vars.ROI_colours[label_nr]), # Convert to rgba
    #         line=dict(
    #             color=rgb_to_rgba(hex_to_rgb(vars.ROI_colours[label_nr]), vars.alpha_val_timeseries_error_bands)),
    #         # Convert to rgba
    #         hoverinfo="skip",
    #         showlegend=False
    #     )
    #
    #     # Plot mean time series
    #     fig_stationary.add_trace(timeseries_scatter, row=row_nr, col=col_nr)
    #
    #     # Plot error bands
    #     fig_stationary.add_trace(timeseries_error_scatter, row=row_nr, col=col_nr)
    #
    #     # Save stand-alone plot
    #     save_subplot_timeseries(label_nr, mask_label, subplot_titles, x_title, y_title,
    #                             filepath_fig_stationary, dict_ent, layout_der, pattern_derivatives_output, nr_PCs,
    #                             data={"timeseries_scatter": timeseries_scatter,
    #                                   "timeseries_error_scatter": timeseries_error_scatter})

    if dict_ent['extension'] == ".json":
        # Read data
        with open(filepath_masked_data, 'r') as file:
            masked_data = json.load(file)  # Read .json file

        # Plot details
        # Define time vector with respect to image acquisition - the same for every PC
        time_vec = list(np.arange(0, TR * len(masked_data["PC1"]), TR))  # Time vector in seconds, steps of TR
        # Concatenate all PC data and flatten list
        all_PC_data = [item for sublist in [masked_data["PC{}".format(PC_nr)] for PC_nr in range(1, nr_PCs+1)] for item in
                       sublist]
        # Find minimum and maximum PC value for histogram edges
        min_PC = np.floor(np.min(all_PC_data)/10)*10 # Round down to nearest 10
        max_PC = np.ceil(np.max(all_PC_data)/10)*10 # Round up to nearest 10
        bin_size = (np.abs(min_PC) + np.abs(max_PC)) / nbins # To make sure every PC's histogram has the same bin width, compute based on PC range
        # Define dictionaries to save plotly objects to
        scatters = {}  # Scatter plotly objects
        scatters_autocorr = {}
        hists = {}
        annos = {}  # Annotations
        annotation_y_PCs = [.65, 0.25, -.2]  # Vector of y coordinates for PCs
        annotation_y_all_PCs = 1.05

        # General annotation with summed explained variance
        if "concat" in filepath_masked_data.as_posix():
            expl_var_PCs_run1 = np.array(masked_data['explained_variance_ratio_']["run-01"])*100
            expl_var_PCs_run3 = np.array(masked_data['explained_variance_ratio_']["run-03"])*100
            expl_var_PCs_total = np.round((np.sum(expl_var_PCs_run1 + expl_var_PCs_run3)/2), 1) # Compute average explained variance across two runs
        else:
            expl_var_PCs = np.array(masked_data['explained_variance_ratio_'])*100
            expl_var_PCs_total = np.round(np.sum(expl_var_PCs), 1)

        # Define title and subtitle
        if "concat" in filepath_masked_data.as_posix():
            subtitle = "Average total explained variance: {}% (run-01: {}%; run-03: {}%)".format(
                expl_var_PCs_total,
                round(np.sum(expl_var_PCs_run1), 1),
                round(np.sum(expl_var_PCs_run3), 1))
            for PC_nr in range(1, nr_PCs+1):
                subplot_titles[PC_nr - 1] = "PC{}<br><sup>(run-01: {}%; run-03: {}%)</sup>".format(PC_nr,
                                                                                   round(expl_var_PCs_run1[PC_nr - 1],
                                                                                         1),
                                                                                   round(expl_var_PCs_run3[PC_nr - 1],
                                                                                         1))
        else:
            subtitle = "Total explained variance: {}%".format(expl_var_PCs_total)
            for PC_nr in range(1, nr_PCs+1):
                subplot_titles[PC_nr - 1] = "{} ({}%)".format(subplot_titles[PC_nr - 1],
                                                              round(expl_var_PCs[PC_nr - 1], 1))
        # subtitle = "Explained variance per PC shown between brackets (%)" if "PC" in raw_or_PC_unstand_or_stand else "Error bands show +/- 1SD"
        master_title = "<b>Stationarity check ROI {}</b><br><sup>{}</sup>".format(
            mask_label.replace("7Networks_", ""),
            subtitle)  # Master title of subplot figure

        # Create plotly subplot figure
        fig_stationary = make_subplots(
            rows=nr_rows, cols=nr_cols,
            subplot_titles=subplot_titles,
            # shared_xaxes=True,
            # x_title=x_title,
            # y_title=y_title,
            vertical_spacing=.175,  # Vertical spacing indicates the distance between the rows of the subplots
            horizontal_spacing=.1,  # Horizontal spacing indicates the distance between the columns of the subplots
            # row_heights=[.2, .2, .2, .2],
        )
        # annos["all_PCs"] = dict(
        #     x=1.025,
        #     y=annotation_y_all_PCs,
        #     xref='x domain',
        #     yref='y domain',
        #     axref='x domain',
        #     ayref='y domain',
        #     ax=1.025,
        #     ay=annotation_y_all_PCs,
        #     xanchor='left',
        #     yanchor='bottom',
        #     text=
        #     # "PC {}<br>({} %)".format(PC_nr,
        #     "Total: {}%".format(expl_var_PCs_total),
        #     showarrow=False,
        #     font=dict(
        #         # family="Courier New, monospace",
        #         size=vars.fig_PC_label_font_size + 1,
        #         color=tint_or_shade_rgb(rgb_str=hex_to_rgb(vars.ROI_colours[label_nr]),
        #                                 factor=.9,
        #                                 shade_or_tint="shade"),
        #     )
        # )

        # For adding vertical lines
        vlines = list()

        # Loop through PCs and create plotly objects
        for PC_nr in range(1, nr_PCs + 1):
            # Plot component line through time
            scatters["PC{}".format(PC_nr)] = go.Scatter(
                x=time_vec,
                y=masked_data["PC{}".format(PC_nr)],
                # line=dict(color=vars.ROI_colours[label_nr], width=vars.line_width_timeseries_mean),
                line=dict(color=tint_or_shade_rgb(rgb_str=hex_to_rgb(vars.ROI_colours[label_nr]),
                                                  factor=vars.fig_PC_shade_tint_factors[PC_nr - 1],
                                                  shade_or_tint=vars.fig_PC_shade_or_tint[PC_nr - 1]),
                          width=vars.fig_PC_lwd[PC_nr - 1]),
                showlegend=False,
                legendgroup="{}".format(mask_label),
                legendgrouptitle={
                    # "text": annos["all_PCs"]["text"],
                                  "font": {"size": vars.fig_PC_label_font_size + 1}
                                  },
                mode='lines'
            )

            ## Add axis titles
            # Row 1
            fig_stationary['layout']['xaxis{}'.format(PC_nr if PC_nr > 1 else "")]['title'] = {'text':'Time (sec)', 'font':{'size':vars.fig_stationary_subplot_axis_title_font_size }}  # Increase font size axis title
            fig_stationary['layout']['xaxis{}'.format(PC_nr if PC_nr > 1 else "")]['nticks'] = vars.fig_stationary_subplot_nr_ticks
            fig_stationary['layout']['xaxis{}'.format(PC_nr if PC_nr > 1 else "")]['tickfont'] = {
            'size': vars.fig_stationary_subplot_tick_font_size}
            fig_stationary['layout']['yaxis{}'.format(PC_nr if PC_nr > 1 else "")]['title'] = {'text':'BOLD', 'font':{'size':vars.fig_stationary_subplot_axis_title_font_size }}  # Increase font size axis title
            fig_stationary['layout']['yaxis{}'.format(PC_nr if PC_nr > 1 else "")]['nticks'] = vars.fig_stationary_subplot_nr_ticks
            fig_stationary['layout']['yaxis{}'.format(PC_nr if PC_nr > 1 else "")]['tickfont'] = {
            'size': vars.fig_stationary_subplot_tick_font_size}

            # Row 2
            fig_stationary['layout']['xaxis{}'.format(PC_nr+nr_rows)]['title'] = {'text': 'Lag',
                                                                                               'font': {
                                                                                                   'size': vars.fig_stationary_subplot_axis_title_font_size}}  # Increase font size axis title
            fig_stationary['layout']['xaxis{}'.format(PC_nr+nr_rows)][
                'nticks'] = vars.fig_stationary_subplot_nr_ticks
            fig_stationary['layout']['xaxis{}'.format(PC_nr+nr_rows)][
                'tickfont'] = {
            'size': vars.fig_stationary_subplot_tick_font_size}
            fig_stationary['layout']['yaxis{}'.format(PC_nr+nr_rows)]['title'] = {'text': 'Autocorrelation', 'font': {
                'size': vars.fig_stationary_subplot_axis_title_font_size}}  # Increase font size axis title
            fig_stationary['layout']['yaxis{}'.format(PC_nr+nr_rows)][
                'nticks'] = vars.fig_stationary_subplot_nr_ticks
            fig_stationary['layout']['yaxis{}'.format(PC_nr+nr_rows)][
                'tickfont'] = {
            'size': vars.fig_stationary_subplot_tick_font_size}

            # Row 3
            fig_stationary['layout']['xaxis{}'.format(PC_nr+nr_rows*2)]['title'] = {'text': 'Values',
                                                                                               'font': {
                                                                                                   'size': vars.fig_stationary_subplot_axis_title_font_size}}  # Increase font size axis title
            fig_stationary['layout']['xaxis{}'.format(PC_nr+nr_rows*2)][
                'nticks'] = vars.fig_stationary_subplot_nr_ticks
            fig_stationary['layout']['xaxis{}'.format(PC_nr+nr_rows*2)][
                'tickfont'] = {
            'size': vars.fig_stationary_subplot_tick_font_size}
            fig_stationary['layout']['yaxis{}'.format(PC_nr+nr_rows*2)]['title'] = {'text': 'Density', 'font': {
                'size': vars.fig_stationary_subplot_axis_title_font_size}}  # Increase font size axis title
            fig_stationary['layout']['yaxis{}'.format(PC_nr+nr_rows*2)][
                'nticks'] = vars.fig_stationary_subplot_nr_ticks
            fig_stationary['layout']['yaxis{}'.format(PC_nr+nr_rows*2)][
                'tickfont'] = {
            'size': vars.fig_stationary_subplot_tick_font_size}

            ## Determine and plot autcorrelation
            # statsmodels.graphics.tsaplots.plot_acf(np.array(masked_data["PC{}".format(PC_nr)])); plt.show()
            acf_x, confint, lags = acf_wrapper(masked_data["PC{}".format(PC_nr)],lags=len(masked_data["PC{}".format(PC_nr)])-1, confint=None, alpha=.05)

            # Plot confidence bands
            # Add lower band
            fig_stationary.add_trace(go.Scatter(x=lags, y=confint[:,0]-acf_x, # Subtract autocorrelation as done in statsmodels.plot_acf
                                     fill=None,
                                     mode='lines',
                                                showlegend=False,
                                     line=dict(width=0.5, color=rgb_to_rgba_255(
                                         tint_or_shade_rgb(
                                             rgb_str=hex_to_rgb(vars.ROI_colours[label_nr]),
                                             factor=vars.fig_PC_shade_tint_factors[PC_nr - 1],
                                             shade_or_tint=vars.fig_PC_shade_or_tint[PC_nr - 1]),
                                         vars.fig_stationarity_alpha_ci + .3)),
                                                ),
                                     row=2, col=PC_nr)
            # Add upper band and fill area between lower and upper confidence band
            fig_stationary.add_trace(go.Scatter(
                x=lags, y=confint[:,1]-acf_x, # Subtract autocorrelation as done in statsmodels.plot_acf
                fill='tonexty',  # fill area between confidence bands
                fillcolor= rgb_to_rgba_255(
                    tint_or_shade_rgb(
                        rgb_str=hex_to_rgb(vars.ROI_colours[label_nr]),
                        factor=vars.fig_PC_shade_tint_factors[PC_nr - 1],
                        shade_or_tint=vars.fig_PC_shade_or_tint[PC_nr - 1]),
                    vars.fig_stationarity_alpha_ci),
                line=dict(width=0.5, color=rgb_to_rgba_255(
                    tint_or_shade_rgb(
                        rgb_str=hex_to_rgb(vars.ROI_colours[label_nr]),
                        factor=vars.fig_PC_shade_tint_factors[0],
                        shade_or_tint=vars.fig_PC_shade_or_tint[0]),
                    vars.fig_stationarity_alpha_ci+.3)),
                mode='lines', #line_color='indigo'
                showlegend=False,
            ),
                                     row=2, col=PC_nr)


            # Add vertical lines
            for i in lags:
                vlines.append({'type': 'line',
                               'xref': 'x{}'.format(PC_nr+nr_rows), # x4, x5, x6
                               'yref': 'y{}'.format(PC_nr+nr_rows), #y4, y5, y6
                               'x0': i,
                               'y0': 0,
                               'x1': i,
                               'y1':  acf_x[i],
                              'line_color': tint_or_shade_rgb(rgb_str=hex_to_rgb(vars.ROI_colours[label_nr]),
                                                  factor=vars.fig_PC_shade_tint_factors[PC_nr - 1],
                                                  shade_or_tint=vars.fig_PC_shade_or_tint[PC_nr - 1])
                               })

            # vlines_all = go.Layout(shapes=vlines)

            # Add dots and vertical lines for each lag
            scatters_autocorr["PC{}".format(PC_nr)] = go.Scatter(
                x=lags,
                y=acf_x,
                showlegend=False,
                # line=dict(color=vars.ROI_colours[label_nr], width=vars.line_width_timeseries_mean),
                line=dict(color=tint_or_shade_rgb(rgb_str=hex_to_rgb(vars.ROI_colours[label_nr]),
                                                  factor=vars.fig_PC_shade_tint_factors[PC_nr - 1],
                                                  shade_or_tint=vars.fig_PC_shade_or_tint[PC_nr - 1]),
                          width=vars.fig_PC_lwd[PC_nr - 1]),
                # name="PC {} ({}%)".format(PC_nr,
                #                           np.round(((masked_data['explained_variance_ratio_']['run-01'][PC_nr - 1] + masked_data['explained_variance_ratio_']['run-03'][PC_nr - 1])/2) * 100, 1)),
                # legendgroup="{}".format(mask_label),
                # legendgrouptitle={"text": annos["all_PCs"]["text"],
                #                   "font": {"size": vars.fig_PC_label_font_size + 1}
                #                   },
                mode='markers',
            )

            # Plot distribution in third row
            # hist, bin_edges = np.histogram(np.array(masked_data["PC{}".format(PC_nr)]),bins=20,density=True)
            hists["PC{}".format(PC_nr)] = {}
            hists["PC{}".format(PC_nr)]["first-half"] = go.Histogram(
                x=np.array(masked_data["PC{}".format(PC_nr)][0:int(len(masked_data["PC{}".format(PC_nr)])/2)]),
                # nbinsx=25,
                xbins=dict(
                    start=min_PC, # min(PCs)
                    end=max_PC,
                    size=bin_size
                ),
                # autobinx=False,
                histnorm='probability',
                marker_color=tint_or_shade_rgb(rgb_str=hex_to_rgb(vars.ROI_colours[label_nr]),
                                                  factor=vars.fig_PC_shade_tint_factors[PC_nr - 1],
                                                  shade_or_tint=vars.fig_PC_shade_or_tint[PC_nr - 1]),
                opacity=0.85, name="First half", showlegend=False)
            hists["PC{}".format(PC_nr)]["second-half"] = go.Histogram(
                x=np.array(masked_data["PC{}".format(PC_nr)][int(len(masked_data["PC{}".format(PC_nr)])/2):len(masked_data["PC{}".format(PC_nr)])]),
                # nbinsx=25,
                xbins=dict(
                    start=min_PC,
                    end=max_PC,
                    size=bin_size
                ),
                # autobinx=False,
                histnorm='probability',
                marker_color=tint_or_shade_rgb(rgb_str=hex_to_rgb(vars.ROI_colours[label_nr]),
                                                  factor=vars.fig_PC_shade_tint_factors[PC_nr - 1],
                                                  shade_or_tint=vars.fig_PC_shade_or_tint[PC_nr - 1]),
    opacity=0.45,
                name="Second half", showlegend=False)

            # fig_stationary.add_shape(vlines, row=1, col=PC_nr)
            # plot_acf(np.array(masked_data["PC{}".format(PC_nr)]))
            # plt.acorr(np.array(masked_data["PC{}".format(PC_nr)]), maxlags=10)

            # # Annotations
            # annos["PC{}".format(PC_nr)] = dict(
            #     x=1.025,
            #     y=annotation_y_PCs[PC_nr - 1],
            #     # , xref = 'paper'
            #     # , yref = 'paper'
            #     xref='x domain',
            #     yref='y domain',
            #     axref='x domain',
            #     ayref='y domain',
            #     ax=1.025,
            #     ay=annotation_y_PCs[PC_nr - 1],
            #     xanchor='left',
            #     yanchor='bottom',
            #     text=
            #     # "PC {}<br>({} %)".format(PC_nr,
            #     "PC {} ({}%)".format(PC_nr, np.round(masked_data['explained_variance_ratio_'][PC_nr - 1] * 100, 1)),
            #     showarrow=False,
            #     font=dict(
            #         # family="Courier New, monospace",
            #         size=vars.fig_PC_label_font_size,
            #         color=tint_or_shade_rgb(rgb_str=hex_to_rgb(vars.ROI_colours[label_nr]),
            #                                 factor=vars.fig_PC_shade_tint_factors[PC_nr - 1],
            #                                 shade_or_tint=vars.fig_PC_shade_or_tint[PC_nr - 1]),
            #         # "white",#
            #     )
            # )

            # Plot PC
            # fig_stationary.add_trace(scatters["PC{}".format(PC_nr)], row=row_nr, col=col_nr)
        # Add vertical lines
        fig_stationary.update_layout(shapes=vlines)

        # Loop through PCs and add line and annotation
        for PC_nr in range(1, nr_PCs + 1):
            # Plot PC
            # scatters["PC{}".format(PC_nr)]["line"]["width"] += 2  # Increase line width
            fig_stationary.add_trace(scatters["PC{}".format(PC_nr)], row = 1, col=PC_nr)
            fig_stationary.add_trace(scatters_autocorr["PC{}".format(PC_nr)], row = 2, col=PC_nr)
            # fig_stationary.add_trace(hists["PC{}".format(PC_nr)], row = 3, col=PC_nr)
            fig_stationary.add_trace(hists["PC{}".format(PC_nr)]["first-half"], row = 3, col=PC_nr)
            fig_stationary.add_trace(hists["PC{}".format(PC_nr)]["second-half"], row = 3, col=PC_nr)

            # For standardized timeseries, show fixed y-range for all
            if ((dict_ent["mask_unstand_or_stand"] == "mask_unstand") & (
                    dict_ent["raw_or_PC_unstand_or_stand"] == "raw")):
                fig_stationary.update_yaxes(row = 1, col=PC_nr,
                                            range=[vars.fig_unst_time_y_range_min, vars.fig_unst_time_y_range_max])
            elif ((dict_ent["mask_unstand_or_stand"] == "mask_stand") & (
                    dict_ent["raw_or_PC_unstand_or_stand"] == "raw")):
                fig_stationary.update_yaxes(row = 1, col=PC_nr,
                                            range=[vars.fig_st_time_y_range_min, vars.fig_st_time_y_range_max])
            elif "unstand" in dict_ent["raw_or_PC_unstand_or_stand"]:
                fig_stationary.update_yaxes(row = 1, col=PC_nr,
                                            range=[vars.fig_unst_PC_y_range_min, vars.fig_unst_PC_y_range_max])
            elif "_stand" in dict_ent["raw_or_PC_unstand_or_stand"]:
                fig_stationary.update_yaxes(row = 1, col=PC_nr,
                                            range=[vars.fig_st_PC_y_range_min, vars.fig_st_PC_y_range_max])

            # Make histograms same x-axis
            fig_stationary.update_xaxes(row=3, col=PC_nr,
                                        range=[min_PC, max_PC])
            # # Add explained variance annotation for PC
            # fig_stationary.add_annotation(annos["PC{}".format(PC_nr)], row=row_nr, col=col_nr)

        # # Add summed explained variance annotation of all PCs
        # fig_stationary.add_annotation(annos["all_PCs"], row=row_nr, col=col_nr)


    ## Overall figure styling
    ## Adjust font sizes of axis labels and titles
    # Add axis lines
    fig_stationary.update_xaxes(showgrid=True, gridwidth=vars.grid_width,
                                gridcolor=vars.grid_color,
                                zeroline=True, zerolinewidth=vars.grid_width,
                                zerolinecolor=vars.grid_color)
    fig_stationary.update_yaxes(showgrid=True, gridwidth=vars.grid_width,
                                gridcolor=vars.grid_color,
                                zeroline=True, zerolinewidth=vars.grid_width,
                                zerolinecolor=vars.grid_color)

    for annotation in fig_stationary['layout']['annotations']:
        if annotation['text'] in subplot_titles:
            annotation['y'] += .001  # Move title up a bit (y-coordinate)
            annotation['font'] = dict(size=vars.fig_stationary_subplot_title_font_size)  # Increase font sizes title
        # annotation['x'] = .5  # Make sure title is centred

    # Update general layout
    fig_stationary.update_layout(
        title={  # Master title
            'text': master_title,
            'font': dict(size=vars.fig_stationary_title_font_size),
            'pad': vars.margins_fig_stationary_title,  # Add padding between master title and plot
            'x': 0.5,  # Centre title
            'xanchor': 'center',
            'y': .97,  # Move title up a bit
            'yanchor': 'middle',
            # 'yref': 'paper'
        },
        height=vars.fig_stationary_height,
        width=vars.fig_stationary_width,
        showlegend=False,
        legend=dict(
            # yanchor="bottom",
            # y=vars.legend_y_coord,
            # xanchor="left",
            # x=vars.legend_x_coord,
            font=dict(
                size=vars.fig_PC_label_font_size,
                # color="black"
            ),
        ),
        # legend_tracegroupgap=5,  # Distance between legend groups in pixels
        plot_bgcolor="white",  # Change background colour
        paper_bgcolor='white',
        font_family=vars.font_family,  # Change font family
        margin=vars.margins_fig_stationary,  # Adjust margins figure
    )

    # Don't show legend for mask timeseries
    if dict_ent["raw_or_PC_unstand_or_stand"] == "raw":
        fig_stationary.update_layout(showlegend=False)

    fig_stationary.show()

    ## Save as html and png

    # Create .png file of plot
    fig_stationary.write_image(filepath_fig_stationary.with_suffix(".png"))

    ## Create stand-alone html of the plot with header
    header = "Subject: {subject}, session: {session}, task: {task}, run: {run}".format(
        subject=dict_ent["subject"],
        session=dict_ent["session"],
        run=dict_ent["run"],
        task=dict_ent["task"])

    # fig_stationary_string = """##{header}  \n""".format(header=header)  # Initialize html python with header
    fig_stationary_string = """##{header}  \n  <br>  """.format(header=header)  # Initialize html python with header
    fig_stationary_string += """{fig_stationary_html_code}  \n"""  # Create placeholder for figure

    fig_stationary_html_template = markdown.markdown(fig_stationary_string)  # Convert markdown to html
    fig_stationary_html_complete = fig_stationary_html_template.replace("{fig_stationary_html_code}",
                                                                        fig_stationary.to_html(full_html=False,
                                                                                               include_plotlyjs='cdn'))  # Finialize html by replacing placeholder with html figure python

    with open(filepath_fig_stationary, 'w') as f:
        f.write(fig_stationary_html_complete)  # Create complete html file

        # f.write(fig_logdata.to_html(full_html=True, include_plotlyjs='cdn'))





    return

def save_anat_img(dict_ent, layout_rawdata, layout_der, type_of_fig="brain"):
    """ Plot and save anatomical image. """
    # Create dictionary of anatomical file
    dict_ent_anat = {'subject': dict_ent["subject"],
                     'datatype': 'anat',
                     'extension': '.nii.gz',
                     'suffix': 'T1w'
                     }

    # Build file path to nifti file (rawdata directory)
    filepath_nifti_anat = pathlib.Path(layout_rawdata.build_path(dict_ent_anat, validate=False, absolute_paths=True))

    # Build file path (derivatives directory)
    filepath_nifti_anat_img = pathlib.Path(
        layout_der.build_path({**dict_ent_anat,
                               'timeseries_or_figs': 'figs',
                               'mask_unstand_or_stand': 'mask_unstand',
                               'raw_or_PC_unstand_or_stand': 'raw',
                               'type_of_fig': 'brain'
                               }, validate=False, absolute_paths=True)).with_suffix("").with_suffix(
        ".png")  # To get rid of the double suffix (i.e. extension, ".nii.gz"), apply with_suffix() twice

    plot_title = "Anatomical image of subject {}".format(dict_ent_anat["subject"])
    fig = plt.figure(
        1)  # Set up one figure which will be overwritten each time, so that not a new image is created for each run of the loop

    # Plot and save
    nilearn.plotting.plot_anat(
        anat_img=filepath_nifti_anat.as_posix(),
        cut_coords=None,
        output_file=filepath_nifti_anat_img.as_posix(),
        display_mode='ortho',
        figure=fig,
        axes=None,
        title=plot_title,
        annotate=True, threshold=None,
        draw_cross=False,
        black_bg='auto', dim='auto')  # cmap=<matplotlib.colors.LinearSegmentedColormap object>)
    plt.close('all')

    return filepath_nifti_anat_img


def save_func_gif(dict_ent, layout_rawdata, layout_der):
    """Plot all individual functional images belonging to dict_ent and create a gif out of them, and subsequently save these to the corresponding directory in the derivatives folder."""
    dict_ent_func = {
        # "subject": dict_ent["subject"],
        #              "session": dict_ent["session"],
        **dict_ent,
        'datatype': 'func',
        'extension': '.nii.gz',
        'suffix': 'bold'
    }

    # Define input and output file paths
    filepath_nifti_func = pathlib.Path(layout_rawdata.build_path(dict_ent_func, validate=False,
                                                                 absolute_paths=True))  # Define file path (rawdata directory)

    # Build file path to static functional images (derivatives directory)
    filepath_nifti_func_img = pathlib.Path(
        layout_der.build_path({**dict_ent_func,
                               'timeseries_or_figs': 'figs',
                               'mask_unstand_or_stand': 'mask_unstand',
                               'raw_or_PC_unstand_or_stand': 'raw',
                               'type_of_fig': 'brain'
                               },
                              validate=False, absolute_paths=True)).with_suffix("").with_suffix(
        ".png")  # To get rid of the double suffix (i.e. extension, ".nii.gz"), apply with_suffix() twice

    # Build file path to gif (derivatives directory)
    filepath_nifti_func_gif = filepath_nifti_func_img.with_suffix(".gif")

    # First create a png image out of every functional image
    img_func = nib.load(filepath_nifti_func.as_posix())  # Load all functional images
    list_img_func = list(nilearn.image.iter_img(img_func))  # Create list of nifti nibabel images

    i = 0  # Initialize counter
    fig = plt.figure(
        1)  # Set up one figure which will be overwritten each time, so that not a new image is created for each run of the loop

    # Iterate through images and plot and save each one
    for img in nilearn.image.iter_img(img_func):
        plot_func_img(i, filepath_nifti_func_img, list_img_func, fig)
        plt.close('all')
        i += 1

        if i % 50 == 0:
            print("Up to image {} saved!".format(i))

    # i = 0
    # for img in nilearn.image.iter_img(img_func):
    #     Schaefer_200_2mm_plot = nilearn.plotting.plot_roi(img,
    #                                                       title="", cmap=vars.cmap_Schaefer_200_2mm,
    #                                                       colorbar=False,
    #                                                       figure = 1,
    #                                                       output_file=filepath_nifti_func_img.with_stem(
    #                                                           filepath_nifti_func_img.stem + "_{}".format(i)).as_posix()
    #                                                       )
    #     time.sleep(3) # Allow time for the plot to close
    #     i += 1
    #     plt.close(1)
    #     plt.close('all')

    # list_img_func = list(nilearn.image.iter_img(img_func))
    # plot_func_img(i, list_img_func)
    #
    # list(
    #     map(functools.partial(plot_func_img, list_img_func=list_img_func),
    #         np.arange(len(list_img_func))))

    # Now stack all functional images to create gif
    fp_in = filepath_nifti_func_img.with_stem(filepath_nifti_func_img.stem + "_*").as_posix()

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    print("Creating gif!")
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=filepath_nifti_func_gif, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)

    return filepath_nifti_func_gif


def plot_func_img(i, filepath_nifti_func_img, list_img_func, fig):
    output_filepath = filepath_nifti_func_img.with_stem(
        filepath_nifti_func_img.stem + "_{}".format(i)).as_posix()

    Schaefer_200_2mm_plot = nilearn.plotting.plot_roi(list_img_func[i],
                                                      title="",
                                                      cmap=vars.cmap_Schaefer_200_2mm,
                                                      figure=fig,
                                                      colorbar=False,
                                                      output_file=output_filepath
                                                      )
    # plt.close(Schaefer_200_2mm_plot)
    # plt.show(block=False)
    # fig = plt.gcf()
    # fig.write_image(output_filepath)
    # Schaefer_200_2mm_plot.savefig(output_filepath,
    #     dpi="figure")
    # plt.close('all')
    # time.sleep(5)

    return None

def plot_phase_space_3D_wrapper(Schaefer_ROIs_df, dict_ent, layout_der, pattern_derivatives_output, mask_unstand_or_stand="mask_stand", raw_or_PC_unstand_or_stand="PC_unstand"):
    if raw_or_PC_unstand_or_stand == "raw":
        return "Plotting the phase space of raw timeseries is currently not possible!"

    dict_ent["timeseries_or_figs"] = "figs"
    dict_ent["mask_unstand_or_stand"] = mask_unstand_or_stand
    dict_ent["raw_or_PC_unstand_or_stand"] = raw_or_PC_unstand_or_stand

    # Define subplot titles and number
    subplot_titles = ["<b>ROI {}: {} ({} voxels)</b>".format(label_nr + 1,
                                                             Schaefer_ROIs_df["ROI_label"][label_nr].replace(
                                                                 "7Networks_", "").replace("Default_", "").replace(
                                                                 "_", " ").replace("RH", "Right").replace("LH",
                                                                                                          "Left"),
                                                             Schaefer_ROIs_df["nr_voxels"][label_nr]) for label_nr
                      in range(len(Schaefer_ROIs_df["ROI_label"].values))]

    Schaefer_ROIs_df.apply(
        lambda x: plot_phase_space_3D(df_row=x, Schaefer_ROIs_df=Schaefer_ROIs_df, subplot_titles=subplot_titles,dict_ent=dict_ent,
                                                    layout_der=layout_der, pattern_derivatives_output=vars.pattern_derivatives_output
                                                    ), axis=1)

    return None



def plot_phase_space_3D(df_row, Schaefer_ROIs_df, subplot_titles, dict_ent, layout_der, pattern_derivatives_output,
        angles=list(["left", "middle", "right"]),
        nr_rows=1,
        nr_cols=3
                        ):
    # Read PC data
    # Find label number, row number, and column number
    mask_label = df_row["ROI_label"]
    label_nr = np.where(Schaefer_ROIs_df["ROI_label"].values == mask_label)[0][0]
    master_title = subplot_titles[label_nr]

    # Prepare entities dictionary to read data
    dict_ent['suffix'] = "PCs_{}".format(mask_label)
    dict_ent['extension'] = ".json"
    dict_ent['timeseries_or_figs'] = "timeseries"
    # No need to specify mask_unstand_or_stand or raw_or_PC_unstand_or_stand as they were already specified in the wrapper function which calls this function

    # Build path for output figure
    filepath_fig_phase_space_3D = pathlib.Path(
        layout_der.build_path(
            {**dict_ent, 'extension': '.html',
             'suffix': dict_ent['suffix'] + '_phase-space_3D',
             'timeseries_or_figs': 'figs',
             'type_of_fig': 'phase-space' # Only specify here so that the timeseries file path does not contain it
             },
            pattern_derivatives_output,
            validate=False,
            absolute_paths=True))

    # filepath_fig_phase_space_3D = filepath_fig_phase_space_3D_template.with_name(
    #     "{}_mask_{}{}".format(filepath_fig_phase_space_3D_template.stem, mask_label,
    #                           filepath_fig_phase_space_3D_template.suffix))

    ## Load data ((standardized) timeseries of all ROIs or PCs of all ROIs)
    # Build path by filling in dict_ent (making sure to include the appropriate suffix and extension) in pattern_derivatives_output to save timeseries file in derivatives directory
    filepath_masked_data = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True))

    # Read data
    with open(filepath_masked_data, 'r') as file:
        masked_data = json.load(file)  # Read .json file

    # Need to specify that each plot is of type scene such that 3D plots can be made
    specs = [[{'type': 'scene'},
              {'type': 'scene'},
              {'type': 'scene'}]]

    # Create plotly subplot figure
    fig_phase_space = make_subplots(
        rows=nr_rows, cols=nr_cols,
        # subplot_titles=subplot_titles,
        # shared_xaxes=True,
        # shared_yaxes=True,
        # x_title=x_title,
        # y_title=y_title,
        specs=specs,
        # vertical_spacing=.1,  # Vertical spacing indicates the distance between the rows of the subplots
        horizontal_spacing=.05,  # Horizontal spacing indicates the distance between the columns of the subplots
        # row_heights=[.2, .2, .2],
        # column_width=[],
        # column_titles=[]
    )

    # Update general layout
    fig_phase_space.update_layout(
        width=vars.fig_phase_space_width,
        height=vars.fig_phase_space_height,
        # plot_bgcolor="white",  # Change background colour
        # paper_bgcolor='white',
        font_family=vars.font_family,  # Change font family
        margin=vars.margins_fig_phase_space,  # Adjust margins figure
        # autosize=False,
        title={  # Master title
            'text': master_title,
            'font': dict(size=vars.fig_phase_space_title_font_size),
            'pad': vars.margins_fig_phase_space_title,  # Add padding between master title and plot
            'x': 0.5,  # Centre title
            'xanchor': 'center',
            'y': .97,  # Move title up a bit
            'yanchor': 'middle',
            # 'yref': 'paper'
        },
        showlegend=False
    )

    row_nrs = np.repeat(np.arange(1, nr_rows + 1), nr_cols)
    col_nrs = (list(range(1, nr_cols + 1)) * nr_rows)

    # Loop through angles, create plotly objects and add them to corresponding subplot
    scatters = {}
    for angle_idx, angle in enumerate(angles):
        # print("angle_idx {}, angle {}".format(angle_idx, angle))
        # print("row_nrs[angle_idx] {}; col_nrs[angle_idx] {}".format(row_nrs[angle_idx], col_nrs[angle_idx]))

        # Create plotly object and add to dictionary
        scatters[angle] = go.Scatter3d(
            x=masked_data["PC1"], y=masked_data["PC2"], z=masked_data["PC3"],
            marker=dict(
                size=vars.fig_phase_space_marker_size,
                # color=z,
                colorscale='Viridis',
            ),
            line=dict(
                color=tint_or_shade_rgb(rgb_str=hex_to_rgb(vars.ROI_colours[label_nr]),
                                        factor=vars.fig_PC_shade_tint_factors[0],
                                        shade_or_tint=vars.fig_PC_shade_or_tint[0]),
                width=vars.fig_phase_space_line_width
            ),
            scene="scene{}".format(angle_idx + 1)
        )

        # Add plotly object to subplot
        fig_phase_space.add_trace(scatters[angle], row=row_nrs[angle_idx], col=col_nrs[angle_idx])

        # Copy general dictionary of scene attributes and modify
        fig_phase_space_dict = {**vars.fig_phase_space_dict_general}
        fig_phase_space_dict["camera"] = vars.fig_phase_space_camera_dict[angle]
        fig_phase_space_dict["xaxis"]["backgroundcolor"] = rgb_to_rgba_255(
            tint_or_shade_rgb(
                rgb_str=hex_to_rgb(vars.ROI_colours[label_nr]),
                factor=vars.fig_PC_shade_tint_factors[0],
                shade_or_tint=vars.fig_PC_shade_or_tint[0]),
            vars.fig_phase_space_alpha_background)
        fig_phase_space_dict["yaxis"]["backgroundcolor"] = rgb_to_rgba_255(
            tint_or_shade_rgb(
                rgb_str=hex_to_rgb(vars.ROI_colours[label_nr]),
                factor=vars.fig_PC_shade_tint_factors[1],
                shade_or_tint=vars.fig_PC_shade_or_tint[1]),
            vars.fig_phase_space_alpha_background)
        fig_phase_space_dict["zaxis"]["backgroundcolor"] = rgb_to_rgba_255(
            tint_or_shade_rgb(
                rgb_str=hex_to_rgb(vars.ROI_colours[label_nr]),
                factor=vars.fig_PC_shade_tint_factors[2],
                shade_or_tint=vars.fig_PC_shade_or_tint[2]),
            vars.fig_phase_space_alpha_background)

        # Adjust range of axes
        if dict_ent["raw_or_PC_unstand_or_stand"] == "PC_unstand":
            fig_phase_space_dict["xaxis"]["range"] = [vars.fig_unst_PC_y_range_min, vars.fig_unst_PC_y_range_max]
            fig_phase_space_dict["yaxis"]["range"] = [vars.fig_unst_PC_y_range_min, vars.fig_unst_PC_y_range_max]
            fig_phase_space_dict["zaxis"]["range"] = [vars.fig_unst_PC_y_range_min, vars.fig_unst_PC_y_range_max]
        elif dict_ent["raw_or_PC_unstand_or_stand"] == "PC_stand":
            fig_phase_space_dict["xaxis"]["range"] = [vars.fig_st_PC_y_range_min, vars.fig_st_PC_y_range_max]
            fig_phase_space_dict["yaxis"]["range"] = [vars.fig_st_PC_y_range_min, vars.fig_st_PC_y_range_max]
            fig_phase_space_dict["zaxis"]["range"] = [vars.fig_st_PC_y_range_min, vars.fig_st_PC_y_range_max]

        # Update layout
        fig_phase_space.update_scenes({**fig_phase_space_dict},
                                      col=int(col_nrs[angle_idx]), row=int(row_nrs[angle_idx]))

        # fig_phase_space.up
        # fig_phase_space["layout"]["scene"]["camera"]
        # fig_phase_space["layout"]["scene1"]["camera"]
        # fig_phase_space["layout"]["scene2"]["camera"]
    # fix the ratio in the top left subplot to be a cube
    # fig.update_layout(scene_aspectmode='cube')
    # manually force the z-axis to appear twice as big as the other two
    # fig.update_layout(scene2_aspectmode='manual',
    #                   scene2_aspectratio=dict(x=1, y=1, z=2))

        ## Save as html and png
        # Create .png file of plot
        fig_phase_space.write_image(filepath_fig_phase_space_3D.with_suffix(".png"))

        ## Create stand-alone html of the plot with header
        header = "Subject: {subject}, session: {session}, task: {task}, run: {run}".format(
            subject=dict_ent["subject"],
            session=dict_ent["session"],
            run=dict_ent["run"],
            task=dict_ent["task"])

        # fig_timeseries_string = """##{header}  \n""".format(header=header)  # Initialize html python with header
        fig_phase_space_string = """##{header}  \n  <br>  """.format(header=header)  # Initialize html python with header
        fig_phase_space_string += """{fig_phase_space_html_code}  \n"""  # Create placeholder for figure

        fig_phase_space_html_template = markdown.markdown(fig_phase_space_string)  # Convert markdown to html
        fig_phase_space_html_complete = fig_phase_space_html_template.replace("{fig_phase_space_html_code}",
                                                                              fig_phase_space.to_html(full_html=False,
                                                                                                      include_plotlyjs='cdn'))  # Finialize html by replacing placeholder with html figure python

        with open(filepath_fig_phase_space_3D, 'w') as f:
            f.write(fig_phase_space_html_complete)  # Create complete html file

            # f.write(fig_logdata.to_html(full_html=True, include_plotlyjs='cdn'))

    return fig_phase_space


# def plot_phase_space(subplot_titles, filepath_fig_phase_space_template,
#                      ):
#     pattern_derivatives_output = vars.pattern_derivatives_output
#     nr_PCs = vars.nr_PCs
#     df_row = Schaefer_ROIs_df.iloc[0, :]
#
#     # Read PC data
#     # Find label number, row number, and column number
#     mask_label = df_row["ROI_label"]
#     label_nr = np.where(Schaefer_ROIs_df["ROI_label"].values == mask_label)[0][0]
#
#     # Preparation
#     dict_ent['suffix'] = "PCs_{}".format(mask_label)
#     dict_ent['extension'] = ".json"
#     dict_ent['timeseries_or_figs'] = "timeseries"
#     dict_ent['type_of_timeseries'] = "PCs"
#
#     # Build path for output figure
#     filepath_fig_phase_space_3D_template = pathlib.Path(
#         layout_der.build_path(
#             {**dict_ent, 'extension': "html", 'suffix': 'PCs_phase-space_3D', 'timeseries_or_figs': 'figs',
#              'type_of_timeseries': 'PCs', 'type_of_fig': 'phase-space'},
#             pattern_derivatives_output, validate=False,
#             absolute_paths=True))
#
#     filepath_fig_phase_space_3D = filepath_fig_phase_space_3D_template.with_name(
#         "{}_mask_{}{}".format(filepath_fig_phase_space_3D_template.stem, mask_label,
#                               filepath_fig_phase_space_3D_template.suffix))
#
#     ## Load data ((standardized) timeseries of all ROIs or PCs of all ROIs)
#     # Build path by filling in dict_ent (making sure to include the appropriate suffix and extension) in pattern_derivatives_output to save timeseries file in derivatives directory
#     filepath_masked_data = pathlib.Path(
#         layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
#                               absolute_paths=True))
#
#     # Read data
#     with open(filepath_masked_data, 'r') as file:
#         masked_data = json.load(file)  # Read .json file
#
#     # Plot
#     master_title = subplot_titles[label_nr]
#     if nr_PCs == 2:
#         print("2D")
#
#     if nr_PCs == 3:
#         # Plot 3D phase space
#         fig_phase_space = go.Figure(data=go.Scatter3d(
#             x=masked_data["PC1"], y=masked_data["PC2"], z=masked_data["PC3"],
#             marker=dict(
#                 size=vars.fig_phase_space_marker_size,
#                 # color=z,
#                 colorscale='Viridis',
#             ),
#             line=dict(
#                 color='darkblue',
#                 width=vars.fig_phase_space_line_width
#             )
#         ))
#
#         fig_phase_space.update_layout(
#             width=vars.fig_phase_space_width,
#             height=vars.fig_phase_space_height,
#             plot_bgcolor="white",  # Change background colour
#             paper_bgcolor='white',
#             font_family=vars.font_family,  # Change font family
#             margin=vars.margins_fig_phase_space,  # Adjust margins figure
#             # autosize=False,
#             title={  # Master title
#                 'text': master_title,
#                 'font': dict(size=vars.fig_phase_space_title_font_size),
#                 'pad': vars.margins_fig_phase_space_title,  # Add padding between master title and plot
#                 'x': 0.5,  # Centre title
#                 'xanchor': 'center',
#                 'y': .97,  # Move title up a bit
#                 'yanchor': 'middle',
#                 # 'yref': 'paper'
#             },
#             showlegend=False,
#             scene=dict(  # Add axis titles
#                 xaxis_title='PC 1',
#                 yaxis_title='PC 2',
#                 zaxis_title='PC 3',
#                 # xaxis=dict(nticks=4, range=[-100, 100], ), # Specify axis range
#                 # yaxis=dict(nticks=4, range=[-50, 100], ),
#                 # zaxis=dict(nticks=4, range=[-100, 100], ), ),
#                 # xaxis=dict( # Customize axis ticks
#                 # ticktext=['TICKS', 'MESH', 'PLOTLY', 'PYTHON'],
#                 # tickvals=[0, 50, 75, -50]),
#                 # yaxis=dict(
#                 #     nticks=5, tickfont=dict(
#                 #         color='green',
#                 #         size=12,
#                 #         family='Old Standard TT, serif', ),
#                 #     ticksuffix='#'),
#                 # zaxis=dict(
#                 #     nticks=4, ticks='outside',
#                 #     tick0=0, tickwidth=4), ),
#                 camera=vars.fig_phase_space_camera_dict_middle,
#                 # aspectratio=dict(x=1, y=1, z=0.7),
#                 aspectmode='cube'  # manual
#             ),
#         )
#
#     # Update general layout
#     # fig_phase_space.update_layout(
#
#     # legend=dict(
#     #     # yanchor="bottom",
#     #     # y=vars.legend_y_coord,
#     #     # xanchor="left",
#     #     # x=vars.legend_x_coord,
#     #     font=dict(
#     #         size=vars.fig_PC_label_font_size,
#     #         # color="black"
#     #     ),
#     # ),
#     # legend_tracegroupgap=5, # Distance between legend groups in pixels
#
#     # )
#     fig_phase_space.show()
#
#     ## Save as html and png
#     # Create .png file of plot
#     fig_phase_space.write_image(filepath_fig_phase_space_3D.with_suffix(".png"))
#
#     ## Create stand-alone html of the plot with header
#     header = "Subject: {subject}, session: {session}, task: {task}, run: {run}".format(
#         subject=dict_ent["subject"],
#         session=dict_ent["session"],
#         run=dict_ent["run"],
#         task=dict_ent["task"])
#
#     # fig_timeseries_string = """##{header}  \n""".format(header=header)  # Initialize html python with header
#     fig_phase_space_string = """##{header}  \n  <br>  """.format(header=header)  # Initialize html python with header
#     fig_phase_space_string += """{fig_phase_space_html_code}  \n"""  # Create placeholder for figure
#
#     fig_phase_space_html_template = markdown.markdown(fig_phase_space_string)  # Convert markdown to html
#     fig_phase_space_html_complete = fig_phase_space_html_template.replace("{fig_phase_space_html_code}",
#                                                                           fig_phase_space.to_html(full_html=False,
#                                                                                                   include_plotlyjs='cdn'))  # Finialize html by replacing placeholder with html figure python
#
#     with open(filepath_fig_phase_space_3D, 'w') as f:
#         f.write(fig_phase_space_html_complete)  # Create complete html file
#
#         # f.write(fig_logdata.to_html(full_html=True, include_plotlyjs='cdn'))
#
#     return fig_phase_space


def plot_timeseries(Schaefer_ROIs_df, nr_PCs, TR, dict_ent, layout_der, pattern_derivatives_output,
                    mask_unstand_or_stand = "mask_stand", raw_or_PC_unstand_or_stand = "PC_unstand"
                    ):
    """

    :param Schaefer_ROIs_df:
    :param dict_ent:
    :param layout_der:
    :param pattern_derivatives_output:
    :return:
    """

    # 'timeseries_or_figs': 'figs',
    # 'mask_unstand_or_stand': 'mask_unstand',
    # 'raw_or_PC_unstand_or_stand': 'raw',
    # 'type_of_fig': 'brain'
    dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
    dict_ent['raw_or_PC_unstand_or_stand'] = raw_or_PC_unstand_or_stand
    dict_ent['suffix'] = "PCs" if "PC" in raw_or_PC_unstand_or_stand else "mask"

    # Build path for output figure
    filepath_fig_timeseries = pathlib.Path(
        layout_der.build_path(
            {**dict_ent, 'extension': ".html", 'timeseries_or_figs': 'figs',
             'type_of_fig': 'timeseries'},
            pattern_derivatives_output, validate=False,
            absolute_paths=True))

    # Prepare for the type of plot (timeseries, standardized timeseries, PCs)

    # Specify subparts of title of plot
    title_prefix = "PCs" if "PC" in raw_or_PC_unstand_or_stand else "Timeseries"
    if raw_or_PC_unstand_or_stand == "raw":
        title_suffix = "unstandardized" if "unstand" in mask_unstand_or_stand else "standardized"
    else:
        title_suffix = "unstandardized" if "unstand" in raw_or_PC_unstand_or_stand else "standardized"

    subtitle = "Explained variance per PC shown between brackets (%)" if "PC" in raw_or_PC_unstand_or_stand else "Error bands show +/- 1SD"
    y_title = "BOLD signal ({})".format(title_suffix)

    master_title = "<b>{} per ROI ({})</b><br><sup>{}</sup>".format(title_prefix, title_suffix, subtitle)  # Master title of subplot figure

    # Define subplot titles and number of rows and columns
    subplot_titles = ["<b>ROI {}: {} ({} voxels)</b>".format(label_nr + 1,
                                                             Schaefer_ROIs_df["ROI_label"][label_nr].replace(
                                                                 "7Networks_", "").replace("Default_", "").replace(
                                                                 "_", " ").replace("RH", "Right").replace("LH",
                                                                                                          "Left"),
                                                             Schaefer_ROIs_df["nr_voxels"][label_nr]) for label_nr
                      in range(len(Schaefer_ROIs_df["ROI_label"].values))]
    nr_rows = int(np.ceil(len(Schaefer_ROIs_df["ROI_label"].values) / 2))
    nr_cols = 2
    x_title = "Time (sec)"

    # Create plotly subplot figure
    fig_timeseries = make_subplots(
        rows=nr_rows, cols=nr_cols,
        subplot_titles=subplot_titles,
        # shared_xaxes=True,
        x_title=x_title,
        y_title=y_title,
        vertical_spacing=.15,  # Vertical spacing indicates the distance between the columns of the subplots
        horizontal_spacing=.175,  # Horizontal spacing indicates the distance between the rows of the subplots
        # row_heights=[.2, .2, .2, .2],
    )

    # df_row = Schaefer_ROIs_df.iloc[0, :]
    Schaefer_ROIs_df.apply(
        lambda x: add_timeseries_to_subplot(df_row=x,
                                            TR=TR,
                                            fig_timeseries=fig_timeseries,
                                            subplot_titles=subplot_titles,
                                            master_title=master_title,
                                            x_title=x_title, y_title=y_title,
                                            nr_rows=nr_rows,
                                            nr_cols=nr_cols,
                                            nr_PCs=nr_PCs,
                                            Schaefer_ROIs_df=Schaefer_ROIs_df,
                                            dict_ent=dict_ent,
                                            layout_der=layout_der,
                                            pattern_derivatives_output=pattern_derivatives_output,
                                            filepath_fig_timeseries=filepath_fig_timeseries
                                            ),
        axis=1)

    ## Overall figure styling
    ## Adjust font sizes of axis labels and titles
    for annotation in fig_timeseries['layout']['annotations']:
        if "ROI" in annotation['text']:
            annotation['y'] += .01  # Move title up a bit (y-coordinate)

            # if which_type == "unstand_timeseries":
            #     annotation['y'] += .02  # Move title up a bit (y-coordinate)
            # else:
            #     annotation['y'] += .05  # Move title up a bit (y-coordinate)
            annotation['font'] = dict(size=vars.fig_timeseries_subplot_title_font_size)  # Increase font sizes title
        # annotation['x'] = .5  # Make sure title is centred

    for i in range(len(Schaefer_ROIs_df["ROI_label"].values)):
        # Adjust tick font size and number of ticks
        fig_timeseries['layout']['yaxis{}'.format(i + 1 if i != 0 else "")]['tickfont'] = {
            'size': vars.fig_timeseries_subplot_tick_font_size}
        fig_timeseries['layout']['xaxis{}'.format(i + 1 if i != 0 else "")]['tickfont'] = {
            'size': vars.fig_timeseries_subplot_tick_font_size}
        fig_timeseries['layout']['xaxis{}'.format(i + 1 if i != 0 else "")][
            'nticks'] = vars.fig_timeseries_subplot_nr_ticks

    # If the axis has a title, increase font size:
    # if 'title' in fig_timeseries['layout']['xaxis{}'.format(i + 1 if i != 0 else "")].__str__():
    # fig_timeseries['layout']['xaxis{}'.format(i + 1 if i != 0 else "")]['title']['font'][
    #     'size'] = vars.fig_timeseries_subplot_axis_title_font_size  # Increase font size axis title
    for annotation in fig_timeseries['layout']['annotations']:
        # print("annotation:\n{}".format(annotation))
        # print("annotation['text']:\n{}".format(annotation['text']))
        # print("((annotation['text'] == x_title) | (annotation['text'] == y_title))\n{}".format(((annotation['text'] == x_title) | (annotation['text'] == y_title))))
        # if ((annotation['text'] == x_title) | (annotation['text'] == y_title)):
        #     print("annotation['font']\n{}".format(annotation['font']))
        # print("annotation['text']['font']['size']\n{}".format( annotation['text']['font']['size']))
        # if ((fig_timeseries['layout']['annotations']['text'] == x_title) | (fig_timeseries['layout']['annotations']['text'] == y_title)):
        if ((annotation['text'] == x_title) | (annotation['text'] == y_title)):
            annotation['font'][
                'size'] = vars.fig_timeseries_subplot_axis_title_font_size  # Increase font size axis title

    # if 'title' in fig_timeseries['layout']['yaxis{}'.format(i + 1 if i != 0 else "")].__str__():
    #     fig_timeseries['layout']['yaxis{}'.format(i + 1 if i != 0 else "")]['title']['font'][
    #         'size'] = vars.fig_timeseries_subplot_axis_title_font_size  # Increase font size title

    # Update general layout
    fig_timeseries.update_layout(
        title={  # Master title
            'text': master_title,
            'font': dict(size=vars.fig_timeseries_title_font_size),
            'pad': vars.margins_fig_timeseries_title,  # Add padding between master title and plot
            'x': 0.5,  # Centre title
            'xanchor': 'center',
            'y': .97,  # Move title up a bit
            'yanchor': 'middle',
            # 'yref': 'paper'
        },
        height=vars.fig_timeseries_height,
        width=vars.fig_timeseries_width,
        showlegend=True,
        legend=dict(
            # yanchor="bottom",
            # y=vars.legend_y_coord,
            # xanchor="left",
            # x=vars.legend_x_coord,
            font=dict(
                size=vars.fig_PC_label_font_size,
                # color="black"
            ),
        ),
        legend_tracegroupgap=5,  # Distance between legend groups in pixels
        plot_bgcolor="white",  # Change background colour
        paper_bgcolor='white',
        font_family=vars.font_family,  # Change font family
        margin=vars.margins_fig_timeseries,  # Adjust margins figure
    )

    # Don't show legend for mask timeseries
    if dict_ent["raw_or_PC_unstand_or_stand"] == "raw":
        fig_timeseries.update_layout(showlegend=False)

    fig_timeseries.show()

    ## Save as html and png

    # Create .png file of plot
    fig_timeseries.write_image(filepath_fig_timeseries.with_suffix(".png"))

    ## Create stand-alone html of the plot with header
    header = "Subject: {subject}, session: {session}, task: {task}, run: {run}".format(
        subject=dict_ent["subject"],
        session=dict_ent["session"],
        run=dict_ent["run"],
        task=dict_ent["task"])

    # fig_timeseries_string = """##{header}  \n""".format(header=header)  # Initialize html python with header
    fig_timeseries_string = """##{header}  \n  <br>  """.format(header=header)  # Initialize html python with header
    fig_timeseries_string += """{fig_timeseries_html_code}  \n"""  # Create placeholder for figure

    fig_timeseries_html_template = markdown.markdown(fig_timeseries_string)  # Convert markdown to html
    fig_timeseries_html_complete = fig_timeseries_html_template.replace("{fig_timeseries_html_code}",
                                                                        fig_timeseries.to_html(full_html=False,
                                                                                               include_plotlyjs='cdn'))  # Finialize html by replacing placeholder with html figure python

    with open(filepath_fig_timeseries, 'w') as f:
        f.write(fig_timeseries_html_complete)  # Create complete html file

        # f.write(fig_logdata.to_html(full_html=True, include_plotlyjs='cdn'))

    return fig_timeseries


def add_timeseries_to_subplot(df_row, TR, fig_timeseries, subplot_titles, master_title, x_title, y_title, nr_rows,
                              nr_cols, nr_PCs, Schaefer_ROIs_df, dict_ent, layout_der,
                              pattern_derivatives_output, filepath_fig_timeseries
                              ):
    # df_row = Schaefer_ROIs_df.iloc[0,:]

    # Find label number, row number, and column number
    mask_label = df_row["ROI_label"]
    label_nr = np.where(Schaefer_ROIs_df["ROI_label"].values == mask_label)[0][0]
    row_nr = np.repeat(np.arange(1, nr_rows + 1), nr_cols)[label_nr]
    col_nr = (list(range(1, nr_cols + 1)) * nr_rows)[label_nr]

    # dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
    # dict_ent['raw_or_PC_unstand_or_stand'] = raw_or_PC_unstand_or_stand
    dict_ent['suffix'] = "PCs_{}".format(mask_label) if "PC" in dict_ent["raw_or_PC_unstand_or_stand"] else "mask_{}".format(mask_label)
    dict_ent['extension'] = ".csv" if dict_ent["raw_or_PC_unstand_or_stand"] == "raw" else ".json"
    dict_ent['timeseries_or_figs'] = "timeseries"
    if "type_of_fig" in dict_ent.keys(): # Make sure the type_of_fig key is not in the dictionary for reading the data
        del dict_ent["type_of_fig"]

    ## Load data ((standardized) timeseries of all ROIs or PCs of all ROIs)
    # Build path by filling in dict_ent (making sure to include the appropriate suffix and extension) in pattern_derivatives_output to save timeseries file in derivatives directory
    filepath_masked_data = pathlib.Path(
        layout_der.build_path({**dict_ent}, pattern_derivatives_output, validate=False,
                              absolute_paths=True))

    # Read data
    if dict_ent['extension'] == ".csv":
        masked_data = pd.read_csv(filepath_masked_data, sep='\t')  # Read csv file

        # Demean if unstandardized version; before we compute the mean timeseries across voxels, demean each voxel's timeseries using its own mean
        if dict_ent["mask_unstand_or_stand"] == "mask_unstand":
            masked_data = (masked_data - masked_data.mean(axis=0))

            # Same result when substracting mean of mean timeseries and when demeaning before, i.e.
            # mean_timeseries - np.array(mean_timeseries).mean()

        # Prepare data for plotting (time vector, mean, and error bands which show standard error of the mean)
        # Define time vector with respect to image acquisition
        time_vec = list(np.arange(0, TR * masked_data.shape[0], TR))  # Time vector in seconds, steps of TR
        mean_timeseries = list(masked_data.mean(axis=1))  # Mean time series
        std_lower_timeseries = list(mean_timeseries - masked_data.std(axis=1))  # Line for lower error band
        std_upper_timeseries = list(mean_timeseries + masked_data.std(axis=1))  # Line for upper error band
        sem_lower_timeseries = list(mean_timeseries - (masked_data.std(axis=1, ddof=1) / np.sqrt(
            masked_data.shape[1])))  # Line for lower error band: standard error of the mean
        sem_upper_timeseries = list(mean_timeseries + (masked_data.std(axis=1, ddof=1) / np.sqrt(
            masked_data.shape[1])))  # Line for upper error band: standard error of the mean

        # Mean time series
        timeseries_scatter = go.Scatter(
            x=time_vec,
            y=mean_timeseries,
            # line=dict(color=vars.ROI_colours[label_nr], width=vars.line_width_timeseries_mean),
            line=dict(color=vars.ROI_colours[label_nr], width=vars.line_width_timeseries_mean),
            # name="",
            mode='lines'
        )
        # Error bands
        timeseries_error_scatter = go.Scatter(
            x=time_vec + time_vec[::-1],  # Time vector, then time vector reversed
            y=std_upper_timeseries + std_lower_timeseries[::-1],  # Upper, then lower reversed
            fill='toself',
            fillcolor=rgb_to_rgba(hex_to_rgb(vars.ROI_colours[label_nr]), vars.alpha_val_timeseries_error_bands),
            # Convert to rgba
            # line=dict(color=vars.ROI_colours[label_nr]), # Convert to rgba
            line=dict(
                color=rgb_to_rgba(hex_to_rgb(vars.ROI_colours[label_nr]), vars.alpha_val_timeseries_error_bands)),
            # Convert to rgba
            hoverinfo="skip",
            showlegend=False
        )

        # Plot mean time series
        fig_timeseries.add_trace(timeseries_scatter, row=row_nr, col=col_nr)

        # Plot error bands
        fig_timeseries.add_trace(timeseries_error_scatter, row=row_nr, col=col_nr)

        # Save stand-alone plot
        save_subplot_timeseries(label_nr, mask_label, subplot_titles, x_title, y_title,
                                filepath_fig_timeseries, dict_ent, layout_der, pattern_derivatives_output, nr_PCs,
                                data={"timeseries_scatter": timeseries_scatter,
                                      "timeseries_error_scatter": timeseries_error_scatter})

    elif dict_ent['extension'] == ".json":
        with open(filepath_masked_data, 'r') as file:
            masked_data = json.load(file)  # Read .json file

        # Plot details
        # Define time vector with respect to image acquisition - the same for every PC
        time_vec = list(np.arange(0, TR * len(masked_data["PC1"]), TR))  # Time vector in seconds, steps of TR

        # Define dictionaries to save plotly objects to
        scatters = {}  # Scatter plotly objects
        annos = {}  # Annotations
        annotation_y_PCs = [.65, 0.25, -.2]  # Vector of y coordinates for PCs
        annotation_y_all_PCs = 1.05

        # General annotation with summed explained variance
        expl_var_PCs_total = np.round(np.sum(masked_data['explained_variance_ratio_']) * 100, 1)

        annos["all_PCs"] = dict(
            x=1.025,
            y=annotation_y_all_PCs,
            xref='x domain',
            yref='y domain',
            axref='x domain',
            ayref='y domain',
            ax=1.025,
            ay=annotation_y_all_PCs,
            xanchor='left',
            yanchor='bottom',
            text=
            # "PC {}<br>({} %)".format(PC_nr,
            "Total: {}%".format(expl_var_PCs_total),
            showarrow=False,
            font=dict(
                # family="Courier New, monospace",
                size=vars.fig_PC_label_font_size + 1,
                color=tint_or_shade_rgb(rgb_str=hex_to_rgb(vars.ROI_colours[label_nr]),
                                        factor=.9,
                                        shade_or_tint="shade"),
            )
        )

        # Loop through PCs and create plotly objects
        for PC_nr in range(1, nr_PCs + 1):
            # Plot component line through time
            scatters["PC{}".format(PC_nr)] = go.Scatter(
                x=time_vec,
                y=masked_data["PC{}".format(PC_nr)],
                # line=dict(color=vars.ROI_colours[label_nr], width=vars.line_width_timeseries_mean),
                line=dict(color=tint_or_shade_rgb(rgb_str=hex_to_rgb(vars.ROI_colours[label_nr]),
                                                  factor=vars.fig_PC_shade_tint_factors[PC_nr - 1],
                                                  shade_or_tint=vars.fig_PC_shade_or_tint[PC_nr - 1]),
                          width=vars.fig_PC_lwd[PC_nr - 1]),
                name="PC {} ({}%)".format(PC_nr,
                                          np.round(masked_data['explained_variance_ratio_'][PC_nr - 1] * 100, 1)),
                legendgroup="{}".format(mask_label),
                legendgrouptitle={"text": annos["all_PCs"]["text"],
                                  "font": {"size": vars.fig_PC_label_font_size + 1}
                                  },
                mode='lines'
            )

            # Annotations
            annos["PC{}".format(PC_nr)] = dict(
                x=1.025,
                y=annotation_y_PCs[PC_nr - 1],
                # , xref = 'paper'
                # , yref = 'paper'
                xref='x domain',
                yref='y domain',
                axref='x domain',
                ayref='y domain',
                ax=1.025,
                ay=annotation_y_PCs[PC_nr - 1],
                xanchor='left',
                yanchor='bottom',
                text=
                # "PC {}<br>({} %)".format(PC_nr,
                "PC {} ({}%)".format(PC_nr, np.round(masked_data['explained_variance_ratio_'][PC_nr - 1] * 100, 1)),
                showarrow=False,
                font=dict(
                    # family="Courier New, monospace",
                    size=vars.fig_PC_label_font_size,
                    color=tint_or_shade_rgb(rgb_str=hex_to_rgb(vars.ROI_colours[label_nr]),
                                            factor=vars.fig_PC_shade_tint_factors[PC_nr - 1],
                                            shade_or_tint=vars.fig_PC_shade_or_tint[PC_nr - 1]),
                    # "white",#
                )
            )

            # Plot PC
            fig_timeseries.add_trace(scatters["PC{}".format(PC_nr)], row=row_nr, col=col_nr)

            # # Add explained variance annotation for PC
            # fig_timeseries.add_annotation(annos["PC{}".format(PC_nr)], row=row_nr, col=col_nr)

        # # Add summed explained variance annotation of all PCs
        # fig_timeseries.add_annotation(annos["all_PCs"], row=row_nr, col=col_nr)

        # Save subplot
        save_subplot_timeseries(label_nr, mask_label, subplot_titles, x_title, y_title,
                                filepath_fig_timeseries, dict_ent, layout_der, pattern_derivatives_output, nr_PCs,
                                data={"scatters": scatters,
                                      "annos": annos
                                      # "PC1_scatter": PC1_scatter, "PC2_scatter": PC2_scatter, "PC1_anno": PC1_anno,
                                      #   "PC2_anno": PC2_anno
                                      })

    # For standardized timeseries, show fixed y-range for all
    if ((dict_ent["mask_unstand_or_stand"] == "mask_unstand") & (dict_ent["raw_or_PC_unstand_or_stand"] == "raw")):
        fig_timeseries.update_yaxes(row=row_nr, col=col_nr,
                                    range=[vars.fig_unst_time_y_range_min, vars.fig_unst_time_y_range_max])
    elif ((dict_ent["mask_unstand_or_stand"] == "mask_stand") & (dict_ent["raw_or_PC_unstand_or_stand"] == "raw")):
        fig_timeseries.update_yaxes(row=row_nr, col=col_nr,
                                    range=[vars.fig_st_time_y_range_min, vars.fig_st_time_y_range_max])
    elif "unstand" in dict_ent["raw_or_PC_unstand_or_stand"]:
        fig_timeseries.update_yaxes(row=row_nr, col=col_nr, range=[vars.fig_unst_PC_y_range_min, vars.fig_unst_PC_y_range_max])
    elif "_stand" in dict_ent["raw_or_PC_unstand_or_stand"]:
        fig_timeseries.update_yaxes(row=row_nr, col=col_nr, range=[vars.fig_st_PC_y_range_min, vars.fig_st_PC_y_range_max])

    # Add axis lines
    fig_timeseries.update_xaxes(row=row_nr, col=col_nr, showgrid=True, gridwidth=vars.grid_width,
                                gridcolor=vars.grid_color,
                                zeroline=True, zerolinewidth=vars.grid_width,
                                zerolinecolor=vars.grid_color)
    fig_timeseries.update_yaxes(row=row_nr, col=col_nr, showgrid=True, gridwidth=vars.grid_width,
                                gridcolor=vars.grid_color,
                                zeroline=True, zerolinewidth=vars.grid_width,
                                zerolinecolor=vars.grid_color)

    return None


def save_subplot_timeseries(label_nr, mask_label, subplot_titles, x_title, y_title, filepath_fig_timeseries,
                            dict_ent, layout_der, pattern_derivatives_output, nr_PCs, data):
    # Modify file path to include mask label
    filepath_fig = filepath_fig_timeseries.with_name("{}_{}".format(filepath_fig_timeseries.stem, mask_label))

    # Create plotly subplot figure
    fig = make_subplots(
        rows=1, cols=1,
        # subplot_titles=subplot_titles[label_nr],
        x_title=x_title,
        y_title=y_title,
    )

    # fig['layout']['annotations']

    if dict_ent["raw_or_PC_unstand_or_stand"] == "raw":
        # Increase line width
        data["timeseries_scatter"]['line']['width'] += .75

        # Plot mean time series
        fig.add_trace(data["timeseries_scatter"])
        # Plot error bands
        fig.add_trace(data["timeseries_error_scatter"])

    elif "PC" in dict_ent["raw_or_PC_unstand_or_stand"]:

        # Loop through PCs and add line and annotation
        for PC_nr in range(1, nr_PCs + 1):
            # Plot PC
            data["scatters"]["PC{}".format(PC_nr)]["line"]["width"] += 2  # Increase line width
            fig.add_trace(data["scatters"]["PC{}".format(PC_nr)])

            # # Add explained variance annotation for PC
            # fig.add_annotation(data["annos"]["PC{}".format(PC_nr)])

        # # Add summed explained variance annotation of all PCs
        # fig.add_annotation(data["annos"]["all_PCs"])
        subplot_titles[label_nr] += "<br><sup>Explained variance per PC shown between brackets (%)</sup>"  # Master title of subplot figure

    # For standardized timeseries, show fixed y-range for all
    if ((dict_ent["mask_unstand_or_stand"] == "mask_unstand") & (dict_ent["raw_or_PC_unstand_or_stand"] == "raw")):
        fig.update_yaxes(range=[vars.fig_unst_time_y_range_min, vars.fig_unst_time_y_range_max])
    elif ((dict_ent["mask_unstand_or_stand"] == "mask_stand") & (dict_ent["raw_or_PC_unstand_or_stand"] == "raw")):
        fig.update_yaxes(range=[vars.fig_st_time_y_range_min, vars.fig_st_time_y_range_max])
    elif "unstand" in dict_ent["raw_or_PC_unstand_or_stand"]:
        fig.update_yaxes(range=[vars.fig_unst_PC_y_range_min, vars.fig_unst_PC_y_range_max])
    elif "_stand" in dict_ent["raw_or_PC_unstand_or_stand"]:
        fig.update_yaxes(range=[vars.fig_st_PC_y_range_min, vars.fig_st_PC_y_range_max])

    # Add axis lines
    fig.update_xaxes(showgrid=True, gridwidth=vars.grid_width,
                     gridcolor=vars.grid_color,
                     zeroline=True, zerolinewidth=vars.grid_width,
                     zerolinecolor=vars.grid_color)
    fig.update_yaxes(showgrid=True, gridwidth=vars.grid_width,
                     gridcolor=vars.grid_color,
                     zeroline=True, zerolinewidth=vars.grid_width,
                     zerolinecolor=vars.grid_color)

    ## Overall figure styling

    # Adjust tick font size and number of ticks
    fig['layout']['yaxis']['tickfont'] = {
        'size': vars.fig_timeseries_subplot_tick_font_size + 4}
    fig['layout']['xaxis']['tickfont'] = {
        'size': vars.fig_timeseries_subplot_tick_font_size + 4}
    fig['layout']['xaxis'][
        'nticks'] = vars.fig_timeseries_subplot_nr_ticks

    # Increase font size axis titles:
    # annotation_y_PCs = [.85, 0.65, .45]  # Vector of y coordinates for annotations of PCs
    annotation_y_PCs = [.7, 0.5, .3]  # Vector of y coordinates for annotations of PCs
    annotation_y_all_PCs = .9  # 1.05

    for annotation in fig['layout']['annotations']:
        if ((annotation['text'] == x_title) | (annotation['text'] == y_title)):
            annotation['font'][
                'size'] = vars.fig_timeseries_subplot_axis_title_font_size + 6  # Increase font size axis title
        # if "PC" in annotation['text']:
        #     annotation['font'][
        #         'size'] = vars.fig_timeseries_subplot_axis_title_font_size  # Increase font size PC label
        # for PC_nr in range(1, nr_PCs + 1):
        #     if "PC {}".format(PC_nr) in annotation['text']:  # Change position PC annotation
        #         annotation['y'] = annotation_y_PCs[PC_nr - 1]
        #         annotation['ay'] = annotation_y_PCs[PC_nr - 1]
        #     if "Total:" in annotation['text']:  # Change position summed explained variance annotation of all PCs
        #
        #         annotation['font'][
        #             'size'] = vars.fig_timeseries_subplot_axis_title_font_size+1  # Increase font size PC label
        # annotation['y'] = annotation_y_all_PCs
        # annotation['ay'] = annotation_y_all_PCs

    # Update general layout
    fig.update_layout(
        title={  # Master title
            'text': subplot_titles[label_nr],
            'font': dict(size=vars.fig_timeseries_title_font_size + 6),
            'pad': vars.margins_fig_timeseries_title,  # Add padding between master title and plot
            'x': 0.5,  # Centre title
            'xanchor': 'center',
            'y': .9,  # Move title up a bit
            'yanchor': 'middle',
            # 'yref': 'paper'
        },
        height=vars.fig_timeseries_height - 350, width=vars.fig_timeseries_width,
        showlegend=False,
        plot_bgcolor="white",  # Change background colour
        paper_bgcolor='white',
        font_family=vars.font_family,  # Change font
        margin=vars.margins_fig_timeseries_subplot_fig,
    )

    if "PC" in dict_ent["raw_or_PC_unstand_or_stand"]:
        fig.update_layout(
            showlegend=True,
            # legend_title_text = data["annos"]["all_PCs"]["text"],
            legend=dict(
                yanchor="bottom",
                y=vars.legend_y_coord,
                xanchor="left",
                x=vars.legend_x_coord,
                font=dict(
                    size=vars.fig_PC_label_font_size,
                    # color="black"
                ),
            ))

    # Create .png file of plot
    fig.write_image(filepath_fig.with_suffix(".png"))

    ## Create stand-alone html of the plot with header
    header = "Subject: {subject}, session: {session}, task: {task}, run: {run}".format(
        subject=dict_ent["subject"],
        session=dict_ent["session"],
        run=dict_ent["run"],
        task=dict_ent["task"])

    # fig_timeseries_string = """##{header}  \n""".format(header=header)  # Initialize html python with header
    fig_string = """##{header}  \n  <br>  """.format(header=header)  # Initialize html python with header
    fig_string += """{fig_html_code}  \n"""  # Create placeholder for figure

    fig_html_template = markdown.markdown(fig_string)  # Convert markdown to html
    fig_html_complete = fig_html_template.replace("{fig_html_code}", fig.to_html(full_html=False,
                                                                                 include_plotlyjs='cdn'))  # Finialize html by replacing placeholder with html figure python

    with open(filepath_fig.with_suffix(".html"), 'w') as f:
        f.write(fig_html_complete)  # Create complete html file

    return None


# Function to rescale axis to data
def define_range_axis(vector):
    return [np.round(np.min(vector) - np.round(np.std(vector) / 3)),
            np.round(np.max(vector) + np.round(np.std(vector) / 3))]


