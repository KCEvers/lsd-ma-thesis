

# Import packages
import vars
from thesis.master.globalimports import *
from thesis.processing import hex_to_rgb, rgb_to_rgba, tint_or_shade_rgb, rgb_to_rgba_255, acf_wrapper

# Convert list to pandas dataframe and use the first entry as column headers
def convert_list_to_df(l):
    return pd.DataFrame(l[1:], index=None, columns=l[0])


def c2_d2_h2_estimate_plot(dict_ent, layout_der, pattern_derivatives_output, Schaefer_ROIs_df, nr_PCs, mask_unstand_or_stand, raw_or_PC_unstand_or_stand, theiler_window=0, tau = 1, emb_dim=1, max_slope=1, max_residuals=.1, min_rsquared=.4
                   # scaling_region=[10**(-.8), 10**0]
              ):
    # (1/tau) -> "Do not forget to divide the h2-estimate by the time lag" (https://www.pks.mpg.de/tisean//Tisean_3.0.1/docs/tutorial/ex4.html)
    # Check whether this is necessary still!

    # Build input file path
    dict_ent['pipeline'] = 'timeseries_analysis'
    dict_ent['extension'] = '.csv'
    dict_ent['suffix'] = 'PCs'
    dict_ent['timeseries_or_figs'] = 'timeseries'
    dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
    dict_ent['raw_or_PC_unstand_or_stand'] = raw_or_PC_unstand_or_stand

    # Define file path for output d2 estimate figure
    filepath_c2_d2_h2_estimate_fig = pathlib.Path(
        layout_der.build_path({**dict_ent,
                               'timeseries_or_figs': 'figs',
                               'type_of_fig': 'd2',
                               'extension':'.png',
                               'suffix': "{}_{}_theiler_{}".format(dict_ent['suffix'], 'd2-h2-estimate', theiler_window)},
                              pattern_derivatives_output, validate=False,
                              absolute_paths=True))

    nr_ROIs = Schaefer_ROIs_df.shape[0]

    # Define file path for output d2 estimate
    filepath_d2_h2_estimate = pathlib.Path(
        layout_der.build_path({**dict_ent,
                               'extension': '.json',
                               'suffix': "{}_{}_theiler_{}".format(dict_ent['suffix'], 'd2-h2-estimate', theiler_window)},
                              pattern_derivatives_output, validate=False,
                              absolute_paths=True))

    # Read d2 and h2 estimate data
    with open(filepath_d2_h2_estimate, 'r') as file:
        d2_h2_estimate_dict = json.load(file)  # Read .json file

    d2_estimate = d2_h2_estimate_dict["global_d2_estimate"]
    d2_eps_scaling_region_min, d2_eps_scaling_region_max = d2_h2_estimate_dict["global_d2_eps_scaling_region"]
    d2_df_select = convert_list_to_df(d2_h2_estimate_dict["d2_df_select"])

    h2_estimate = d2_h2_estimate_dict["global_h2_estimate"]
    h2_eps_scaling_region_min, h2_eps_scaling_region_max = d2_h2_estimate_dict["global_h2_eps_scaling_region"]
    h2_df_select = convert_list_to_df(d2_h2_estimate_dict["h2_df_select"])

    # Plotly
    subplot_titles = ['$\\text{{Correlation sum }}C_2^m(\epsilon)$'.format(),
                      "$\\text{{Correlation dimension }}D_2^m(\epsilon)\\approx{0:.2f}$".format(d2_estimate), # d2_eps_scaling_region_min, d2_eps_scaling_region_max
                      "$\\text{{Correlation entropy }}h_2^m(\epsilon)\\approx{0:.2f}$".format(h2_estimate) # , h2_eps_scaling_region_min, h2_eps_scaling_region_max
                      ]

    # Create plot skeleton
    fig_c2_d2_h2_estimate = make_subplots(
        rows=1, cols=3,
        subplot_titles=subplot_titles,
        # shared_xaxes=True,
        # shared_yaxes=False,
        # x_title=x_title,
        # y_title=y_title,
        # vertical_spacing=.175,  # Vertical spacing indicates the distance between the rows of the subplots
        horizontal_spacing=.15,  # Horizontal spacing indicates the distance between the columns of the subplots
        # row_heights=[.2, .2, .2, .2],
    )

    # # Plot horizontal line for d2 estimate
    # fig_c2_d2_h2_estimate.add_shape(type="line",
    #                           x0=0, y0=d2_estimate, x1=1, y1=d2_estimate,
    #                           xref='paper',
    #                           name="$D_2$ estimate",
    #                           # By specifying xref='paper', we can let the line span the entire x-axis using x0=0 and x1=1
    #                           line=dict(
    #                               color=vars.fig_d2_colour_d2_estimate,
    #                               width=4,
    #                               dash="dash",
    #                           ), row=1, col=2
    #                           )
    # # Plot horizontal line for h2 estimate
    # fig_c2_d2_h2_estimate.add_shape(type="line",
    #                                 x0=0, y0=h2_estimate, x1=1, y1=h2_estimate,
    #                                 xref='paper',
    #                                 name="$h_2$ estimate",
    #                                 # By specifying xref='paper', we can let the line span the entire x-axis using x0=0 and x1=1
    #                                 line=dict(
    #                                     color=vars.fig_d2_colour_d2_estimate,
    #                                     width=4,
    #                                     dash="dash",
    #                                 ), row=1, col=3
    #                                 )

    # Add estimate horizontal lines
    fig_c2_d2_h2_estimate.add_trace(go.Scatter(
        x=[0, 100],
        y=[d2_estimate, d2_estimate],
        mode="lines",
        line=dict(color=vars.fig_d2_colour_d2_estimate,
                  width=vars.line_width_timeseries_mean),
        name="Estimate",
        showlegend=True,
    ), row=1, col=2)
    fig_c2_d2_h2_estimate.add_trace(go.Scatter(
        x=[0,100],
        y=[h2_estimate, h2_estimate],
        mode="lines",
        line=dict(color=vars.fig_d2_colour_d2_estimate,
                  width=vars.line_width_timeseries_mean),
        # name="Scaling region",
        showlegend=False,
    ), row=1, col=3)

    # Add annotation for scaling region

    fig_c2_d2_h2_estimate.add_trace(go.Scatter(
        x=[10],#[(d2_eps_scaling_region_min + d2_eps_scaling_region_max)/2],
        y=[24],
        mode="text",
        # name="Scaling region",
        text=["Scaling region [{:.2f},{:.2f}]".format(d2_eps_scaling_region_min, d2_eps_scaling_region_max)],
        # textposition="bottom center",
        showlegend=False,
        textfont=dict(
            family=vars.font_family,
            size=vars.fig_d2_axis_title_font_size+4,
            color=vars.fig_d2_color_inside_scaling_region
        ),
    ), row=1,col=2)

    fig_c2_d2_h2_estimate.add_trace(go.Scatter(
        x=[10],#[(h2_eps_scaling_region_min + h2_eps_scaling_region_max)/2],
        y=[1.5],
        mode="text",
        # name="Scaling region",
        showlegend=False,
        text=["Scaling region [{:.2f},{:.2f}]".format(h2_eps_scaling_region_min, h2_eps_scaling_region_max)],
        # textposition="bottom center",
        textfont=dict(
            family=vars.font_family,
            size=vars.fig_d2_axis_title_font_size+4,
            color=vars.fig_d2_color_inside_scaling_region
        ),
    ), row=1,col=3)

    # Loop through embedding dimension and for each plot C_2, D_2, and h_2 with overlapping scaling regions
    for emb_nr in np.unique(d2_df_select["emb_nr"]):
        # Select only the current embedding dimension
        # d2_df_emb = d2_df[d2_df.emb_nr == emb_nr].sort(by="epsilon")
        d2_df_emb = d2_df_select[d2_df_select.emb_nr == emb_nr].dropna(subset=["d2", "epsilon"]).sort_values(by="epsilon")
        d2_df_emb = d2_df_emb.reset_index(drop=True)  # Reset index without adding new column

        # Find epsilon value closest to global scaling region
        d2_eps_scaling_region_min_emb = min(d2_df_emb["epsilon"], key=lambda x: abs(x - d2_eps_scaling_region_min))
        d2_eps_scaling_region_max_emb = min(d2_df_emb["epsilon"], key=lambda x: abs(x - d2_eps_scaling_region_max))
        d2_eps_scaling_region_min_emb_idx = np.where(d2_df_emb["epsilon"] == d2_eps_scaling_region_min_emb)[0][0]
        d2_eps_scaling_region_max_emb_idx = np.where(d2_df_emb["epsilon"] == d2_eps_scaling_region_max_emb)[0][0]

        # Find already identified scaling region for C_2, D_2, and h_2 in dataframe with specific embedding dimension
        d2_df_scaling_emb = d2_df_emb[d2_eps_scaling_region_min_emb_idx:d2_eps_scaling_region_max_emb_idx + 1]
        # d2_df_scaling_emb = d2_df_scaling[d2_df_scaling.emb_nr == emb_nr]

        # Correlation entropy
        h2_df_emb = h2_df_select[h2_df_select.emb_nr == emb_nr].dropna(subset=["h2", "epsilon"]).sort_values(by="epsilon")
        h2_df_emb = h2_df_emb.reset_index(drop=True)  # Reset index without adding new column

        # Find epsilon value closest to global scaling region
        h2_eps_scaling_region_min_emb = min(h2_df_emb["epsilon"], key=lambda x: abs(x - h2_eps_scaling_region_min))
        h2_eps_scaling_region_max_emb = min(h2_df_emb["epsilon"], key=lambda x: abs(x - h2_eps_scaling_region_max))
        h2_eps_scaling_region_min_emb_idx = np.where(h2_df_emb["epsilon"] == h2_eps_scaling_region_min_emb)[0][0]
        h2_eps_scaling_region_max_emb_idx = np.where(h2_df_emb["epsilon"] == h2_eps_scaling_region_max_emb)[0][0]

        # Find already identified scaling region for C_2, D_2, and h_2 in dataframe with specific embedding dimension
        h2_df_scaling_emb = h2_df_emb[h2_eps_scaling_region_min_emb_idx:h2_eps_scaling_region_max_emb_idx + 1]
        # d2_df_scaling_emb = d2_df_scaling[d2_df_scaling.emb_nr == emb_nr]


        # Correlation sums
        # ax1.loglog(d2_df_emb["epsilon"], d2_df_emb["c2"], color = vars.fig_d2_color_outside_scaling_region, label = "{}".format(emb_nr))
        # ax1.loglog(d2_df_scaling_emb["epsilon"], d2_df_scaling_emb["c2"], color = vars.fig_d2_color_inside_scaling_region, label = "Scaling {}".format(emb_nr))
        # Only add one legend
        # if emb_nr==1:
        #     ## Correlation sums
        #     fig_c2_d2_h2_estimate.add_trace(
        #         go.Scatter(
        #             x=d2_df_emb["epsilon"], y=d2_df_emb["c2"],
        #             line=dict(color=vars.fig_d2_color_outside_scaling_region,
        #                       width=vars.line_width_timeseries_mean),
        #             # showlegend=False,
        #             name="Outside scaling region",
        #             mode='lines'
        #         ), row=1, col=1)
        #     # Correlation sums: scaling region
        #     fig_c2_d2_h2_estimate.add_trace(
        #         go.Scatter(
        #             x=d2_df_scaling_emb["epsilon"], y=d2_df_scaling_emb["c2"],
        #             line=dict(color=vars.fig_d2_color_inside_scaling_region,
        #                       width=vars.line_width_timeseries_mean),
        #             # showlegend=False,
        #             name="Inside scaling region",
        #             mode='lines'
        #         ), row=1, col=1)
        # else:
        ## Correlation sums
        fig_c2_d2_h2_estimate.add_trace(
            go.Scatter(
                x=d2_df_emb["epsilon"], y=d2_df_emb["c2"],
                line=dict(color=vars.fig_d2_color_outside_scaling_region,
                          width=vars.line_width_timeseries_mean),
                showlegend=False,
                mode='lines'
            ), row=1, col=1)
        # Correlation sums: scaling region
        fig_c2_d2_h2_estimate.add_trace(
            go.Scatter(
                x=d2_df_scaling_emb["epsilon"], y=d2_df_scaling_emb["c2"],
                line=dict(color=vars.fig_d2_color_inside_scaling_region,
                          width=vars.line_width_timeseries_mean),
                showlegend=False,
                mode='lines'
            ), row=1, col=1)

        ## Correlation dimension
        fig_c2_d2_h2_estimate.add_trace(
            go.Scatter(
                x=d2_df_emb["epsilon"], y=d2_df_emb["d2"],
                line=dict(color=vars.fig_d2_color_outside_scaling_region,
                          width=vars.line_width_timeseries_mean),
                showlegend=False,
                mode='lines'
            ), row=1, col=2)
        # Correlation dimension: scaling region
        fig_c2_d2_h2_estimate.add_trace(
            go.Scatter(
                x=d2_df_scaling_emb["epsilon"], y=d2_df_scaling_emb["d2"],
                line=dict(color=vars.fig_d2_color_inside_scaling_region,
                          width=vars.line_width_timeseries_mean),
                showlegend=False,
                mode='lines'
            ), row=1, col=2)

        ## Correlation entropy
        fig_c2_d2_h2_estimate.add_trace(
            go.Scatter(
                x=h2_df_emb["epsilon"], y=h2_df_emb["h2"] * (1 / tau),
                line=dict(color=vars.fig_d2_color_outside_scaling_region,
                          width=vars.line_width_timeseries_mean),
                showlegend=False,
                mode='lines'
            ), row=1, col=3)
        # Correlation entropy: scaling region
        fig_c2_d2_h2_estimate.add_trace(
            go.Scatter(
                x=h2_df_scaling_emb["epsilon"], y=h2_df_scaling_emb["h2"] * (1 / tau),
                line=dict(color=vars.fig_d2_color_inside_scaling_region,
                          width=vars.line_width_timeseries_mean),
                showlegend=False,
                mode='lines'
            ), row=1, col=3)

        # (1/tau) -> "Do not forget to divide the h2-estimate by the time lag" (https://www.pks.mpg.de/tisean//Tisean_3.0.1/docs/tutorial/ex4.html)
        # Check whether this is necessary still!

    # Convert plot 1 to log-log plot, and plot 2 and 3 to semi-logx plots
    fig_c2_d2_h2_estimate.update_xaxes(type="log", row=1, col=1) # type="log", range=[0, 5]  # log range: 10^0=1, 10^5=100000
    fig_c2_d2_h2_estimate.update_yaxes(type="log", row=1, col=1)

    fig_c2_d2_h2_estimate.update_xaxes(type="log", range=[0,2], row=1, col=2) # type="log", range=[0, 5]  # log range: 10^0=1, 10^5=100000
    # fig.update_yaxes(type="log", row=1, col=2)

    fig_c2_d2_h2_estimate.update_xaxes(type="log", range=[0,2], row=1, col=3) # type="log", range=[0, 5]  # log range: 10^0=1, 10^5=100000
    # fig.update_yaxes(type="log", row=1, col=3)

    fig_c2_d2_h2_estimate.update_xaxes(showgrid=True, gridwidth=vars.grid_width,
                                 gridcolor=vars.grid_color,
                                 zeroline=True, zerolinewidth=vars.grid_width,
                                 zerolinecolor=vars.grid_color)
    fig_c2_d2_h2_estimate.update_yaxes(showgrid=True, gridwidth=vars.grid_width,
                                 gridcolor=vars.grid_color,
                                 zeroline=True, zerolinewidth=vars.grid_width,
                                 zerolinecolor=vars.grid_color)

    ## Adjust axis labels and ticks
    # Correlation sums (row 1, col 1)
    fig_c2_d2_h2_estimate['layout']['xaxis']['title'] = {'text': r'$\log{\epsilon}$', 'font': {
        'size': vars.fig_stationary_subplot_axis_title_font_size}}  # Increase font size axis title
    fig_c2_d2_h2_estimate['layout']['xaxis'][
        'nticks'] = vars.fig_stationary_subplot_nr_ticks
    fig_c2_d2_h2_estimate['layout']['xaxis']['tickfont'] = {
        'size': vars.fig_stationary_subplot_tick_font_size}

    # tickvals = [5.1, 5.9, 6.3, 7.5]

    fig_c2_d2_h2_estimate['layout']['yaxis']['title'] = {'text': r'$C_2^m(\epsilon)$', 'font': {
        'size': vars.fig_stationary_subplot_axis_title_font_size}}  # Increase font size axis title
    fig_c2_d2_h2_estimate['layout']['yaxis']['nticks'] = vars.fig_stationary_subplot_nr_ticks
    fig_c2_d2_h2_estimate['layout']['yaxis']['tickfont'] = {
        'size': vars.fig_stationary_subplot_tick_font_size}

    # Correlation dimension (row 1, col 2)
    fig_c2_d2_h2_estimate['layout']['xaxis2']['title'] = {'text': r'$\log{\epsilon}$', 'font': {
        'size': vars.fig_stationary_subplot_axis_title_font_size}}  # Increase font size axis title
    fig_c2_d2_h2_estimate['layout']['xaxis2'][
        'nticks'] = 10#vars.fig_stationary_subplot_nr_ticks
    fig_c2_d2_h2_estimate['layout']['xaxis2']['tickfont'] = {
        'size': vars.fig_stationary_subplot_tick_font_size}

    fig_c2_d2_h2_estimate['layout']['yaxis2']['title'] = {'text': r'$D_2^m(\epsilon)$', 'font': {
        'size': vars.fig_stationary_subplot_axis_title_font_size}}  # Increase font size axis title
    # fig_c2_d2_h2_estimate['layout']['yaxis2']['nticks'] = vars.fig_stationary_subplot_nr_ticks
    fig_c2_d2_h2_estimate['layout']['yaxis2']['tickfont'] = {
        'size': vars.fig_stationary_subplot_tick_font_size}

    # Correlation entropy (row 1, col 3)
    fig_c2_d2_h2_estimate['layout']['xaxis3']['title'] = {'text': r'$\log{\epsilon}$', 'font': {
        'size': vars.fig_stationary_subplot_axis_title_font_size}}  # Increase font size axis title
    fig_c2_d2_h2_estimate['layout']['xaxis3'][
        'nticks'] = 10#vars.fig_stationary_subplot_nr_ticks
    fig_c2_d2_h2_estimate['layout']['xaxis3']['tickfont'] = {
        'size': vars.fig_stationary_subplot_tick_font_size}

    fig_c2_d2_h2_estimate['layout']['yaxis3']['title'] = {'text': r'$h_2^m(\epsilon)$', 'font': {
        'size': vars.fig_stationary_subplot_axis_title_font_size}}  # Increase font size axis title
    fig_c2_d2_h2_estimate['layout']['yaxis3']['nticks'] = vars.fig_stationary_subplot_nr_ticks
    fig_c2_d2_h2_estimate['layout']['yaxis3']['tickfont'] = {
        'size': vars.fig_stationary_subplot_tick_font_size}

    # Increase font size subplot titles
    for annotation_nr in range(len(fig_c2_d2_h2_estimate['layout']['annotations'])):
        if "Correlation" in fig_c2_d2_h2_estimate['layout']['annotations'][annotation_nr]['text']:
            fig_c2_d2_h2_estimate['layout']['annotations'][annotation_nr]['font']['size'] = vars.fig_d2_suptitle_font_size

    # Update general layout
    fig_c2_d2_h2_estimate.update_layout(
        title={  # Master title
            # 'text': "Correlation sums, dimension, and entropy estimate<br><sup>(Theiler window: {}; scaling region defined as slope < {}, residuals < {}, and $R^2$ > {})</sup>".format(int(theiler_window), max_slope, max_residuals, min_rsquared),
            # 'text': "Correlation sums, dimension, and entropy estimate (Theiler window: {}; scaling region defined as slope < {}, residuals < {}, and $R^2$ > {}) ".format(int(theiler_window), max_slope, max_residuals, min_rsquared),
            # 'text': r"Correlation sums, dimension, and entropy estimate<br><sup>(Theiler window: {}; scaling region defined as slope < {}, residuals < {}, and {} > {})</sup>".format(
            #     int(theiler_window), max_slope, max_residuals, r"$R^2$", min_rsquared),
            'text': r"Correlation sums, dimension, and entropy estimate<br><sup>(Theiler window: {}; scaling region defined as slope < {}, residuals < {}, and R<sup>2</sup> > {})".format(int(theiler_window), max_slope, max_residuals, min_rsquared),
            'font': dict(size=vars.fig_stationary_title_font_size+4),
            'pad': vars.margins_fig_stationary_title,  # Add padding between master title and plot
            'x': 0.5,  # Centre title
            'xanchor': 'center',
            'y': .95,  # Move title up a bit
            'yanchor': 'middle',
            # 'yref': 'paper'
        },
        height=vars.fig_d2_height,
        width=vars.fig_d2_width,
        showlegend=True,
        # legend_title="Percentage of neighbours covered",
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
        margin=vars.margins_fig_d2,  # Adjust margins figure
    )

    fig_c2_d2_h2_estimate.show()

    ## Save as .png file
    fig_c2_d2_h2_estimate.write_image(filepath_c2_d2_h2_estimate_fig)

    # ## Plot scaling region and d2 estimate
    # fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(16,8), sharex=True, sharey=True)
    #
    # # Plot horizontal line for d2 estimate
    # ax2.axhline(y=d2_estimate, linestyle="dotted")
    # ax2.set_title(r'Correlation dimension $D_2^m(\epsilon)\approx$ {:.2f} (scaling region: [{:.2f}, {:.2f}])'.format(d2_estimate, eps_scaling_region_min, eps_scaling_region_max), fontsize=vars.fig_d2_suptitle_font_size)
    #
    # for emb_nr in np.unique(d2_df_select["emb_nr"]):
    #     # Select only the current embedding dimension
    #     # d2_df_emb = d2_df[d2_df.emb_nr == emb_nr]
    #     d2_df_emb = d2_df_select[d2_df.emb_nr == emb_nr].dropna(subset=["d2", "epsilon"]).sort_values(by="epsilon")
    #     d2_df_emb = d2_df_emb.reset_index(drop=True)  # Reset index without adding new column
    #
    #     # Find epsilon value closest to overall scaling region
    #     eps_scaling_region_min_emb = min(d2_df_emb["epsilon"], key=lambda x: abs(x - eps_scaling_region_min))
    #     eps_scaling_region_max_emb = min(d2_df_emb["epsilon"], key=lambda x: abs(x - eps_scaling_region_max))
    #     eps_scaling_region_min_emb_idx = np.where(d2_df_emb["epsilon"] == eps_scaling_region_min_emb)[0]
    #     eps_scaling_region_max_emb_idx = np.where(d2_df_emb["epsilon"] == eps_scaling_region_max_emb)[0]
    #
    #
    #
    #     d2_df_scaling_emb = d2_df_scaling[d2_df_scaling.emb_nr == emb_nr]
    #
    #     # Correlation sums
    #     ax1.loglog(d2_df_emb["epsilon"], d2_df_emb["c2"], color = vars.fig_d2_color_outside_scaling_region, label = "{}".format(emb_nr))
    #     ax1.loglog(d2_df_scaling_emb["epsilon"], d2_df_scaling_emb["c2"], color = vars.fig_d2_color_inside_scaling_region, label = "Scaling {}".format(emb_nr))
    #
    #     # Correlation dimension
    #     ax2.semilogx(d2_df_emb["epsilon"], d2_df_emb["d2"], color = vars.fig_d2_color_outside_scaling_region, label="{}".format(emb_nr))
    #     ax2.semilogx(d2_df_emb["epsilon"][eps_scaling_region_min_emb_idx:eps_scaling_region_max_emb_idx], d2_df_emb["d2"][eps_scaling_region_min_emb_idx:eps_scaling_region_max_emb_idx], color = vars.fig_d2_color_inside_scaling_region, label="Scaling {}".format(emb_nr))
    #
    #     # Correlation entropy
    #     ax3.semilogx(d2_df_emb["epsilon"], (1/tau) * d2_df_emb["h2"], color = vars.fig_d2_color_outside_scaling_region, label="{}".format(emb_nr))
    #     ax3.semilogx(d2_df_scaling_emb["epsilon"], (1/tau) * d2_df_scaling_emb["h2"], color = vars.fig_d2_color_inside_scaling_region, label="Scaling {}".format(emb_nr))
    #     # (1/tau) -> "Do not forget to divide the h2-estimate by the time lag" (https://www.pks.mpg.de/tisean//Tisean_3.0.1/docs/tutorial/ex4.html)
    #     # Check whether this is necessary still!
    #     #  Question: same scaling region for correlation entropy?
    #
    # ax1.set_xlabel(r"$\epsilon$", fontsize=vars.fig_d2_axis_title_font_size)
    # ax1.set_ylabel(r"$C_2^m(\epsilon)$", fontsize=vars.fig_d2_axis_title_font_size)
    # ax1.set_title(r'Correlation sums $C_2^m(\epsilon)$', fontsize=vars.fig_d2_suptitle_font_size)
    #
    # # Correlation dimension
    # ax2.set_xlabel(r"$\epsilon$", fontsize=vars.fig_d2_axis_title_font_size)
    # ax2.set_ylabel(r"$D_2^m(\epsilon)$", fontsize=vars.fig_d2_axis_title_font_size)
    # ax2.set_title(r'Correlation dimension $D_2^m(\epsilon)$', fontsize=vars.fig_d2_suptitle_font_size)
    #
    # # Correlation entropy
    # ax3.set_xlabel(r"$\epsilon$", fontsize=vars.fig_d2_axis_title_font_size)
    # ax3.set_ylabel(r"$h_2^m(\epsilon)$", fontsize=vars.fig_d2_axis_title_font_size)
    # ax3.set_title(r'Correlation entropy $h_2^m(\epsilon)$', fontsize=vars.fig_d2_suptitle_font_size)
    # # fig.suptitle(r"Scaling regions in $D_2^m(\epsilon)$ of different embedding dimensions<br><sup>Average scaling region [{:.2f}, {:.2f}]</sup>".format(eps_scaling_region_min, eps_scaling_region_max), fontsize=vars.fig_d2_suptitle_font_size)
    #
    # # Adjust spacing between subplots
    # plt.subplots_adjust(left=.125, # the left side of the subplots of the figure
    #                     bottom=.1,  # the bottom of the subplots of the figure
    #                     right=.9, # the right side of the subplots of the figure
    #                     top=.9,  # the top of the subplots of the figure
    #                     wspace=.3, # the amount of width reserved for blank space between subplots
    #                     hspace=.2 # the amount of height reserved for white space between subplots
    #                     )
    #
    # # plt.legend()
    # plt.show()
    #
    # # Save plot
    # plt.savefig(filepath_c2_d2_h2_estimate_fig)


    return filepath_c2_d2_h2_estimate_fig

def d2_or_h2_regression_plot(dict_ent, layout_der, pattern_derivatives_output, Schaefer_ROIs_df, nr_PCs, mask_unstand_or_stand, raw_or_PC_unstand_or_stand, d2_or_h2 = "d2", theiler_window=0, tau = 1, emb_dim=1, max_slope=1, max_residuals=.1, min_rsquared=.4
                   # scaling_region=[10**(-.8), 10**0],
              ):
    # (1/tau) -> "Do not forget to divide the h2-estimate by the time lag" (https://www.pks.mpg.de/tisean//Tisean_3.0.1/docs/tutorial/ex4.html)
    # Check whether this is necessary still!

    # Build input file path
    dict_ent['pipeline'] = 'timeseries_analysis'
    dict_ent['extension'] = '.csv'
    dict_ent['suffix'] = 'PCs'
    dict_ent['timeseries_or_figs'] = 'timeseries'
    dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
    dict_ent['raw_or_PC_unstand_or_stand'] = raw_or_PC_unstand_or_stand

    # Define file path for figure output d2 estimate using regression (showing how we found the scaling region)
    filepath_d2_or_h2_regression_fig = pathlib.Path(
        layout_der.build_path({**dict_ent,
                                'timeseries_or_figs': 'figs',
                               'type_of_fig': 'd2',
                               'extension':'.png',
                               'suffix': "{}_{}-{}_theiler_{}".format(dict_ent['suffix'], d2_or_h2, 'regression-per-emb-dim', theiler_window)},
                              pattern_derivatives_output, validate=False,
                              absolute_paths=True))

    # Define file path for output d2 estimate
    filepath_d2_h2_estimate = pathlib.Path(
        layout_der.build_path({**dict_ent,
                               'extension': '.json',
                               'suffix': "{}_{}_theiler_{}".format(dict_ent['suffix'], 'd2-h2-estimate', theiler_window)},
                              pattern_derivatives_output, validate=False,
                              absolute_paths=True))

    # Now check for theiler_window, tau, and emb_dim
    nr_ROIs = Schaefer_ROIs_df.shape[0]

    # Read d2 estimate and regression data
    with open(filepath_d2_h2_estimate, 'r') as file:
        d2_h2_estimate_dict = json.load(file)  # Read .json file

    estimate = d2_h2_estimate_dict["global_{}_estimate".format(d2_or_h2)]
    eps_scaling_region_min, eps_scaling_region_max = d2_h2_estimate_dict["global_{}_eps_scaling_region".format(d2_or_h2)]
    df_select = convert_list_to_df(d2_h2_estimate_dict["{}_df_select".format(d2_or_h2)])
    eps_scaling_region_dfs = convert_list_to_df(d2_h2_estimate_dict["{}_eps_scaling_region_dfs".format(d2_or_h2)])
    model_dfs = convert_list_to_df(d2_h2_estimate_dict["{}_model_dfs".format(d2_or_h2)])
    model_scaling_dfs = convert_list_to_df(d2_h2_estimate_dict["{}_model_scaling_dfs".format(d2_or_h2)])
    plateau_dfs = convert_list_to_df(d2_h2_estimate_dict["{}_plateau_dfs".format(d2_or_h2)])

    ## Plot how we found the scaling region: for each embedding dimension, show scaling region and slope of $D_2$
    nr_rows=4
    max_emb_dim = np.max(np.unique(df_select["emb_nr"]))+1
    nr_cols=int(max_emb_dim/nr_rows)
    # If we cannot cover all dimensions with the current number of subplots, add one column
    while (nr_rows * nr_cols < max_emb_dim):
        nr_cols+= 1

    fig, axs = plt.subplots(ncols=nr_cols, nrows=nr_rows, figsize=(22,10), sharex=True, sharey=True)
    plt.setp(axs, ylim=(-5,12.5)) # Setting the same y-axis limit for all axes.
    axs = axs.ravel() # Unravel axes for better indexing

    for emb_nr in np.unique(df_select["emb_nr"]):

        # Get values belonging to current embedding dimension
        df_emb = df_select[df_select.emb_nr == emb_nr].dropna(subset=[d2_or_h2, "epsilon"]).sort_values(by="epsilon")
        df_emb = df_emb.reset_index(drop=True)  # Reset index without adding new column
        # plateau_dfs_emb = plateau_dfs[plateau_dfs.emb_nr == emb_nr]
        model_dfs_emb = model_dfs[model_dfs.emb_nr == emb_nr]

        # Find scaling region in dataframe
        eps_scaling_region_dfs_emb = eps_scaling_region_dfs[eps_scaling_region_dfs.emb_nr == emb_nr]
        min_eps_scaling_region_idx = np.where(df_emb["epsilon"].values == eps_scaling_region_dfs_emb["min_eps_scaling_region"].values[0])[0][0]
        max_eps_scaling_region_idx = np.where(df_emb["epsilon"].values == eps_scaling_region_dfs_emb["max_eps_scaling_region"].values[0])[0][0]
        df_scaling_emb = df_emb[min_eps_scaling_region_idx:max_eps_scaling_region_idx+1]

        # Plot complete d2
        outside_line = axs[int(emb_nr)].semilogx(df_emb["epsilon"], df_emb[d2_or_h2],
                     color=vars.fig_d2_color_outside_scaling_region,
                                  label="Outside scaling region")

        # Plot scaling region
        # Note that the plateaus look smoother because it is using the rolling regression with a sliding window
        # axs[int(emb_nr)].semilogx(plateau_dfs_emb["epsilon"], plateau_dfs_emb["d2"],
        #              color=vars.fig_color_inside_scaling_region)
        inside_line = axs[int(emb_nr)].semilogx(df_scaling_emb["epsilon"], df_scaling_emb[d2_or_h2],
                     color=vars.fig_d2_color_inside_scaling_region,
                                  label="Inside scaling region")

        # Plot regression line
        regression_line = axs[int(emb_nr)].scatter(model_dfs_emb["epsilon"], model_dfs_emb["slope"], color="blue", s=5, label="Regression slope")
        axs[int(emb_nr)].set_title('Emb dimension {}; scaling: [{:.2f},{:.2f}]'.format(int(emb_nr), eps_scaling_region_dfs_emb["min_eps_scaling_region"].values[0], eps_scaling_region_dfs_emb["max_eps_scaling_region"].values[0]), fontsize=vars.fig_d2_suptitle_font_size-6)

    # Add common axis titles
    fig.suptitle("Scaling regions in ${}_2^m(\epsilon)$ of different embedding dimensions for Theiler window = {}\nAverage scaling region [{:.2f}, {:.2f}] (scaling region defined as slope < {}, residuals < {}, and $R^2$ > {})".format(d2_or_h2[0].upper(), int(theiler_window), eps_scaling_region_min, eps_scaling_region_max, max_slope, max_residuals, min_rsquared), fontsize=vars.fig_d2_suptitle_font_size)
    fig.text(0.5, 0.04, r'$\log{\epsilon}$', ha='center', fontsize=vars.fig_d2_axis_title_font_size)
    fig.text(0.04, 0.5, r'${}_2^m(\epsilon)$'.format(d2_or_h2[0].upper()), va='center', rotation='vertical', fontsize=vars.fig_d2_axis_title_font_size)

    # Add shared legend
    fig.legend([outside_line, inside_line, regression_line],  # The line objects
               labels=["Outside scaling region", "Inside scaling region", "Regression of slope"],  # The labels for each line
               loc="upper right",  # Position of legend
               borderaxespad=0.1,  # Small spacing around legend box
               title=""  # Title for the legend
               )

    plt.show()

    # Save plot
    plt.savefig(filepath_d2_or_h2_regression_fig)

    return filepath_d2_or_h2_regression_fig



# def d2_regression_plot(dict_ent, layout_der, pattern_derivatives_output, Schaefer_ROIs_df, nr_PCs, mask_unstand_or_stand, raw_or_PC_unstand_or_stand,theiler_window=0, tau = 1
#                    # scaling_region=[10**(-.8), 10**0],
#               ):
#     # (1/tau) -> "Do not forget to divide the h2-estimate by the time lag" (https://www.pks.mpg.de/tisean//Tisean_3.0.1/docs/tutorial/ex4.html)
#     # Check whether this is necessary still!
#     #  Question: same scaling region for correlation entropy?
#
#     # Build input file path
#     dict_ent['pipeline'] = 'timeseries_analysis'
#     dict_ent['extension'] = '.csv'
#     dict_ent['suffix'] = 'PCs'
#     dict_ent['timeseries_or_figs'] = 'timeseries'
#     dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
#     dict_ent['raw_or_PC_unstand_or_stand'] = raw_or_PC_unstand_or_stand
#
#     # Define file path for figure output d2 estimate using regression (showing how we found the scaling region)
#     filepath_d2_regression_fig = pathlib.Path(
#         layout_der.build_path({**dict_ent,
#                                 'timeseries_or_figs': 'figs',
#                                'type_of_fig': 'd2',
#                                'extension':'.png',
#                                'suffix': "{}_{}".format(dict_ent['suffix'], 'd2-regression-per-emb-dim')},
#                               pattern_derivatives_output, validate=False,
#                               absolute_paths=True))
#
#     # Define file path for output d2 estimate
#     filepath_d2_estimate = pathlib.Path(
#         layout_der.build_path({**dict_ent,
#                                'extension': '.json',
#                                'suffix': "{}_{}".format(dict_ent['suffix'], 'd2-estimate')},
#                               pattern_derivatives_output, validate=False,
#                               absolute_paths=True))
#
#     # Read d2 estimate and regression data
#     with open(filepath_d2_estimate, 'r') as file:
#         d2_h2_estimate_dict = json.load(file)  # Read .json file
#
#     d2_estimate = d2_h2_estimate_dict["global_d2_estimate"]
#     eps_scaling_region_min, eps_scaling_region_max = d2_h2_estimate_dict["global_d2_eps_scaling_region"]
#     d2_df_select = convert_list_to_df(d2_h2_estimate_dict["d2_df_select"])
#     eps_scaling_region_dfs = convert_list_to_df(d2_h2_estimate_dict["d2_eps_scaling_region_dfs"])
#     model_dfs = convert_list_to_df(d2_h2_estimate_dict["d2_model_dfs"])
#     model_scaling_dfs = convert_list_to_df(d2_h2_estimate_dict["d2_model_scaling_dfs"])
#     plateau_dfs = convert_list_to_df(d2_h2_estimate_dict["d2_plateau_dfs"])
#
#     ## Plot how we found the scaling region: for each embedding dimension, show scaling region and slope of $D_2$
#     nr_rows=4
#     max_emb_dim = np.max(np.unique(d2_df_select["emb_nr"]))
#     nr_cols=int(max_emb_dim/nr_rows)
#     fig, axs = plt.subplots(ncols=nr_cols, nrows=nr_rows, figsize=(22,10), sharex=True, sharey=True)
#     plt.setp(axs, ylim=(-5,12.5)) # Setting the same y-axis limit for all axes.
#     axs = axs.ravel() # Unravel axes for better indexing
#
#     for emb_nr in np.unique(d2_df_select["emb_nr"]):
#
#         # Get values belonging to current embedding dimension
#         d2_df_emb = d2_df_select[d2_df_select.emb_nr == emb_nr].dropna(subset=["d2", "epsilon"]).sort_values(by="epsilon")
#         d2_df_emb = d2_df_emb.reset_index(drop=True)  # Reset index without adding new column
#         # plateau_dfs_emb = plateau_dfs[plateau_dfs.emb_nr == emb_nr]
#         d2_model_dfs_emb = d2_model_dfs[d2_model_dfs.emb_nr == emb_nr]
#
#         # Find scaling region in dataframe
#         d2_eps_scaling_region_dfs_emb = d2_eps_scaling_region_dfs[d2_eps_scaling_region_dfs.emb_nr == emb_nr]
#         d2_min_eps_scaling_region_idx = np.where(d2_df_emb["epsilon"].values == d2_eps_scaling_region_dfs_emb["min_eps_scaling_region"].values[0])[0][0]
#         d2_max_eps_scaling_region_idx = np.where(d2_df_emb["epsilon"].values == d2_eps_scaling_region_dfs_emb["max_eps_scaling_region"].values[0])[0][0]
#         d2_df_scaling_emb = d2_df_emb[d2_min_eps_scaling_region_idx:d2_max_eps_scaling_region_idx+1]
#
#         # Plot complete d2
#         axs[int(emb_nr)].semilogx(d2_df_emb["epsilon"], d2_df_emb["d2"],
#                      color=vars.fig_d2_color_outside_scaling_region)
#
#         # Plot scaling region
#         # Note that the plateaus look smoother because it is using the rolling regression with a sliding window
#         # axs[int(emb_nr)].semilogx(plateau_dfs_emb["epsilon"], plateau_dfs_emb["d2"],
#         #              color=vars.fig_d2_color_inside_scaling_region)
#         axs[int(emb_nr)].semilogx(d2_df_scaling_emb["epsilon"], d2_df_scaling_emb["d2"],
#                      color=vars.fig_d2_color_inside_scaling_region)
#
#         # Plot regression line
#         axs[int(emb_nr)].scatter(d2_model_dfs_emb["epsilon"], d2_model_dfs_emb["slope"], color="blue", s=5)
#         axs[int(emb_nr)].set_title('Emb dimension {}; scaling: [{:.2f},{:.2f}]'.format(int(emb_nr),
#                                                                                        d2_eps_scaling_region_dfs_emb["min_eps_scaling_region"].values[0],
#                                                                                        d2_eps_scaling_region_dfs_emb["max_eps_scaling_region"].values[0]), fontsize=vars.fig_d2_suptitle_font_size-6)
#
#     # Add common axis titles
#     fig.suptitle("Scaling regions in $D_2^m(\epsilon)$ of different embedding dimensions \nAverage scaling region [{:.2f}, {:.2f}]".format(d2_eps_scaling_region_min, d2_eps_scaling_region_max), fontsize=vars.fig_d2_suptitle_font_size)
#     fig.text(0.5, 0.04, r'$\log{\epsilon}$', ha='center', fontsize=vars.fig_d2_axis_title_font_size)
#     fig.text(0.04, 0.5, r'$D_2^m(\epsilon)$', va='center', rotation='vertical', fontsize=vars.fig_d2_axis_title_font_size)
#     plt.show()
#
#     # Save plot
#     plt.savefig(filepath_d2_regression_fig)
#
#     return filepath_d2_regression_fig


def space_time_traj_plot(dict_ent, layout_der, pattern_derivatives_output,
                mask_unstand_or_stand,
                raw_or_PC_unstand_or_stand,
                nr_timepoints
                ):

    # Build file path to data
    dict_ent['pipeline'] = 'timeseries_analysis'
    dict_ent['extension'] = '.csv'
    dict_ent['suffix'] = 'PCs'
    dict_ent['timeseries_or_figs'] = 'timeseries'
    dict_ent['mask_unstand_or_stand'] = mask_unstand_or_stand
    dict_ent['raw_or_PC_unstand_or_stand'] = raw_or_PC_unstand_or_stand

    # Prepare output file path
    filepath_spaceTimeTraj_output = pathlib.Path(
        layout_der.build_path({**dict_ent,
                               'suffix': "{}_{}".format(dict_ent['suffix'], 'space-time-traj'),
                               },
        pattern_derivatives_output, validate=False,
                              absolute_paths=True)).as_posix()

    # Build path to theiler window estimate
    filepath_theiler_estimate = pathlib.Path(
        layout_der.build_path({**dict_ent,
                               'suffix': "{}_{}_{}".format(dict_ent['suffix'], 'space-time-traj', 'theiler-estimate'),
                               'extension': '.json'}, pattern_derivatives_output, validate=False,
                              absolute_paths=True)).as_posix()

    filepath_spaceTimeTraj_fig = pathlib.Path(
        layout_der.build_path({**dict_ent,
                               'timeseries_or_figs': 'figs',
                               'type_of_fig': 'space-time-traj',
                               'extension': '.png',
                               'suffix': "{}_{}".format(dict_ent['suffix'], 'space-time-traj'),
                               },
                              pattern_derivatives_output, validate=False,
                              absolute_paths=True)).as_posix()

    # Read data
    data_spacetime = pd.read_csv(filepath_spaceTimeTraj_output, index_col=None)  # Read dataframe
    # Get column names
    cols = list(data_spacetime.columns)
    cols.remove("Delta_t") # Remove time index column

    # Get estimate of theiler window
    # Read .json file
    with open(filepath_theiler_estimate, 'r') as file:
        data_theiler_est = json.load(file)  # Read .json file

    theiler_window = data_theiler_est["theiler"]
    first_peak_idx_list = data_theiler_est["first_peak_idx"]
    first_peak_idx = convert_list_to_df(first_peak_idx_list) # Convert list to dataframe

    # # Plot first peaks; Get estimate first peak for each percentage lne
    # first_peak_idx = timeseries_analysis.estimate_theiler_window(dict_ent, layout_der, pattern_derivatives_output,
    #             mask_unstand_or_stand,
    #             raw_or_PC_unstand_or_stand,
    #             nr_timepoints,
    #             method, # Fixed recurrence rate
    #             thresh)
    #
    # theiler_window = np.max(first_peak_idx["idx"].values) + 1


    # Plot
    fig_spacetime = make_subplots(
            rows=1, cols=1,
            # subplot_titles=subplot_titles,
            # shared_xaxes=True,
            # x_title=x_title,
            # y_title=y_title,
            # vertical_spacing=.175,  # Vertical spacing indicates the distance between the rows of the subplots
            # horizontal_spacing=.1,  # Horizontal spacing indicates the distance between the columns of the subplots
            # row_heights=[.2, .2, .2, .2],
        )

    for i in range(len(cols)):
        fig_spacetime.add_trace(
            go.Scatter(
                x=data_spacetime["Delta_t"],
                y=data_spacetime[cols[i]].values,
                line=dict(color=vars.space_time_trajs[i],
                          width=vars.line_width_timeseries_mean),
                name=int(float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)",cols[i])[0])*100), # Get percentage
                # line=dict(color=tint_or_shade_rgb(rgb_str=hex_to_rgb(vars.ROI_colours[label_nr]),
                #                                   factor=vars.fig_PC_shade_tint_factors[PC_nr - 1],
                #                                   shade_or_tint=vars.fig_PC_shade_or_tint[PC_nr - 1]),
                #           width=vars.fig_PC_lwd[1]),
                # showlegend=False,
                # legendgroup="{}".format(mask_label),
                # legendgrouptitle={
                #     # "text": annos["all_PCs"]["text"],
                #     "font": {"size": vars.fig_PC_label_font_size + 1}
                # },
                mode='lines'
            ), row=1, col=1)

    # Plot first peaks of each percentage line
    fig_spacetime.add_trace(go.Scatter(
        x=data_spacetime["Delta_t"][first_peak_idx["idx"].values],
        y=first_peak_idx["value"],
        marker=dict(color="orange"),
        mode="markers",
        showlegend=False
    ),
        row=1, col=1)

    # Adjust axis labels and ticks
    fig_spacetime['layout']['xaxis']['title'] = {'text': r'$\Delta_t$', 'font': {
        'size': vars.fig_stationary_subplot_axis_title_font_size}}  # Increase font size axis title
    fig_spacetime['layout']['xaxis'][
        'nticks'] = vars.fig_stationary_subplot_nr_ticks
    fig_spacetime['layout']['xaxis']['tickfont'] = {
        'size': vars.fig_stationary_subplot_tick_font_size}
    fig_spacetime['layout']['yaxis']['title'] = {'text': r'$\epsilon$', 'font': {
        'size': vars.fig_stationary_subplot_axis_title_font_size}}  # Increase font size axis title
    fig_spacetime['layout']['yaxis']['nticks'] = vars.fig_stationary_subplot_nr_ticks
    fig_spacetime['layout']['yaxis']['tickfont'] = {
        'size': vars.fig_stationary_subplot_tick_font_size}

    fig_spacetime.update_xaxes(showgrid=True, gridwidth=vars.grid_width,
                                gridcolor=vars.grid_color,
                                zeroline=True, zerolinewidth=vars.grid_width,
                                zerolinecolor=vars.grid_color)
    fig_spacetime.update_yaxes(showgrid=True, gridwidth=vars.grid_width,
                                gridcolor=vars.grid_color,
                                zeroline=True, zerolinewidth=vars.grid_width,
                                zerolinecolor=vars.grid_color)
    # Update general layout
    fig_spacetime.update_layout(
        title={  # Master title
            'text': "Space-time trajectory<br><sup>Orange dots mark first peaks; estimated Theiler window: {}</sup>".format(int(theiler_window)),
            'font': dict(size=vars.fig_stationary_title_font_size),
            'pad': vars.margins_fig_stationary_title,  # Add padding between master title and plot
            'x': 0.5,  # Centre title
            'xanchor': 'center',
            'y': .95,  # Move title up a bit
            'yanchor': 'middle',
            # 'yref': 'paper'
        },
        height=int(vars.fig_stationary_height/2),
        width=int(vars.fig_stationary_width/2),
        showlegend=True,
        legend_title="Percentage of neighbours covered",
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

    # fig_spacetime.update_layout(showlegend=False)
    fig_spacetime.show()

    ## Save as .png file
    fig_spacetime.write_image(filepath_spaceTimeTraj_fig)

    return filepath_spaceTimeTraj_fig
