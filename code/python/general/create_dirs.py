"""
Script to create derivatives directory (including subdirectories of subjects and sessions) based on an example rawdata folder.
"""

from python.master.globalimports import *


def create_derivatives_dir(overwrite=False):
    ## Create derivatives directory; if you want to overwrite, it will not be checked whether it exists already
    if overwrite:
        # First remove directory
        shutil.rmtree(vars.path_to_derivatives)  # Removes all the subdirectories!
        vars.path_to_derivatives.mkdir(exist_ok=True, parents=True)
    elif not overwrite:
        if not vars.path_to_derivatives.exists():
            vars.path_to_derivatives.mkdir(exist_ok=False, parents=False)

    ## Create dataset_description.json file in derivatives folder so it is recognized as a BIDS folder
    json_dict = {"Name": "Carhart-Harris et al. (2016) PLRNN analysis",
                 "BIDSVersion": "1.6.1",
                 "License": "CC0",
                 "DatasetType": "derivative",
                 # "HowToAcknowledge": "Please cite this paper: https://www.pnas.org/content/113/17/4853.short",
                 "Authors": ["Kyra Caitlin Evers"],
                 "DatasetDOI": "10.18112/openneuro.rawdata.v1.0.0"
                 }
    with open(pathlib.Path.joinpath(vars.path_to_derivatives, "dataset_description.json"), 'w',
              encoding='utf-8') as outfile:
        json.dump(json_dict, outfile, sort_keys=True, indent=4, ensure_ascii=True)  # indent makes it easier to read

    ## Create BIDS layouts
    layout_rawdata = bids.BIDSLayout(vars.path_to_datafolder)  # Create pybids layout
    layout_der = bids.BIDSLayout(vars.path_to_derivatives)  # Create pybids layout for derivatives

    ## Find .json files in rawdata layout
    entities_folders = {"datatype": "func",  # Create dictionary specifying which entities you want to extract
                        "task": "rest",
                        "suffix": "bold",
                        "extension": ".json"}

    rawdata_bold_json_files = layout_rawdata.get(
        **entities_folders)  # Extract file names of entities of interest from layout; there should be a .json file for two sessions for each subject
    # len(rawdata_bold_json_files) # Should be number of subjects * number of sessions

    for pipeline in vars.pipelines:

        # Create pipeline directory
        dir_path_pipeline = pathlib.Path(vars.path_to_derivatives, pipeline)

        # Create pipeline directory
        if not dir_path_pipeline.exists():  # No need to create this if it already exists
            dir_path_pipeline.mkdir(exist_ok=False,
                                    parents=True)  # Setting exist_ok=False makes sure pathlib throws a warning if the directory already exists; Setting parents=True makes sure that parent directories are created as necessary for nested directories

        # if pipeline == "preproc-rois":
        # Now for each of these files, extract entities and build a new path to build in the derivatives directory
        for file_nr, filename in enumerate(rawdata_bold_json_files):

            # print("\n\nfile_nr: {}".format(file_nr))
            # print("file name: {}".format(rawdata_bold_json_files[file_nr].filename))
            dict_ent = rawdata_bold_json_files[file_nr].get_entities()  # Dictionary with entities of file
            # print("dict_ent:\n{}".format(dict_ent))

            # Add pipeline
            dict_ent["pipeline"] = pipeline

            ## Create the same rawdata subject and session folders in derivative folder; even if the path leads to nested subdirectories whose parents directories have not been created yet (e.g. "derivatives/sub-001/ses-01" for which "sub-001" does not exist yet), pathlib is able to handle this

            # "func" directory: Create func directory with subdirectories timeseries and figs, which each have subdirectories unstand_timeseries, stand_timeseries, and PCs
            for timeseries_or_figs in ["timeseries", "figs"]:
                # for type_of_timeseries in ["unstand_timeseries", "stand_timeseries", "unstand_PCs", "stand_PCs"]:

                dict_ent['timeseries_or_figs'] = timeseries_or_figs

                for mask_unstand_or_stand in ["mask_unstand", "mask_stand"]:
                    dict_ent["mask_unstand_or_stand"] = mask_unstand_or_stand
                    for raw_or_PC_unstand_or_stand in ["raw", "PC_unstand", "PC_stand"]:
                        dict_ent["raw_or_PC_unstand_or_stand"] = raw_or_PC_unstand_or_stand

                        # Create type of figure subdirectories
                        if timeseries_or_figs == "figs":
                            for type_of_fig in vars.type_of_figs:
                                dict_ent["type_of_fig"] = type_of_fig

                                ## Build path to /func and /anat directories
                                dict_ent["datatype"] = "func"
                                make_dir(layout_der, dict_ent, vars.pattern_derivatives)

                                # Create /anat directory by first copying entities dictionary without session key and change data type
                                # dict_ent_anat = {**{key: val for key, val in dict_ent.items() if key != 'session'},
                                #                  'datatype': 'anat'}
                                dict_ent_anat = {**dict_ent, 'datatype': 'anat', 'session': 'PLCB'}
                                make_dir(layout_der, dict_ent_anat, vars.pattern_derivatives)

                                # # "anat" directory
                                # dir_path_anat = pathlib.Path(
                                #     layout_der.build_path({**dict_ent, 'datatype': 'anat'}, vars.pattern_derivatives_anat, validate=False,
                                #                           absolute_paths=True))
                                #
                                # # Create anat and func directories
                                # if not dir_path_anat.exists():  # No need to create this if it already exists
                                #     dir_path_anat.mkdir(exist_ok=False,
                                #                         parents=True)  # Setting exist_ok=False makes sure pathlib throws a warning if the directory already exists; Setting parents=True makes sure that parent directories are created as necessary for nested directories
                                # # Make anat directory
                                # make_dir(layout_der, {**dict_ent, 'datatype': 'anat'}, vars.pattern_derivatives)
                        elif timeseries_or_figs == "timeseries":

                            ## Build path to /func and /anat directories
                            dict_ent["datatype"] = "func"
                            make_dir(layout_der, dict_ent, vars.pattern_derivatives)

                            # Create /anat directory by first copying entities dictionary without session key and change data type
                            # dict_ent_anat = {**{key: val for key, val in dict_ent.items() if key != 'session'},
                            #                  'datatype': 'anat'}
                            dict_ent_anat = {**dict_ent, 'datatype': 'anat', 'session': 'PLCB'}
                            make_dir(layout_der, dict_ent_anat, vars.pattern_derivatives)

    return
    # # Create subdirectories for figures
    # if timeseries_or_figs == "figs":
    #     for subdir in vars.type_of_figs:
    #         subdir_path = pathlib.Path(
    #             layout_der.build_path({**dict_ent, 'timeseries_or_figs': timeseries_or_figs,
    #                                    'type_of_timeseries': type_of_timeseries, 'type_of_fig': subdir},
    #                                   vars.pattern_derivatives,
    #                                   validate=False,
    #                                   absolute_paths=True))
    #         if not subdir_path.exists():
    #             subdir_path.mkdir(exist_ok=False, parents=True)  # Create directory
    #
    #         print("subdir_path: {}".format(subdir_path))
    # (BIDS manual)
    # "A README file SHOULD be found at the root of the “ sourcedata ” or the “ derivatives ” folder (or both).
    # This file should describe the nature of the raw data or the derived data. In the case of the existence of a
    # “ derivatives ” folder, we RECOMMEND to include details about the software stack and settings used to
    # generate the results. Non-imaging objects to improve reproducibility are encouraged (scripts, setting files,
    # etc.).

    ##### Create README file
    # ## Read README file first
    # filename_README = "README"
    # filepath_README = pathlib.Path.joinpath(vars.path_to_datafolder, datafolder, filename_README)
    # with open(filepath_README, 'r', encoding="utf8") as file:
    #     README_txt = file.read()
    # README_txt

    # 3. We RECOMMEND to include the PDF print-out with the actual sequence parameters generated by the
    # scanner in the “ sourcedata ” folder" (BIDS manual)


def make_dir(layout, dict_ent, pattern):
    # Build path
    dir_path = pathlib.Path(
        layout.build_path(
            {**dict_ent},
            pattern,
            validate=False,
            absolute_paths=True))
    if not dir_path.exists():
        dir_path.mkdir(exist_ok=False, parents=True)  # Create directory
    # print("dir_path: {}".format(dir_path))

    return dir_path


# def restructure_rawdata_dir_v0():
#     # Get layout rawdata directory
#     layout_rawdata = bids.BIDSLayout(vars.path_to_datafolder)  # Create pybids layout
#
#     # Copy rawdata directory - takes about 10 min
#     print("Copying rawdata directory (start: {})".format(datetime.datetime.now()))
#     source_dir = pathlib.Path(vars.path_to_datafolder)
#     destination_dir = source_dir.with_name("rawdata_orig_structure")
#     shutil.copytree(source_dir, destination_dir)
#
#     # In rawdata directory, for each subject:
#     subject_nrs = np.array(layout_rawdata.get_subjects(datatype="func"))  # Extract subject numbers
#     session_names = np.array(layout_rawdata.get_sessions(datatype="func"))  # Extract subject numbers
#
#     for subject_nr in subject_nrs:
#
#         # 1. Create new /func directory under sub-{} directory
#         dict_ent = {}
#         dict_ent["subject"] = subject_nr
#         dict_ent["datatype"] = "func"
#         make_dir(layout_rawdata, dict_ent, vars.pattern_rawdata)
#
#         # Copy contents from original directory sub-{}/ses-{}/func to sub-{}/func of each session
#         for session_name in session_names:
#             dict_ent["session"] = session_name
#
#             # 2. Create session subdirectory in /func directory
#             make_dir(layout_rawdata, dict_ent, vars.pattern_rawdata)
#
#             # Build path to session directory
#             dir_ses_path = pathlib.Path(layout_rawdata.build_path(
#                 {**dict_ent},
#                 vars.pattern_rawdata,
#                 validate=False,
#                 absolute_paths=True))
#             # Original file path
#             original = dir_ses_path.parent.with_name(dir_ses_path.name)  # Remove the /func directory
#
#             # 3. Move files from original directory /func to /ses-{}
#             shutil.move(pathlib.Path(original, "func"), dir_ses_path)
#
#             # 4. Delete original directory
#             pathlib.Path.rmdir(original)
#
#             # shutil.move(original, dir_ses_path)
#     return None

def restructure_rawdata_dir(copy_rawdata_dir=True):
    # Get layout rawdata directory
    layout_rawdata = bids.BIDSLayout(vars.path_to_datafolder)  # Create pybids layout

    # Copy rawdata directory - takes about 10 min
    if copy_rawdata_dir:
        print("Copying rawdata directory (start: {})".format(datetime.datetime.now()))
        source_dir = pathlib.Path(vars.path_to_datafolder)
        destination_dir = source_dir.with_name("rawdata_orig_structure")
        shutil.copytree(source_dir, destination_dir)

    # In rawdata directory, for each subject:
    subject_nrs = np.array(layout_rawdata.get_subjects(datatype="func"))  # Extract subject numbers

    for subject_nr in subject_nrs:

        # Move /anat directory to ses-PLCB
        # Create new /func directory under sub-{} directory
        dict_ent = {}
        dict_ent["subject"] = subject_nr
        dict_ent["datatype"] = "anat"
        dict_ent["session"] = "PLCB"

        # Build path to session directory
        dir_ses_path = pathlib.Path(layout_rawdata.build_path(
            {**dict_ent},
            vars.pattern_rawdata,
            validate=False,
            absolute_paths=True))
        # Original file path
        original = dir_ses_path.parent.with_name(dir_ses_path.name)  # Remove the /func directory

        # Move directory
        if original.exists():
            shutil.move(original, dir_ses_path)

    return None
