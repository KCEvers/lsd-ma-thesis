# Based on matlab python from https://www.nitrc.org/frs/?group_id=477
from master.globalimports import *


def convert_MNI_to_XYZ(MNI_x,MNI_y,MNI_z,mm = 2,origin=list([45, 63, 36])):
    '''

    :param x: x-coordinate in MNI space;
    :param y: y-coordinate in MNI space;
    :param z: z-coordinate in MNI space;
    :param mm: size of single voxel in mm, default 2mm;
    :param origin: origin coordinate [x,y,z] in matrix indices; defaults to [45 63 36]. Note that the origin refers
    to the anterior commissure and is defined as [0,0,0] in MNI space;
    :return: x,y,z coordinates in XYZ system/matrix indices.
    '''

    x = np.around(origin[0] - MNI_x / mm)
    y = np.around(MNI_y / mm + origin[1])
    z = np.around(MNI_z / mm + origin[2])
    # print("z")
    return np.array([int(x),int(y),int(z)])


def find_atlas_ROI_label_of_coord(idx, atlas_data,atlas_labels,study_coords_xyz):
    """ Function to find the ROI label in a chosen atlas of a specified coordinate.

    :param atlas_data: 3D ndarray with ROI numbers indicating the location of the ROIs.
    :param atlas_labels: Dictionary with ROI numbers as keys and ROI labels as values.
    :param study_coords_xyz: Dataframe containing coordinates (in xyz space) of which the atlas ROI label will be given.
    :param idx: Index to dataframe study_coords_xyz.
    :return:
    """
    # First convert xyz coordinates to MNI space.
    x,y,z = convert_MNI_to_XYZ(study_coords_xyz["MNI x"][idx],study_coords_xyz["MNI y"][idx],study_coords_xyz["MNI z"][idx])
    ROI_nr = int(atlas_data[x,y,z]) # Number of ROI in atlas
    ROI_label = atlas_labels[ROI_nr] # Label of ROI in atlas
    nr_voxels_in_ROI = (atlas_data == ROI_nr).sum() # Number of voxels in ROI
    print("\nIn Schaefer's 200 atlas, the given MNI coordinate ({}: {},{},{})\ncorresponds to region {}: {}, which has\n{} voxels in total".format(study_coords_xyz["Label"][idx],x,y,z,ROI_nr, ROI_label, nr_voxels_in_ROI))
    return [ROI_nr, ROI_label, nr_voxels_in_ROI, study_coords_xyz["MNI x"][idx],study_coords_xyz["MNI y"][idx],study_coords_xyz["MNI z"][idx]]


def rgb_to_hex(rgb_str):
    # First convert rgb string (e.g. 'rgb(86, 177, 163)') to tuple
    rgb_code = tuple([int(x) for x in rgb_str[rgb_str.find("(") + 1: rgb_str.find(")")].split(",")])
    return '#%02x%02x%02x' % rgb_code # Return hex-formatted python

# Following three functions are by Kerry Halupka (https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72)
def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))



def rgb_to_rgba(rgb_str, alpha_val):
    rgb_str = str(rgb_str)# Ensure it is in string format
    # First extract r,g,b values
    r, g, b = [float(x) for x in rgb_str[rgb_str.find("(") + 1: rgb_str.find(")")].split(",")]
    # If the values are all larger than 1, divide by 255 so they range between 0-1
    if np.any(np.array([r, g, b]) > 1):

        r, g, b = r / 255, g / 255, b / 255
    return 'rgba({},{},{},{})'.format(r, g, b, alpha_val)  # Return rgba in string format


def rgb_to_rgba_255(rgb_str, alpha_val):
    """Convert rgb to rgba by adding alpha value and converting rgb to 255 format if necessary (meaning the rgb values range from 0-255).

        Parameters
        ----------
        rgb_str : string
            Specifies rgb-formatted colour.
        alpha_val : float
            Transparency value; float between 0 (transparent) to 1 (opaque).

        Returns
        -------
        string
            Specifies rgba-formatted colour in 255 format (meaning the rgb values range from 0-255).
        """

    # First extract r,g,b values
    r, g, b = [float(x) for x in rgb_str[rgb_str.find("(") + 1: rgb_str.find(")")].split(",")]
    # If the values are all larger than 1, divide by 255 so they range between 0-1
    if np.any((np.array([r, g, b]) < 1) & (np.array([r, g, b]) != 0)):
        r, g, b = r * 255, g * 255, b * 255
    return 'rgba({},{},{},{})'.format(r, g, b, alpha_val)  # Return rgba in string format




def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]


def tint_or_shade_rgb(rgb_str, factor, shade_or_tint):
    """ Lighten or darken a given rgb colour.

    :return:
    """
    # Ensure it is in string format
    rgb_str = str(rgb_str)

    # First extract r,g,b values
    r, g, b = [int(x) for x in rgb_str[rgb_str.find("(") + 1: rgb_str.find(")")].split(",")]

    # If the values are all larger than 1, divide by 255 so they range between 0-1
    if np.all(np.array([r, g, b]) > 1):
        r, g, b = r / 255, g / 255, b / 255

    if shade_or_tint == "shade":
        # # To shade:
        # newR = r * (1 - factor)
        # newG = g * (1 - factor)
        # newB = b * (1 - factor)
        aR, aG, aB = 0, 0, 0

    elif shade_or_tint == "tint":
        # To tint:
        # newR = r + (255 - r) * factor
        # newG = g + (255 - g) * factor
        # newB = b + (255 - b) * factor
        aR, aG, aB = 1, 1, 1

    newR = r + (aR - r) * factor
    newG = g + (aG - g) * factor
    newB = b + (aB - b) * factor

    return 'rgb({},{},{})'.format(newR, newG, newB)  # Return rgb in string format


def get_continuous_cmap(hex_list, float_list=None):
    ''' Creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

        Parameters
        ----------
        hex_list: list of hex python strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = matplotlib.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp
