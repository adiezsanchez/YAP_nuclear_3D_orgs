from czifile import CziFile
from xml.etree import ElementTree as ET


def extract_scaling_metadata(file_path):

    with CziFile(file_path) as czi:
        # Extract the metadata XML from the CZI file
        metadata_xml = czi.metadata()

    # Parse the XML metadata
    root = ET.fromstring(metadata_xml)

    # Extract scaling information
    scaling_x = root.find('.//Scaling/Items/Distance[@Id="X"]/Value').text
    scaling_y = root.find('.//Scaling/Items/Distance[@Id="Y"]/Value').text
    scaling_z = root.find('.//Scaling/Items/Distance[@Id="Z"]/Value').text

    # Convert the string values to floats and then to micrometers for readability
    scaling_x_um = round((float(scaling_x) * 1e6), 2)
    scaling_y_um = round((float(scaling_y) * 1e6), 2)
    scaling_z_um = round((float(scaling_z) * 1e6), 2)

    return scaling_x_um, scaling_y_um, scaling_z_um