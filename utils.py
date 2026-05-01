from czifile import CziFile
import numpy as np
import pyclesperanto_prototype as cle
from skimage.segmentation import relabel_sequential
from xml.etree import ElementTree as ET
from cellpose import models, core, io

io.logger_setup()  # run this to get printing of progress

_CELLPOSE_MODEL = None

def extract_scaling_metadata(file_path: str) -> tuple[float, float, float]:
    """
    Extract physical pixel scaling information from a CZI file's metadata.

    Args:
        file_path (str): Path to the CZI file.

    Returns:
        tuple[float, float, float]: Physical pixel sizes (in micrometers) along x, y, and z axes,
            as (scaling_x_um, scaling_y_um, scaling_z_um).
    """
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

def _get_cellpose_model(require_gpu: bool = True):
    """
    Lazily initialize and cache the Cellpose model.

    This keeps module import lightweight and allows workflows that only load
    precomputed results to run on CPU-only environments.
    """
    global _CELLPOSE_MODEL

    if _CELLPOSE_MODEL is not None:
        return _CELLPOSE_MODEL

    has_gpu = core.use_gpu()
    if require_gpu and not has_gpu:
        raise RuntimeError(
            "Cellpose nuclei prediction requires GPU, but no GPU was detected. "
            "You can still run workflows that load precomputed nuclei labels."
        )

    _CELLPOSE_MODEL = models.CellposeModel(gpu=has_gpu)
    return _CELLPOSE_MODEL

def _resolve_napari_viewer(viewer):
    """
    Return a napari Viewer for optional visualization.

    If ``viewer`` is passed, it is used. Otherwise tries ``napari.current_viewer()``;
    if none exists, creates ``napari.Viewer()``. Import is deferred until visualization runs.
    """
    if viewer is not None:
        return viewer
    import napari

    v = napari.current_viewer()
    if v is not None:
        return v
    return napari.Viewer()

def _keep_objects_in_size_range(labels: np.ndarray, min_max_size: tuple[int, int]) -> np.ndarray:
    """
    Keep only labeled objects whose voxel count is within a min/max range.

    Args:
        labels (np.ndarray): Labeled image where 0 is background.
        min_max_size (tuple[int, int]): Inclusive size range as (min_size, max_size).

    Returns:
        np.ndarray: Filtered labels, relabeled sequentially from 1..N.
    """
    min_size, max_size = min_max_size
    counts = np.bincount(labels.ravel())
    keep = (counts >= max(min_size, 0)) & (counts <= max_size)
    keep[0] = False  # keep background as 0

    filtered = labels.copy()
    filtered[~keep[labels]] = 0
    filtered, _, _ = relabel_sequential(filtered)
    return filtered

def predict_nuclei_labels(image: np.ndarray, rescale_factor: float, nuclei_channel: int, min_max_nuclei_volume: tuple[int, int] | None = None, visualize=False, viewer=None) -> np.ndarray:
    """
    Predict nuclei labels using CellposeSAM using anisotropy correction.

    Args:
        image (np.ndarray): Image to predict nuclei labels from.
        rescale_factor (float): Rescale factor to apply to the Z-axis for isotropic scaling (z_um / mean([x_um, y_um])).
        nuclei_channel (int): Channel index of the nuclei channel in the image.
        min_max_nuclei_volume (tuple[int, int] | None, optional): Inclusive min/max nuclei
            volume used to filter predicted labels. Set to ``None`` to skip filtering.
            Defaults to ``None``.
        visualize (bool, optional): If True, display the predicted nuclei labels in Napari.
        viewer (optional): Napari ``Viewer`` instance. If ``visualize`` is True and this is omitted,
            the current viewer (if any) is used, otherwise a new ``napari.Viewer()`` is created.

    Returns:
        np.ndarray: Nuclei labels.
    """
    model = _get_cellpose_model(require_gpu=True)

    # Predict nuclei labels
    nuclei_labels, _ , _ = model.eval(image[nuclei_channel], do_3D=True, anisotropy=rescale_factor, z_axis=0, niter=1000)
    # Filter nuclei labels to keep only those within the specified size range.
    if min_max_nuclei_volume is not None:
        nuclei_labels = _keep_objects_in_size_range(nuclei_labels, min_max_nuclei_volume)

    # Display the resulting nuclei labels in Napari if requested.
    if visualize:
        v = _resolve_napari_viewer(viewer)
        v.add_labels(nuclei_labels)

    return nuclei_labels

def simulate_cytoplasm(
    nuclei_labels: np.ndarray,
    cytoplasm_thickness: int = 2,
    nuclei_padding: int = 0
) -> np.ndarray:
    """
    Generates approximate cytoplasmic regions surrounding each nucleus by dilating nuclear label masks.
    Optionally, an initial padding can be applied to the nuclei before cytoplasm simulation. 
    The 'cytoplasm_thickness' parameter defines how many pixels/voxels outward from the nucleus (or optionally padded nucleus) 
    to assign cytoplasmic regions. The resulting cytoplasm region for each nucleus 
    is assigned the same label, excluding the nucleus itself. Background remains 0.

    Args:
        nuclei_labels (np.ndarray): Integer-labeled nuclei mask (0 = background, >0 = nucleus).
        cytoplasm_thickness (int, optional): Number of pixels/voxels outward from the (optionally padded) 
            nucleus used to define the cytoplasmic region. Defaults to 2.
        nuclei_padding (int, optional): Number of pixels/voxels to optionally expand nuclei before cytoplasm simulation. 
            Useful to create a buffer zone between nuclei and cytoplasm. Defaults to 0 (no padding).

    Returns:
        np.ndarray: Integer-labeled cytoplasmic regions, where each region surrounding a nucleus receives the nucleus's label, and the nuclei remain excluded (set to 0).
    """
    labels_for_cytoplasm = nuclei_labels
    if nuclei_padding > 0:
        # Optionally pre-dilate nuclei to create a buffer between nuclei and cytoplasm border
        padded_nuclei_labels = cle.dilate_labels(nuclei_labels, radius=nuclei_padding)
        labels_for_cytoplasm = cle.pull(padded_nuclei_labels)

    # Dilate labels outward by cytoplasm_thickness to generate cytoplasmic regions
    cyto_labels = cle.dilate_labels(labels_for_cytoplasm, radius=cytoplasm_thickness)
    cytoplasm = cle.pull(cyto_labels)

    # Remove (optionally padded) nucleus core regions by setting them to background (zero)
    cytoplasm[labels_for_cytoplasm > 0] = 0

    return cytoplasm