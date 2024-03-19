# Aortic Aneurysm Detection

## Introduction
This project is focused on detecting aortic aneurysms based on cross-section diameters from NRRD image files. It provides a set of functionalities to preprocess the images, extract relevant features, and detect potential aneurysms.

## Installation
To run this project, ensure you have Python installed. You can then install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage
The main functionality is provided through the `detect` function in the `aneurysm_detection` module. You can use the provided script `run.py` to detect aortic aneurysms from a single NRRD image or process multiple images within a directory.

```bash
poetry run detect -p path/to/image.seg.nrrd
```

### Optional Arguments:
- `-c, --closing_structure_shape`: Shape of the structure for binary image closing.
- `-cla, --consists_legs_artery`: Indicates whether the image includes arteries located in the legs.
- `-of, --outlier_factor`: Factor for detecting outliers based on the neighborhood average.
- `-ns, --neighborhood_size`: Size of the neighborhood for detecting outliers.
- `-odf, --outlier_difference_factor`: Factor for detecting outliers based on the difference between the current and first neighbors.
- `-cs, --comparison_range`: Distance to the previous and next neighbors used for comparison to determine outlier.
- `-ba, --bound_angles`: Angles for rotation around the x and y axes used to find the boundary between two different parts of the aorta.
- `-adt, --abdominal_and_descending_aorta_threshold`: Threshold for the aortic diameter for the abdominal and descending part of the aorta.
- `-aat, --aortic_arch_and_asceding_aorta_threshold`: Threshold for the aortic diameter for the aortic arch and ascending part of the aorta.
- `-awt, --aneurysm_width_tolerance`: Maximum allowed distance of the boundary from the point that is first or last above the aneurysm threshold.
- `-bs, --bound_step`: Step used to find aneurysm bound cross section.
- `-d, --debug_mode`: Activates debugging mode for plotting each algorithm step.

## Example
```bash
poetry run detect -p data/image.seg.nrrd -cla -d
```

## Author
This project is developed and maintained by Saša Marjanović.
