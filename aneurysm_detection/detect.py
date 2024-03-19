import nrrd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import dijkstra3d
import os
from aneurysm_detection.plot import Plot
from aneurysm_detection.utils import Utils
from scipy.ndimage import zoom, rotate, binary_closing
from skimage.morphology import skeletonize_3d
from skimage.measure import label, regionprops



def detect(
        path, 
        closing_structure_shape = (5,5,1),
        consists_legs_artery=False, 
        outlier_factor = 1.3,
        neighborhood_size = 5,
        outlier_difference_factor = 0.15,
        comparison_range = 2,
        bound_angles=(135,0),
        abdominal_and_descending_aorta_threshold=55, 
        aortic_arch_and_asceding_aorta_threshold=60,
        aneurysm_width_tolerance = 6,
        bound_step = 2,
        debug_mode = False
):

    warnings.filterwarnings("ignore", category=UserWarning)
    image, header = nrrd.read(path)
    space_directions = header['space directions']
    original_shape = image.shape

    folder_path = Utils.get_folder_path(path)

    Plot.plot_voxel_image(
        image, folder_path=folder_path, 
        title='original_image',
        debug_mode=debug_mode
    )

    image = get_scaled_image(image)
    image = binary_closing(image, structure=np.ones(closing_structure_shape)).astype(np.int16)
    
    Plot.plot_voxel_image(image, folder_path=folder_path, title='scaled_image', debug_mode=debug_mode)

    file_path = Utils.get_cross_sections_file_path(path)

    if not os.path.exists(file_path):
        centerline_mask, centerline = get_aortic_centerline(image, consists_legs_artery)
        image[centerline_mask > 0] = np.arange(3, 3 + len(centerline))

        Plot.plot_voxel_image(
            image, centerline_mask, folder_path=folder_path, 
            title='centerline',
            debug_mode=debug_mode
        )

        angles = [0, 45, 90, 135]
        images = get_rotated_images(image, angles)
        cross_sections = find_min_cross_sections(centerline, image, images, debug_mode)
        cross_sections = get_cross_sections_with_physical_length_diameter(
                            cross_sections, 
                            space_directions, 
                            original_shape, 
                            image.shape
                        )
        cross_sections, removed_cross_sections, \
        cross_sections_inds, removed_cross_sections_inds = filter_by_diameter(
                                                    cross_sections, 
                                                    outlier_factor, 
                                                    neighborhood_size,
                                                    outlier_difference_factor,
                                                    comparison_range
                                                )
        Plot.plot_filtering_cross_sections(
            cross_sections, 
            removed_cross_sections, 
            cross_sections_inds, 
            removed_cross_sections_inds,
            folder_path,
            debug_mode=debug_mode
        )
        Utils.save_cross_sections(cross_sections,file_path)
    else:
        cross_sections = Utils.read_cross_sections(file_path)

    if debug_mode:
        Plot.plot_cross_sections(cross_sections, folder_path, debug_mode=debug_mode)   
        Plot.plot_each_slice_plane(cross_sections, image, debug_mode=debug_mode)

    bound_cross_sections, bound, bound_cross_sections_inds = detect_aneurysm(
                                    cross_sections, 
                                    bound_angles,
                                    abdominal_and_descending_aorta_threshold, 
                                    aortic_arch_and_asceding_aorta_threshold, 
                                    aneurysm_width_tolerance,
                                    bound_step
                                )

    Plot.plot_finding_bound_cross_sections(
        cross_sections, 
        bound_cross_sections,
        bound_cross_sections_inds,
        abdominal_and_descending_aorta_threshold, 
        aortic_arch_and_asceding_aorta_threshold, 
        bound, 
        folder_path,
        debug_mode=debug_mode
    )

    Plot.plot_detection(image, bound_cross_sections, folder_path)


def get_scaled_image(image):
    scaling_factors = (128 / image.shape[0], 128 / image.shape[1], 256 / image.shape[2])
    image = zoom(image, scaling_factors, order=0).astype(np.int16)

    return image


def get_rotated_images(image, angles):
    images = {}

    for angle_x in angles:
        for angle_y in angles:
            if (angle_x == 90 and (90,0) in images.keys()) or (angle_y == 90 and (0,90) in images.keys()):
                continue
            rotated_image = rotate(image, angle_x, axes=(1,2), order=0)
            rotated_image = rotate(rotated_image, angle_y, axes=(0,2), order=0)
            key = (angle_x, angle_y)
            images[key] = rotated_image
    
    return images


def get_aortic_centerline(image, consists_legs_artery):

    def get_centerline(centerline_mask):
        centerline_mask = centerline_mask.astype(bool)
        centerline = np.where(centerline_mask > 0)
        centerline = list(zip(centerline[0], centerline[1], centerline[2]))
        return centerline
    
    def get_main_centerline(centerline_mask):
    
        def get_centerline_source(centerline_mask):
            start = 0.2 if consists_legs_artery else 0
            z_min_treshold = round(centerline_mask.shape[2]*start)
            for z in range(z_min_treshold, centerline_mask.shape[2], 5):
                mask_slice = centerline_mask[:,:,z]
                labeled_slice = label(mask_slice)
                regions = regionprops(labeled_slice)
                if(len(regions)==1):
                    x,y = np.where(mask_slice>0)
                    x,y = x[0], y[0]
                    return (x,y,z)
                        
        def get_centerline_target(centerline_mask):
            z_min_treshold = round(centerline_mask.shape[2]*0.6)
            for z in range(z_min_treshold, centerline_mask.shape[2],3):
                mask_slice = centerline_mask[:,:,z]
                labeled_slice = label(mask_slice)
                regions = regionprops(labeled_slice)
                if(len(regions)>1):
                    target_y = centerline_mask.shape[1]
                    xs, ys = np.where(mask_slice>0)
                    for x,y in zip(xs,ys):
                        if y < target_y:
                            target_y = y
                            target_x = x
                    return (target_x,target_y,z)

        source = get_centerline_source(centerline_mask)
        target = get_centerline_target(centerline_mask)
        return dijkstra3d.binary_dijkstra(centerline_mask, source, target, background_color=0)   

    def get_centerline_mask(image, centerline):
        mask = np.zeros_like(image)
        for x,y,z in centerline:
            mask[x, y, z] = 1
        return mask

    centerline_mask = skeletonize_3d(image).astype(bool)
    centerline = get_centerline(centerline_mask)
    centerline = get_main_centerline(centerline_mask)
    centerline_mask = get_centerline_mask(image, centerline)

    return centerline_mask, centerline


def get_cross_sections_with_physical_length_diameter(
        cross_sections, 
        space_directions, 
        original_shape, 
        scaled_shape
    ):
    
    x_direction, y_direction, z_direction = space_directions
    
    x_scale = original_shape[0] / scaled_shape[0]
    y_scale = original_shape[1] / scaled_shape[1]
    # z_scale = original_shape[2] / scaled_shape[2]

    x_length = np.sqrt(x_direction[0]**2 + x_direction[1]**2 + x_direction[2]**2) * x_scale
    y_length = np.sqrt(y_direction[0]**2 + y_direction[1]**2 + y_direction[2]**2) * y_scale
    # z_length = np.sqrt(z_direction[0]**2 + z_direction[1]**2 + z_direction[2]**2) * z_scale

    for cross_section in cross_sections:
        cross_section[1] = cross_section[1] * np.sqrt(x_length**2 + y_length**2) #+ z_length**2)

    return cross_sections


def filter_by_diameter(cross_sections: list, outlier_factor, neighborhood_size, outlier_difference_factor, comparison_range):
    
    def is_outlier(cross_sections, index):
        
        def get_nearby_diameters(cross_sections, index):
            start_index_previous = max(0, index - neighborhood_size)
            end_index_next = min(len(cross_sections), index + neighborhood_size + 1)
            
            previous_diameters = [diameter for _, diameter, _ in cross_sections[start_index_previous:index]]
            next_diameters = [diameter for _, diameter, _ in cross_sections[index + 1:end_index_next]]
            
            return previous_diameters, next_diameters
        
        def is_bigger_difference(current_diameter, previous_diameter, next_diameter, average):
            difference_threshold = average * outlier_difference_factor
            return (current_diameter - previous_diameter > difference_threshold and 
                    current_diameter - next_diameter > difference_threshold
            )

        current_diameter = cross_sections[index][1]

        previous_diameters, next_diameters = get_nearby_diameters(cross_sections, index)
        diameters = previous_diameters + next_diameters
        average = np.average(diameters)
        outlier_threshold = average * outlier_factor
        
        if index > 0 and len(previous_diameters) > comparison_range-1 and len(next_diameters) > comparison_range-1:
            return (current_diameter >= outlier_threshold 
                    and is_bigger_difference(current_diameter, previous_diameters[-comparison_range], 
                                             next_diameters[comparison_range-1],average)
            )
        
        return current_diameter >= outlier_threshold 

    removed_cross_sections = []
    removed_cross_sections_inds = []
    cross_sections_inds = []
    index = 0
    removed = 0
    while index < len(cross_sections):
        cross_section = cross_sections[index]
        if cross_section[1]<20 or is_outlier(cross_sections, index):
            removed_cross_sections.append(cross_sections[index])
            removed_cross_sections_inds.append(index+removed)
            cross_sections.pop(index)
            removed += 1
        else:
            cross_sections_inds.append(index+removed)
            index += 1

    return cross_sections, removed_cross_sections, cross_sections_inds, removed_cross_sections_inds


def find_min_cross_sections(centerline, image, images, debug_mode):

    def find_min_cross_section(center_value, images):
        result = None
        min_diameter = np.finfo(np.float16).max
        angles = images.keys()
        data = {}
        for angle in angles:
            img = images[angle]
            centers = np.where(img == center_value)
            if(len(centers[0])>0):
                x,y,z = centers[0][0], centers[1][0], centers[2][0]
                img_slice = img[:,:,z]
                ct_slice = (img_slice > 0).astype(int)
                labeled_slice = label(ct_slice)
                regions = regionprops(labeled_slice)
                label_of_center = labeled_slice[x,y]
                if len(regions) > 0:
                    min_rotation_diameter = np.finfo(np.float16).max
                    for region in regions: 
                        if label_of_center == region.label and region.axis_major_length < min_diameter:
                            min_diameter = region.axis_major_length
                            result = (min_diameter, angle)
                            min_rotation_diameter = region.axis_major_length
                            break
                    data[angle] = (img, angle, centers, center_value, ct_slice, regions, min_rotation_diameter, min_diameter)
            else:
                return None
            
        Plot.plot_finding_min_cross_section(data, debug_mode)
        return result

    cross_sections = []
    for i, center in enumerate(centerline):
        center_value = image[center[0],center[1], center[2]]
        cross_section = find_min_cross_section(center_value, images)
        if cross_section is not None:
            cross_sections.append([center, cross_section[0], cross_section[1]])    

    return cross_sections 


def detect_aneurysm(
        cross_sections, 
        bound_angles,
        abdominal_and_descending_aorta_threshold, 
        aortic_arch_and_asceding_aorta_threshold, 
        aneurysm_width_tolerance,
        bound_step = 2,
):

    def get_index_bound():
        index_threshold = round(len(cross_sections) * 0.6)
        for i,cross_section in enumerate(cross_sections):
            center, diameter, angles = cross_section
            if angles[0] == bound_angles[0] and i >= index_threshold:
                return i
            
    def find_start_cross_section_index(current_index):
        bound_index = max(0, current_index - aneurysm_width_tolerance - 1)
        previous_diameter = cross_sections[current_index][1]
        for index in range(current_index, bound_index, -bound_step):
            _, diameter, _ = cross_sections[index]
            if diameter > previous_diameter:
                return index + 1
            previous_diameter = diameter
        return bound_index + 1
    
    def find_end_cross_section_index(current_index):
        bound_index = min(len(cross_sections), current_index + aneurysm_width_tolerance + 1)
        previous_diameter = cross_sections[current_index][1]
        for index in range(current_index, bound_index, bound_step):
            _, diameter, _ = cross_sections[index]
            if diameter > previous_diameter:
                return index - 1
            previous_diameter = diameter
        return bound_index - 1
    
    index_bound = get_index_bound()
    bound_cross_sections = []
    bound_cross_sections_inds = []
    start_cross_section_index = None
    end_cross_section_index = None

    for index,cross_section in enumerate(cross_sections):
        threshold = (abdominal_and_descending_aorta_threshold if index < index_bound 
                    else aortic_arch_and_asceding_aorta_threshold)
        center, diameter, angles = cross_section

        if diameter > threshold and start_cross_section_index == None:
                start_cross_section_index = find_start_cross_section_index(index)
                if end_cross_section_index != None and start_cross_section_index <= end_cross_section_index:
                    bound_cross_sections.pop()
                    bound_cross_sections_inds.pop()
                else:
                    bound_cross_sections.append(cross_sections[start_cross_section_index])
                    bound_cross_sections_inds.append(start_cross_section_index)
        elif start_cross_section_index != None:
                end_cross_section_index = find_end_cross_section_index(index)
                bound_cross_sections.append(cross_sections[end_cross_section_index])
                bound_cross_sections_inds.append(end_cross_section_index)
                start_cross_section_index = None

    return bound_cross_sections, index_bound, bound_cross_sections_inds


import argparse


def run():
    parser = argparse.ArgumentParser(description="Detects aortic aneurysms based on the cross sections diameters.")
    parser.add_argument("-p","--path", type=str, help="Path to the nrrd image.")
    parser.add_argument("-c", "--closing_structure_shape", type=int, nargs=3, default=(5, 5, 1),
                        help="Shape of the structure for binary image closing.")
    parser.add_argument("-cla", "--consists_legs_artery", action='store_true', 
                        help="Indicates whether the image includes arteries located in the legs.")
    parser.add_argument("-of", "--outlier_factor", type=float, default=1.3,
                        help="Factor for detecting outliers based on the neighborhood average.")
    parser.add_argument("-ns", "--neighborhood_size", type=int, default=5,
                        help="Size of the neighborhood for detecting outliers.")
    parser.add_argument("-odf", "--outlier_difference_factor", type=float, default=0.15,
                        help="Factor for detecting outliers based on the difference between the current and first neighbors.")
    parser.add_argument("-cs","--comparison_range", type=int, default=2, 
                        help="Distance to the previous and next neighbors used for comparison to determine outlier")
    parser.add_argument("-ba", "--bound_angles", type=int, nargs=2, default=(135, 0),
                        help="Angles for rotation around the x and y axes used to find the boundary between two different parts of the aorta.")
    parser.add_argument("-adt", "--abdominal_and_descending_aorta_threshold", type=int, default=55,
                        help="Threshold for the aortic diameter for the abdominal and descending part of the aorta.")
    parser.add_argument("-aat", "--aortic_arch_and_asceding_aorta_threshold", type=int, default=60,
                        help="Threshold for the aortic diameter for the aortic arch and ascending part of the aorta.")
    parser.add_argument("-awt", "--aneurysm_width_tolerance", type=int, default=6,
                        help="Maximum allowed distance of the boundary from the point that is first or last above the aneurysm threshold.")
    parser.add_argument("-bs", "--bound_step", type=int, default=2,
                        help="Step used to find aneurysm bound cross section.")
    parser.add_argument("-d", "--debug_mode", action='store_true', 
                        help="Activates debugging mode for plotting each algorithm step.")

    args = parser.parse_args()

    detect(
        args.path, 
        args.closing_structure_shape,
        args.consists_legs_artery,
        args.outlier_factor,
        args.neighborhood_size,
        args.outlier_difference_factor,
        args.comparison_range,
        args.bound_angles,
        args.abdominal_and_descending_aorta_threshold,
        args.aortic_arch_and_asceding_aorta_threshold,
        args.aneurysm_width_tolerance,
        args.bound_step,
        args.debug_mode
    )


def process_files_in_path(path):
    # Proverava da li je putanja validna
    if not os.path.exists(path):
        print(f"Putanja '{path}' nije validna.")
        return
    
    # Prolazi kroz sve fajlove u putanji
    for root, dirs, files in os.walk(path):
        for file_name in files:
            # Proverava da li je fajl sa ekstenzijom .seg.nrrd
            if file_name.endswith('.seg.nrrd'):
                # Formira punu putanju do fajla
                file_path = os.path.join(root, file_name)
                
                # Poziva funkciju detect
                detect(file_path)

if __name__ == '__main__':
    run()
    # path = '/Users/sasamarjanovic/Documents/ferit/dipl/obrada_slike_i_racunali_vid/detection_of_aortic_aneurysm/data/Rider/R1 (AD)/R1.seg.nrrd'
    # detect = Detect(path, consists_legs_artery=True)
    # detect.detect()
    # path = '/Users/sasamarjanovic/Documents/ferit/dipl/obrada_slike_i_racunali_vid/detection_of_aortic_aneurysm/data/Dongyang'
    # process_files_in_path(path)