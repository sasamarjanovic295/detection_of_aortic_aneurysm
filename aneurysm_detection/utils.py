import os
import queue
import numpy as np
import csv
import ast
import json
from scipy.ndimage import zoom, rotate
import cv2 as cv

class Utils:
    __inv_rot_matrices = {}
    __rotation_angles = []
    __rotated_normal_vectors = {}

    @staticmethod
    def get_rotation_angles(angles):
        images = []
        image = np.zeros((31,31,31))
        image[:,:,15] = 1
        for angle_x in angles:
            for angle_y in angles:
                for angle_z in angles:
                    angle = (angle_x, angle_y, angle_z)
                    rotated_image = Utils.rotate_image(image, angle)
                    are_parallel = any(np.array_equal(rotated_image[10:21,10:21, 10:21], img[10:21,10:21, 10:21]) for img in images)
                    if len(images) == 0 or not are_parallel:
                        images.append(rotated_image)
                        Utils.__rotation_angles.append(angle)
        return Utils.__rotation_angles


    @staticmethod
    def get_rotated_normal_vector(angle, normal_vector=[0, 0, 1]):
        if angle not in Utils.__rotated_normal_vectors.keys():
            inv_rotation_matrix = Utils.__get_inv_rot_matrix(angle)
            rotated_normal_vector = np.dot(inv_rotation_matrix, normal_vector)
            print(rotated_normal_vector, angle)
            Utils.__rotated_normal_vectors[angle] = rotated_normal_vector
        return Utils.__rotated_normal_vectors[angle]


    @staticmethod
    def __get_inv_rot_matrix(angle):
        if angle not in Utils.__inv_rot_matrices:
            angle_x, angle_y, angle_z = angle
            angle_x_rad = np.radians(angle_x)
            angle_y_rad = np.radians(angle_y)
            angle_z_rad = np.radians(angle_z)
            rotation_matrix_x = np.array([
                [1, 0, 0],
                [0, np.cos(angle_x_rad), -np.sin(angle_x_rad)],
                [0, np.sin(angle_x_rad), np.cos(angle_x_rad)]
            ])
            rotation_matrix_y = np.array([
                [np.cos(angle_y_rad), 0, np.sin(angle_y_rad)],
                [0, 1, 0],
                [-np.sin(angle_y_rad), 0, np.cos(angle_y_rad)]
            ])
            rotation_matrix_z = np.array([
                [np.cos(angle_z_rad), -np.sin(angle_z_rad), 0],
                [np.sin(angle_z_rad), np.cos(angle_z_rad), 0],
                [0, 0, 1]
            ])
            rotation_matrix = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))
            inv_rotation_matrix = np.linalg.inv(rotation_matrix)
            Utils.__inv_rot_matrices[angle] = inv_rotation_matrix

        return Utils.__inv_rot_matrices[angle]
    
    @staticmethod
    def __are_parallel(u, v):
        u = np.array(u)
        v = np.array(v)

        dot_product = np.dot(u, v)

        norm_u_squared = np.dot(u, u)
        norm_v_squared = np.dot(v, v)

        if np.isclose(norm_u_squared, 0) or np.isclose(norm_v_squared, 0):
            return True  
        else:
            cosine_similarity = abs(dot_product / np.sqrt(norm_u_squared * norm_v_squared))
            return np.isclose(cosine_similarity, 1)



    @staticmethod
    def get_plane_bbox(range, inds, shape):
        x_min = max(round(inds[0] - range), 0)
        x_max = min(round(inds[0] + range), shape[0])
        y_min = max(round(inds[1] - range), 0)
        y_max = min(round(inds[1] + range), shape[1])
        z_min = max(round(inds[2] - range), 0)
        z_max = min(round(inds[2] + range), shape[2])

        return x_min, x_max, y_min, y_max, z_min, z_max


    @staticmethod
    def get_equation_of_plane(point, normal_vector):
        A, B, C = normal_vector
        x0, y0, z0 = point
        D = -A*x0 + -B*y0 + -C*z0
        return np.array([A,B,C,D])
    

    def is_point_on_plane(point, coefficients):
        A, B, C, D = coefficients
        x, y, z = point
        return np.isclose(A*x + B*y + C*z + D, 0)


    @staticmethod
    def get_cross_sections_file_path(path):
        folder_path = Utils.get_folder_path(path)
        file_path = f"{folder_path}/cross_sections.txt"
        return file_path
    

    @staticmethod
    def get_folder_path(path):
        _, name = path.split('/data/')
        name = name.split('/')[-1].split('.')[0]
        work_dir = os.getcwd()
        folder_path = f"{work_dir}/results/{name}"
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    

    @staticmethod
    def save_cross_sections(data, file_path):
        with open(file_path, 'w') as file:
            for cross_section in data:
                center = list(cross_section[0])
                diameter = cross_section[1]
                rotation_angles = cross_section[2]
                row = f"{center}-{diameter}-{rotation_angles}\n"
                file.write(row)


    @staticmethod
    def read_cross_sections(file_path):
        with open(file_path, 'r') as file:
            data = []
            for line in file:
                values = line.split('-')
                center_coordinates = ast.literal_eval(values[0])
                diameter = float(values[1])
                rotation_angles = ast.literal_eval(values[2])
                cross_section = (center_coordinates, diameter, rotation_angles)
                data.append(cross_section)
            return data
        

    @staticmethod
    def get_detection_mask(
        image,cross_sections,  
        bound_cross_sections, 
        bound_cross_sections_inds,   
        bound_value, space_directions, 
        original_shape
    ):

        def add_bound_cross_sectiion(image, bound_cross_section, bound_value):
            center, diameter, angles = bound_cross_section
            normal_vector = Utils.get_rotated_normal_vector(angles)
            A, B, C, D = Utils.get_equation_of_plane(center, normal_vector)
            x_min, x_max, y_min, y_max, z_min, z_max = Utils.get_plane_bbox(diameter, center, image.shape)
            points = Utils.get_bound_points(center, x_min, x_max, y_min, y_max, z_min, z_max, A, B, C, D)
            for point in points:
                if image[point[0], point[1], point[2]] != 0:
                    image[point[0], point[1], point[2]] = bound_value
            return image

        def flood_fill(image, new_value, start_point, boundary_values):
            filled_image = np.copy(image)
            depth, height, width = image.shape
            q = queue.Queue()
            q.put(start_point)
            neighbors_offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
            while not q.empty():
                current_point = q.get()
                z, y, x = current_point
                if (0 <= z < depth) and (0 <= y < height) and (0 <= x < width) and (filled_image[z, y, x] != new_value):
                    filled_image[z, y, x] = new_value
                    for dz, dy, dx in neighbors_offsets:
                        nz, ny, nx = z + dz, y + dy, x + dx
                        if (0 <= nz < depth) and (0 <= ny < height) and (0 <= nx < width) and (filled_image[nz, ny, nx] not in boundary_values):
                            q.put((nz, ny, nx))
            return filled_image
        
        bound_cross_sections = Utils.get_cross_sections_with_voxel_diameter(
                                    bound_cross_sections, 
                                    space_directions, 
                                    original_shape, 
                                    image.shape
                                )
        for i in range(0, len(bound_cross_sections), 2):
            image = add_bound_cross_sectiion(image, bound_cross_sections[i], bound_value)
            if i+1 < len(bound_cross_sections):
                image = add_bound_cross_sectiion(image, bound_cross_sections[i+1], bound_value)
            next_cross_section_ind = bound_cross_sections_inds[i] + 2
            start_point, _, _ = cross_sections[next_cross_section_ind]
            image = flood_fill(image, bound_value, start_point, [0, bound_value])
            
        return image == bound_value
    
    @staticmethod
    def get_bound_points(center, x_min, x_max, y_min, y_max, z_min, z_max, A, B, C, D):
        x = np.arange(x_min, x_max, 0.1)
        y = np.arange(y_min, y_max, 0.1)
        if np.isclose(C,0):
            z = center[2] 
            x,y,z = np.meshgrid(x,y,z)
        else:
            x,y = np.meshgrid(x,y)
            z = (-D - A*x - B*y) / C
        points_out = np.column_stack((x.flatten(), y.flatten(), z.flatten())).astype(int)
        return list(filter(lambda p: z_min <= p[2] < z_max, points_out))
    
    @staticmethod
    def get_cross_sections_with_voxel_diameter(
                cross_sections, 
                space_directions, 
                original_shape, 
                scaled_shape,
    ):
        x_direction, y_direction, z_direction = space_directions
        x_scale = original_shape[0] / scaled_shape[0]
        y_scale = original_shape[1] / scaled_shape[1]
        x_length = np.sqrt(x_direction[0]**2 + x_direction[1]**2 + x_direction[2]**2) * x_scale
        y_length = np.sqrt(y_direction[0]**2 + y_direction[1]**2 + y_direction[2]**2) * y_scale
        for i in range(0, len(cross_sections)):
            cross_section = list(cross_sections[i])
            cross_section[1] = cross_section[1] / np.sqrt(x_length**2 + y_length**2) #+ z_length**2)
            cross_sections[i] = cross_section
        return cross_sections
    
    @staticmethod
    def delete_elements_at_indices(arr, inds):
        inds = np.sort(inds)[::-1]
        for idx in inds:
            arr = np.delete(arr, idx)
        return arr
    
    @staticmethod
    def extract_subimage(image, radius, center):
        x_dim, y_dim, z_dim = image.shape
        x_max_radius = min(center[0], x_dim - center[0])
        y_max_radius = min(center[1], y_dim - center[1])
        z_max_radius = min(center[2], z_dim - center[2])
        x_radius = min(radius, x_max_radius)
        y_radius = min(radius, y_max_radius)
        z_radius = min(radius, z_max_radius)
        x_start = center[0] - x_radius
        y_start = center[1] - y_radius
        z_start = center[2] - z_radius
        x_end = center[0] + x_radius + 1
        y_end = center[1] + y_radius + 1
        z_end = center[2] + z_radius + 1
        return image[x_start:x_end, y_start:y_end, z_start:z_end]
    
    @staticmethod
    def rotate_image(image, angles):
        angle_x, angle_y, angle_z = angles
        rotated_image = rotate(image, angle_x, axes=(1,2), reshape=False, order=0, prefilter=False)
        rotated_image = rotate(rotated_image, angle_y, axes=(0,2), reshape=False, order=0, prefilter=False)
        rotated_image = rotate(rotated_image, angle_z, axes=(0,1), reshape=False, order=0, prefilter=False)
        return rotated_image