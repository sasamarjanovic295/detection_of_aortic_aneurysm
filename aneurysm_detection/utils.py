import os
import queue
import numpy as np
import csv
import ast
import json

class Utils:

    __inv_rot_matrices = {}
    
    @staticmethod
    def get_normal_vector(angles):
        if(angles[0]):
            return [0,1,0]
        if(angles[1]):
            return [1,0,0]
        inv_rotation_matrix = Utils.__get_inv_rot_matrix(angles)
        normal_vector = np.array([0,0,1])
        return np.dot(inv_rotation_matrix, normal_vector)


    def __get_inv_rot_matrix(angles):
        if angles not in Utils.__inv_rot_matrices.keys():
            angle_x, angle_y = angles
            angle_x_rad=np.radians(angle_x)
            angle_y_rad = np.radians(angle_y)

            rotation_matrix_x = np.array([
                [1, 0, 0], 
                [0, np.cos(angle_x_rad), -np.sin(angle_x_rad)], 
                [0, np.sin(angle_x_rad), np.cos(angle_x_rad)]])
            rotation_matrix_y = np.array([
                [np.cos(angle_y_rad), 0, np.sin(angle_y_rad)], 
                [0, 1, 0], 
                [-np.sin(angle_y_rad), 0, np.cos(angle_y_rad)]])
            
            rotation_matrix = np.dot(rotation_matrix_x, rotation_matrix_y)
            inv_rotation_matrix = np.linalg.inv(rotation_matrix)
            Utils.__inv_rot_matrices[angles] = inv_rotation_matrix

        return Utils.__inv_rot_matrices[angles]


    @staticmethod
    def get_plane_bbox(range, inds):
        x_min = round(inds[0] - range - 1)
        x_max = round(inds[0] + range + 1)
        y_min = round(inds[1] - range - 1)
        y_max = round(inds[1] + range + 1)
        z_min = round(inds[2] - range - 1)
        z_max = round(inds[2] + range + 1)

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
            normal_vector = Utils.get_normal_vector(angles)
            A, B, C, D = Utils.get_equation_of_plane(center, normal_vector)
            x_min, x_max, y_min, y_max, z_min, z_max = Utils.get_plane_bbox(diameter, center)
            x = center[0] if angles[0] == 90 else np.arange(x_min, x_max)
            y = center[1] if angles[1] == 90 else np.arange(y_min, y_max)
            if 90 in angles:
                z = np.arange(z_min, z_max)
                x,y,z = np.meshgrid(x,y,z)
            elif np.isclose(C,0):
                z = center[2]
                x,y,z = np.meshgrid(x,y,z)
            else:
                x,y = np.meshgrid(x,y)
                z = (-D - A*x - B*y) / C
            points = np.column_stack((x.flatten(), y.flatten(), z.flatten())).astype(int)
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
            print(bound_cross_sections[i])
            image = add_bound_cross_sectiion(image, bound_cross_sections[i], bound_value)
            image = add_bound_cross_sectiion(image, bound_cross_sections[i+1], bound_value)
            next_cross_section_ind = bound_cross_sections_inds[i] + 1
            start_point, diameter, angles = cross_sections[next_cross_section_ind]
            image = flood_fill(image, bound_value, start_point, [0, bound_value])
            
        return image == bound_value
    
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
            cross_section[1] = cross_section[1] / np.sqrt(x_length**2 + y_length**2) + 10 #+ z_length**2)
            cross_sections[i] = cross_section
        return cross_sections
    
    @staticmethod
    def delete_elements_at_indices(arr, inds):
        inds = np.sort(inds)[::-1]
        for idx in inds:
            arr = np.delete(arr, idx)
        return arr