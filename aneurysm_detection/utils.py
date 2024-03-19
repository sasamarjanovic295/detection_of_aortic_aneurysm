import os
import numpy as np
import csv
import ast
import json

class Utils:

    __inv_rot_matrices = {}
    
    @staticmethod
    def get_normal_vector(angles):
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
    def get_plane_bbox(range, inds, ind= None):
        x_min = round(inds[0] - range + 1)
        x_max = round(inds[0] + range + 1)
        y_min = round(inds[1] - range + 1)
        y_max = round(inds[1] + range + 1)
        z_min = round(inds[2] - range + 1)
        z_max = round(inds[2] + range + 1)

        return x_min, x_max, y_min, y_max, z_min, z_max


    @staticmethod
    def get_equation_of_plane(point, normal_vector):
        A, B, C = normal_vector
        x0, y0, z0 = point
        D = A*x0 + B*y0 + C*z0

        return np.array([A,B,C,D])


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