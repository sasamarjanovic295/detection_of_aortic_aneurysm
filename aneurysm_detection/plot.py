import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from aneurysm_detection.utils import Utils
import cv2 as cv
import random as rng

class Plot:

    @staticmethod
    def plot_voxel_image(
        image, 
        mask=None, 
        image_threshold=0.5, 
        mask_threshold=0.5, 
        folder_path=None, 
        title=None,
        debug_mode = False
    ):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        Plot.__show_voxel_image(ax, image, image_threshold)

        if mask is not None:
            face_color = [1, 0, 0]
            Plot.__show_voxel_image(ax, mask, mask_threshold, 1, face_color)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        title = title if title != None else 'image'
        plt.title(title)
        if folder_path != None:
            plt.savefig(f"{folder_path}/{title}.png")
        plt.show(block=debug_mode)
        plt.close()


    @staticmethod
    def plot_each_slice_plane(cross_sections, image, debug_mode = False):
        if debug_mode:
            for cross_section in cross_sections:
                center, diameter, angles = cross_section
                center_value = image[center[0],center[1], center[2]]
                Plot.plot_slice_plane(
                    center, center_value, angles, 
                    diameter, image
                )


    @staticmethod
    def plot_slice_plane(center, angles, radius, image, debug_mode = False):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        Plot.__show_voxel_image(ax, image)
        ax.scatter(center[0], center[1], center[2], color="red", s=3)
        Plot.__show_slice_plane(ax, image, center, angles)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f"{center} -  {angles} - {radius}")
        plt.show(block=debug_mode)
        plt.close()


    @staticmethod
    def plot_finding_min_cross_section(image, result,data:dict, debug_mode = False):
        if debug_mode:
            fig = plt.figure()
            i = 1
            for key in data.keys():
                rotated_image, angles, center, ct_slice, regions, min_diameter = data[key]
                cord = rotated_image.shape[0]
                ax = fig.add_subplot(5, 6, i)
                radiuses = [round(region.axis_major_length,2) for region in regions]
                title = f"{angles} {radiuses}"
                ax.set_title(title)
                ax.imshow(Plot.draw_contours(ct_slice))
                cord = ct_slice.shape[0] // 2
                ax.scatter(cord,cord, color="red", s=1)
                for region in regions:
                    x, y = round(region.centroid[0]), round(region.centroid[1])
                    ax.scatter(y,x, color="blue", s=1)
                i+=1

            fig = plt.figure()
            i = 1
            for key in data.keys():
                rotated_image, angles, center, ct_slice, regions, min_diameter = data[key]
                cord = rotated_image.shape[0] // 2
                ax = fig.add_subplot(5,6,i, projection='3d')
                ax.set_title(str((angles, center)))
                Plot.__show_voxel_image(ax, rotated_image)
                ax.scatter(cord, cord, cord, color="red")
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                i+=1

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.set_title(str((center, result[0], result[1])))
            # Plot.__show_voxel_image(ax, image)
            # ax.scatter(center[0], center[1], center[2], color="red", s=1)
            # Plot.__show_slice_plane(ax, image, center, result[1])
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            # i+=1
            # plt.show(block=debug_mode)
            # plt.close()
            
            plt.show(block=debug_mode)
            plt.close()


    @staticmethod
    def plot_filtering_cross_sections(
        cross_sections, 
        removed_cross_sections, 
        cross_sections_inds, 
        removed_cross_sections_inds,
        folder_path=None, 
        debug_mode = False
    ):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        Plot.__show_cross_sections(ax, cross_sections, cross_sections_inds)
        Plot.__show_cross_sections(ax, removed_cross_sections, removed_cross_sections_inds, color="red")
        plt.ylabel('diameter')
        plt.xlabel('index')
        plt.title('filtering cross sections')
        if folder_path != None:
            plt.savefig(f"{folder_path}/filtering_cross_sections.png")
        plt.show(block=debug_mode)
        plt.close()


    @staticmethod
    def plot_finding_bound_cross_sections(
        cross_sections, 
        bound_cross_sections,
        bound_cross_sections_inds,
        abdominal_and_descending_aorta_threshold, 
        aortic_arch_and_asceding_aorta_threshold, 
        bound, 
        folder_path=None,
        debug_mode = False
    ):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        Plot.__show_cross_sections(ax, cross_sections)
        Plot.__show_cross_sections(ax, bound_cross_sections, bound_cross_sections_inds, color="red")
        plt.hlines(
            y=[abdominal_and_descending_aorta_threshold, aortic_arch_and_asceding_aorta_threshold], 
            color='red', 
            xmin=[0,bound],xmax=[bound,len(cross_sections)-1], 
            linestyle='--', 
        )
        plt.ylabel('diameter')
        plt.xlabel('index')
        plt.title('finding bound cross sections')
        if folder_path != None:
            plt.savefig(f"{folder_path}/finding_bound_cross_sections.png")
        plt.show(block=debug_mode)
        plt.close()


    @staticmethod
    def plot_cross_sections(cross_sections, fit_values, folder_path=None, debug_mode = False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        Plot.__show_cross_sections(ax, cross_sections)
        ax.plot(fit_values, color="red")
        plt.ylabel('diameter')
        plt.xlabel('index')
        plt.title('cross sections diameters')
        if folder_path != None:
            plt.savefig(f"{folder_path}/cross_sections.png")
        plt.show(block = debug_mode)
        plt.close()


    @staticmethod 
    def plot_detection(
        image, 
        scaled_image,
        cross_sections,  
        bound_cross_sections, 
        bound_cross_sections_inds, 
        space_directions, 
        original_shape, 
        folder_path=None,
        bound_value = 2
    ):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        Plot.__show_voxel_image(ax, scaled_image)

        if len(bound_cross_sections) > 0:
            plt.title("Aneurysm is detected")    
            detection_mask = Utils.get_detection_mask(
                image, cross_sections,  
                bound_cross_sections, 
                bound_cross_sections_inds,  
                bound_value, space_directions, 
                original_shape
            )
            image[detection_mask] = bound_value
            scaled_image[detection_mask] = bound_value
            Plot.__show_voxel_image(ax, scaled_image, bound_value-1, 0.5,[1,0,0])
        
        else:
            plt.title("Aneurysm is not detected")
        if folder_path != None:
            plt.savefig(f"{folder_path}/detection.png")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()


    def __show_voxel_image(ax, image, threshold = 0.5, alpha = 0.2,face_color = [0.5, 0.5, 1]):
        verts, faces, normals, values = marching_cubes(image, threshold)
        mesh = Poly3DCollection(verts[faces], alpha=alpha)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        ax.set_xlim(0, image.shape[0])
        ax.set_ylim(0, image.shape[1])
        ax.set_zlim(0, image.shape[2])


    def __show_slice_plane(ax, image, center, angles, alpha = 0.5, face_color = [1,0,0]):
        image = (image > 0).astype(int)
        normal_vector = Utils.get_rotated_normal_vector(angles)
        A,B,C,D = Utils.get_equation_of_plane(center, normal_vector)
        plane_radius = 15
        x_min, x_max, y_min, y_max, z_min, z_max = Utils.get_plane_bbox(plane_radius, center, image.shape)
        points = Utils.get_bound_points(center, x_min, x_max, y_min, y_max, z_min, z_max, A, B, C, D)
        for point in points:
                image[point[0], point[1], point[2]] = 2
        Plot.__show_voxel_image(ax, image, 1, alpha, face_color)
        

    def __show_cross_sections(ax, cross_sections, inds=None, color="blue"):
        if inds == None:
            inds = np.arange(len(cross_sections))
        y = [radius for center, radius, angles in cross_sections]
        ax.scatter(inds,y, s=3)

    @staticmethod
    def draw_contours(binary_image):
        # Find contours in the binary image
        binary_image = np.array(binary_image, np.uint8)
        contours, _ = cv.findContours(binary_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        drawing = binary_image

        for i, c in enumerate(contours):
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            
            if len(c) > 5:
                ellipse = cv.fitEllipse(c)
                print("elipse", ellipse)
                #minimum width i hight
                cv.ellipse(drawing, ellipse, 2, thickness=1)

        return drawing