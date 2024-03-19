import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from aneurysm_detection.utils import Utils



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
            Plot.__show_voxel_image(ax, mask, mask_threshold, face_color)

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
    def plot_slice_plane(center, center_value, angles, radius, image, debug_mode = False):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        Plot.__show_voxel_image(ax, image)
        ax.scatter(center[0], center[1], center[2], color="red", s=3)
        Plot.__show_slice_plane(ax, center, angles)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f"{center} - {center_value} - {angles} - {radius}")
        plt.show(block=debug_mode)
        plt.close()


    @staticmethod
    def plot_finding_min_cross_section(data:dict, debug_mode = False):
        if debug_mode:
            fig = plt.figure()
            nrows = 6
            
            i = 1
            for key in data.keys():
                image, angles, center, center_value, ct_slice, regions, min_rotation_radius, min_radius = data[key]
                x,y,z = center[0][0], center[1][0], center[2][0]
                ax = fig.add_subplot(nrows, 4, i, projection='3d')
                ax.set_title(str((angles, len(center[0]), (x,y,z),center_value)))
                Plot.__show_voxel_image(ax, image)
                ax.scatter(center[0], center[1], center[2], color="red")

                i+=1
                ax = fig.add_subplot(nrows, 4, i)
                radiuses = [round(region.axis_major_length,2) for region in regions]
                title = f"{angles}, {round(np.float64(min_rotation_radius),2)}, {round(np.float64(min_radius), 2)}\n{radiuses}"
                ax.set_title(title)
                ax.imshow(ct_slice)
                ax.scatter(y,x, color="red", s=1)
                for region in regions:
                    x, y = round(region.centroid[0]), round(region.centroid[1])
                    ax.scatter(y,x, color="blue", s=1)
                i+=1
            plt.tight_layout()
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
    def plot_cross_sections(cross_sections, folder_path=None, debug_mode = False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        Plot.__show_cross_sections(ax, cross_sections)
        plt.ylabel('diameter')
        plt.xlabel('index')
        plt.title('cross sections diameters')
        if folder_path != None:
            plt.savefig(f"{folder_path}/cross_sections.png")
        plt.show(block = debug_mode)
        plt.close()


    @staticmethod 
    def plot_detection(image, bound_cross_sections, folder_path=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        Plot.__show_voxel_image(ax, image)
        if len(bound_cross_sections) > 0:
            plt.title("aneurysm is detected")
            for center, diameter, angles in bound_cross_sections:
                Plot.__show_slice_plane(ax, center, angles)
        else:
            plt.title("Aneurysm is not detected")
        if folder_path != None:
            plt.savefig(f"{folder_path}/detection.png")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()



    def __show_voxel_image(ax, image, threshold = 0.5, face_color = None):
        verts, faces, normals, values = marching_cubes(image, threshold)
        mesh = Poly3DCollection(verts[faces], alpha=0.2)
        if face_color is None:
            face_color = [0.5, 0.5, 1]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        ax.set_xlim(0, image.shape[0])
        ax.set_ylim(0, image.shape[1])
        ax.set_zlim(0, image.shape[2])


    def __show_slice_plane(ax, center, angles):
        normal_vector = Utils.get_normal_vector(angles)
        A,B,C,D = Utils.get_equation_of_plane(center, normal_vector)
        plane_radius = 15
        x_min, x_max, y_min, y_max, z_min, z_max = Utils.get_plane_bbox(plane_radius, center)

        y = np.arange(y_min, y_max, 0.1)
        x = np.arange(x_min, x_max, 0.1)
        x, y = np.meshgrid(x, y)
        z = (D - x*A - y*B) / C 

        if angles[0] == 90:
            y = center[1]
            z = np.clip(z, z_min, z_max)
        elif angles[1] == 90:
            x = center[0]
            z = np.clip(z, z_min, z_max)

        ax.plot_wireframe(x, y, z, color="red", alpha=0.2)

    def __show_cross_sections(ax, cross_sections, inds=None, color="blue"):
        if inds == None:
            inds = np.arange(len(cross_sections))
        y = [radius for center, radius, angles in cross_sections]
        ax.scatter(inds,y, s=3)