"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple

import cv2

from numpy.linalg import svd
from scipy.interpolate import   griddata


PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                    match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        # return homography
        """INSERT YOUR CODE HERE"""

        np.asarray(match_p_src)
        np.asarray(match_p_dst)

        match_pts = match_p_src.shape[1]

        x_s, y_s = match_p_src
        x_d, y_d = match_p_dst

        A = np.array([[x_s[0], y_s[0], 1, 0, 0, 0, (-x_d[0])*(x_s[0]), (-x_d[0])*(y_s[0]), (-x_d[0])],
                    [0, 0, 0, x_s[0], y_s[0], 1, (-y_d[0])*(x_s[0]), (-y_d[0])*(y_s[0]), (-y_d[0])]])

        for i in range(1, match_pts):

            A = np.append(A, [[x_s[i], y_s[i], 1, 0, 0, 0, (-x_d[i])*(x_s[i]), (-x_d[i])*(y_s[i]), (-x_d[i])],
                    [0, 0, 0, x_s[i], y_s[i], 1, (-y_d[i])*(x_s[i]), (-y_d[i])*(y_s[i]), (-y_d[i])]] , axis =0)

        _, _, Vt = svd(A.T @ A)

        eigvec = Vt[-1,:]
        eigvec_N = eigvec / eigvec[-1]

        H_naive = np.reshape(eigvec_N, (3,3))           

        return H_naive
        pass    

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        """INSERT YOUR CODE HERE"""
        
        dst_image = np.zeros(dst_image_shape, dtype=src_image.dtype)
        Y, X, _ = src_image.shape

        for y in range(Y):
            for x in range(X):
                projected_point = homography @ (np.array([x,y,1]).transpose())
                xp = round((projected_point[0]/projected_point[2]).astype(np.int32))
                yp = round((projected_point[1]/projected_point[2]).astype(np.int32))
                if ( xp>=0 and xp <= dst_image_shape[1] and yp>=0 and yp <= dst_image_shape[0]):
                    dst_image[yp,xp,:] = src_image[y,x,:]

        dst_image = dst_image.astype(np.uint8)
        return dst_image
        pass

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        """INSERT YOUR CODE HERE"""
        
        # creat matrix for storing the homogeneous coordinates of the source 
        # (x,y,1) vector for each pixel.

        H_src = src_image.shape[0]
        W_src = src_image.shape[1]

        x = np.linspace(0, W_src-1, W_src)
        y = np.linspace(0, H_src-1, H_src)

        xv, yv = np.meshgrid(x,y, indexing='ij')

        q = np.array([xv.flatten(),yv.flatten(),np.ones_like(yv.flatten())])


        # transform the projection onto the dst image in homogeneous plan
        p_homog = homography @ q

        # transform the dst image back to Carteisian plan
        p_cartes = (p_homog[0:2,:]/p_homog[2,:]).astype(np.int32)

        #
        xp = p_cartes[0,:]
        yp = p_cartes[1,:]

        valid = np.all(( xp>=0, xp < dst_image_shape[1], yp < dst_image_shape[0], yp>=0 ), axis=0)

        xp_v=xp[valid]
        yp_v=yp[valid]

        img_out = np.zeros((dst_image_shape))
        img_out[yp_v,xp_v] = src_image[q[1,valid].astype(np.int32),q[0,valid].astype(np.int32)]
        img_out = img_out.astype(np.uint8)

        return img_out
        pass

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        # return fit_percent, dist_mse
        """INSERT YOUR CODE HERE"""
         # Finding inliers
        
        # src image in homogenous coords
        Us = np.vstack((match_p_src, np.ones(match_p_src.shape[1])))
        # projecting src into dst
        Ud = homography @ Us
        Ud = Ud / Ud[2, :]
        # finding distances of calculated projections from ref match_p_dst
        distances = np.linalg.norm(match_p_dst - Ud[:2, ], axis=0)
        # classifying all points with error less than max_err as inliers
        inliers = distances <= max_err
        
        N = match_p_src.shape[1]
        
        if np.sum(inliers) == 0:
            fit_percent = 0
            dist_mse = 10 ** 9
        else:
            fit_percent = np.mean(inliers)
            dist_mse = np.mean(distances[inliers] ** 2)  # (np.sum(distances[inliers]) ** 2) / np.sum(inliers)
        
        return fit_percent, dist_mse.astype(np.int)
        
        pass

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        # return mp_src_meets_model, mp_dst_meets_model
        """INSERT YOUR CODE HERE"""
        
        # pm - perfect match
        # h - homogeneous
        # c - cartesias
        # s - src
        # d - dst

        pm_s_c = match_p_src
        pm_d_c = match_p_dst

        pm_h = np.append(pm_s_c, np.ones((1,pm_s_c.shape[1])), axis = 0)

        map_pm_h = homography @ pm_h
        map_mp_c = (map_pm_h/map_pm_h[2])[0:2,:].astype(np.int32)

        src_inlires = [[],[]]
        dst_inlires = [[],[]]

        for i in range(pm_s_c.shape[1]):
            dist_px = np.linalg.norm(map_mp_c[:,i] - pm_d_c[:,i])
            if (dist_px<max_err):
                src_inlires[0].append(pm_s_c[0,i])
                src_inlires[1].append(pm_s_c[1,i])
                dst_inlires[0].append(pm_d_c[0,i])
                dst_inlires[1].append(pm_d_c[1,i])

        mp_src_meets_model = np.asarray(src_inlires).astype(np.int32)
        mp_dst_meets_model = np.asarray(dst_inlires).astype(np.int32)
        return mp_src_meets_model, mp_dst_meets_model
        pass

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # use class notations:
        w = inliers_percent
        # t = max_err
        # p = parameter determining the probability of the algorithm to
        # succeed
        p = 0.99
        # the minimal probability of points which meets with the model
        d = 0.5
        # number of points sufficient to compute the model
        n = 4
        # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        # return homography
        """INSERT YOUR CODE HERE"""
        H = None
        prev_error = None
        H_new = None
        idxs = list(np.arange(len(match_p_src[0])))

        for i in range(k):
            # randomly select n points
            rand_n_pts = sample(idxs,4)
            
            # compute the model using the n points
            pm_n_src_pts = match_p_src[:, rand_n_pts]
            pm_n_dst_pts = match_p_dst[:, rand_n_pts]
            H_new = Solution.compute_homography_naive(pm_n_src_pts, pm_n_dst_pts)

            # calculate the probability of points which meets with the model
            inliers_percent, _ = Solution.test_homography(H_new, match_p_src, match_p_dst, max_err)

            if inliers_percent >= d :
                # recompute the model using all the inliers
                meets_model_src, meet_model_dst = Solution.meet_the_model_points(H_new, match_p_src, match_p_dst, max_err)
                H_new = Solution.compute_homography_naive(meets_model_src, meet_model_dst)
                _, dist_mse = Solution.test_homography(H_new,match_p_src,match_p_dst,max_err)

                if prev_error == None:
                    prev_error = dist_mse
                    H = H_new
                if dist_mse < prev_error:
                    prev_error = dist_mse
                    H = H_new

        return H
        pass

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """

        # return backward_warp
        """INSERT YOUR CODE HERE"""
        x = np.linspace(0,dst_image_shape[1]-1,dst_image_shape[1])
        y = np.linspace(0,dst_image_shape[0]-1,dst_image_shape[0])

        xv, yv = np.meshgrid(x.astype(np.int32),y.astype(np.int32), indexing='ij')
        xv_f = xv.flatten()
        yv_f = yv.flatten()
        ones_f = np.ones_like(xv_f)

        dst_h_grid = np.array([xv_f, yv_f, ones_f])

        new_dst_h_grid = backward_projective_homography @ dst_h_grid
        new_dst_c_grid = new_dst_h_grid / new_dst_h_grid[2]


        x_src = np.linspace(0,src_image.shape[1]-1,src_image.shape[1])
        y_src = np.linspace(0,src_image.shape[0]-1,src_image.shape[0])

        xv_src, yv_src = np.meshgrid(x_src.astype(np.int32),y_src.astype(np.int32), indexing='ij')
        xv_src_f = xv_src.flatten()
        yv_src_f = yv_src.flatten()


        src_c_grid = np.array([xv_src_f, yv_src_f], dtype=np.int32)

        src_values = src_image.T.reshape(3, -1)

        grid = griddata(src_c_grid.T, src_values.T, new_dst_c_grid[:2].T, 
                        method='cubic', fill_value=0)

        new_image = grid.reshape((dst_image_shape[0], dst_image_shape[1], 3), order='F')

        return np.clip(new_image, 0, 255)
        pass

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        

        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        # return final_homography
        """INSERT YOUR CODE HERE"""

        translation_matrix = [[1, 0, -pad_left], [0, 1, -pad_up], [0, 0, 1]]
        translation_matrix = np.array(translation_matrix)
        composed_homography = backward_homography @ translation_matrix
        final_homography = composed_homography / composed_homography[2, 2]

        return final_homography
        pass

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        # return np.clip(img_panorama, 0, 255).astype(np.uint8)
        """INSERT YOUR CODE HERE"""

        forward_homography = self.compute_homography(match_p_src ,match_p_dst, inliers_percent, max_err)
        panorama_rows_num, panorama_cols_num, pad_struct =  Solution.find_panorama_shape(src_image,dst_image,forward_homography)
        backward_homography = np.linalg.inv(forward_homography)
        backward_homography_translated = Solution.add_translation_to_backward_homography(backward_homography,pad_struct.pad_left,pad_struct.pad_up)
        backward_wraping = Solution.compute_backward_mapping(backward_homography_translated, src_image, (panorama_rows_num,panorama_cols_num)) # calculate src backward
        backward_wraping[pad_struct.pad_up: pad_struct.pad_up + dst_image.shape[0] ,pad_struct.pad_left : pad_struct.pad_left + dst_image.shape[1]] = dst_image # plant dst image over src backward

        return np.clip(backward_wraping, 0, 255).astype(np.uint8)
        pass
