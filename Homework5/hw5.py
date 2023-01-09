import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
import random


def compute_F(pts1, pts2):
    # print("pts1  : ", len(pts1))

    # TO DO
    print("compute_F called")

    l1 = len(pts1)
    l2 = len(pts2)
    # print("l1 = ",l1)
    
    
    max_inlier_so_far = 0
    ransac_iters = 5000
    best_F = np.zeros((3,3))
    # inlier_max = 0
    thresh = 1
    r_i = 0
    # min_loss = 99999999
    for r_i in range(ransac_iters):
    # while max_inlier_so_far < 260:

        indices = np.random.permutation(l1)
        pts1_permutated = pts1[indices]
        pts2_permutated = pts2[indices]

        pts1_considered = pts1_permutated[0:8:1,:]
        pts2_considered = pts2_permutated[0:8:1,:]
        # print("pts 1 considered : ", pts1_considered)
    
        pts1_3 = np.hstack((pts1, np.ones((len(pts1), 1)))) 
        pts2_3 = np.hstack((pts2, np.ones((len(pts1), 1))))
        # print("pts1 = ",pts1_considered)
        # print("pts2 = ",pts2_considered)
        A = np.zeros((8,9))
        
        
        # for i in range(8):

        # u = pts1_considered[i]
        # v = pts2_considered[i]
    
        
        A[:, 0] = pts1_considered[:,0] * pts2_considered[:,0]
        A[:, 1] = pts1_considered[:,1] * pts2_considered[:,0]
        A[:, 2] = 1 * pts2_considered[:,0]
        A[:, 3] = pts1_considered[:,0] * pts2_considered[:,1]
        A[:, 4] = pts1_considered[:,1] * pts2_considered[:,1]
        A[:, 5] = 1 * pts2_considered[:,1] 
        A[:, 6] = pts1_considered[:,0] * 1
        A[:, 7] = pts1_considered[:,1] * 1
        A[:, 8] = 1 * 1

        # print("A is ", A)
        # print("A looks like this :", np.shape(A))    
        u, s, Vt = np.linalg.svd(A, full_matrices=False)
        # last = Vt[-1, :]
        last = Vt.T[ :,-1]
        l_reshape = last.reshape((3,3))
        u, s, Vt = np.linalg.svd(l_reshape, full_matrices=False)
        s[2] = 0
        diag_s = np.diag(s, k=0)


        # t = np.dot(diag_s, Vt)
        # F = np.dot(u,t)
        F = u @ diag_s @ Vt

    
        

        loss = 0
            
        # for p1, p2 in zip(pts1,pts2) :
        #     p1_v = np.asarray([p1[0],p1[1],1])
        #     p2_v = np.asarray([p2[0],p2[1],1])
        #     t = np.matmul(p2_v,F)
        #     sample_loss = np.dot(t,p1_v)
        #     loss += sample_loss**2

        #Ithe we got the loss of this F for all pts.
        # if loss < min_loss:
        #     min_loss = loss
        #     best_F = F
        #     print("loss = ", loss)
        # if loss < 1:
        #     break

        inliers = 0
        for i in range(len(pts1)):
            d = abs(pts2_3[i,:] @ F @ (pts1_3[i,:].T) )

            if d < thresh:
                inliers += 1

        if inliers > max_inlier_so_far:
            best_F = F
            max_inlier_so_far = inliers
            print("inliers, iteration no = ", inliers,r_i)
        r_i += 1
    # print("F shape = ", np.shape(best_F))
    print("compute_F finished")
    return best_F






def triangulation(P1, P2, pts1, pts2):
    # TO DO
    print("triangulation started")

    l = len(pts1)
    pts3D = np.zeros((l,3))

    for i in range(l):
        pt1 = pts1[i,:]

        pt2 = pts2[i,:]
        # skew symetric asle pahijet
        P1_mul = np.array([[0,-1,pt1[1]],[1,0,-pt1[0]],[-pt1[1],pt1[0],0]])
        # skew symetric asle pahijet
        P2_mul = np.array([[0,-1,pt2[1]],[1,0,-pt2[0]],[-pt2[1],pt2[0],0]])
        
        # stack the 2 outputs vertically
        big_matrix = np.vstack((P1_mul@P1,P2_mul@P2))

        u, s, vh = np.linalg.svd(big_matrix, full_matrices=True)
        vh_T = vh.T
        M2 = vh_T[:,-1]

        M2 = M2/M2[3] # last element has to be 1

        pts3D[i,:] += M2[:-1]

    print("triangulation finished")
    return pts3D


def disambiguate_pose(Rs, Cs, pts3Ds):
    # TO DO

    print("diambiguate_pose started")
    # print("Rs ", Rs)
    # print("Cs ", Cs)
    # print("pts3Ds ", pts3Ds)
    max_count = 0

    for i in range(len(Rs)):



        count = 0
        R = Rs[i]
        C = Cs[i]
        Pt3D = pts3Ds[i]


        for j in Pt3D:
            # print("Look here : ", j)
            M = j.reshape(3,1)
            
            condition1 = (R[2, :]) @ M
            # print("condition 1 : ", condition1)
            condition2 = j[2] 
            #pratyek vela same kartoy veglya combinations sathi. 
            #Max uchlaycha
            if (condition1 > 0) and (condition2 > 0):
                count += 1
        print("count : ", count)
        if count >= max_count:
            max_count = count
            R_best = R
            C_best = C
            pts3D_best = pts3D

    print("disambiguate_pose finished")
    return R_best, C_best, pts3D_best




def compute_rectification(K, R, C):
    print("compute rectification started")
  
    
    

    C_normalized = C / np.linalg.norm(C)
    eye_3 = (np.eye(3)[2]).reshape(3,1)
#    colmn vector
    # print("Eye :, ", np.eye(3)[:,2] ) 
    # print("Eye :, ", np.eye(3)[2].T )

    lh = np.dot(C_normalized.T , eye_3)


    r_z = eye_3 - lh * C_normalized


    r_z_normalized = r_z / np.linalg.norm(r_z)


    r_y = np.cross(r_z_normalized.T , C_normalized.T) #r_y will ne orthogonal to other 2
    # LEcture slides

    rectification =  np.vstack((C_normalized.T,r_y,r_z_normalized.T))

    K_1 = np.linalg.inv(K)

    H1_r = rectification @ K_1
    # print(np.shape(H1_r))

    H2_r2 = R.T @ K_1
    # print(np.shape(H1_r2))
    H2_r = rectification @ H2_r2
    # print(np.shape(H2_r))


    H1 = K @ H1_r
    H2 = K @ H2_r

    print("H1 :", H1)
    print("H2 :", H2)
    print("compute rectification finished")
    return H1, H2


def dense_match(img1, img2, descriptors1, descriptors2):
    # TO DO
    print("dense_match started")
    disparity = np.zeros_like(img2)


    for i in range(len(img1)):

        nbrs = NearestNeighbors(n_neighbors=1).fit(descriptors1[i])

        d, indices = nbrs.kneighbors(descriptors2[i])

        # print(indices)
        # print(len(indices),len(indices[0]))

        index_distance = range(len(indices)) - indices.reshape(1,480)
        d = d.reshape(1,480)
        # print(len(index_distance))
        i_2 = np.square(index_distance)       

        # print(len(i_2),len(i_2[1])) 
        # d = np.reshape(d,(len(img2[1]),1))
        d = np.asarray(d)
        d_2 = np.abs(d)

        # d_2 = np.array(d*d)
        # print("disparity cha shape = ", np.shape(disparity[i] ))
        # print("d_2 cha shape = ", np.shape(d_2) )

        disparity[i] = d_2.ravel()
        print("dense_match finished")

    return disparity


# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()


def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = (F @ np.array([[p[0], p[1], 1]]).T).flatten()
    p1, p2 = (0, int(-el[2] / el[1])), (img.shape[1], int((-img_width * el[0] - el[2]) / el[1]))
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
        ax.title.set_text('Configuration {}'.format(i))
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
        ax.title.set_text('Configuration {}'.format(i))
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    disparity[disparity > 150] = 150
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('./left.bmp', 1)
    img_right = cv2.imread('./right.bmp', 1)
    visualize_img_pair(img_left, img_right)

    # Step 0: get correspondences between image pair
    data = np.load('./correspondence.npz')
    pts1, pts2 = data['pts1'], data['pts2']
    visualize_find_match(img_left, img_right, pts1, pts2)

    # Step 1: compute fundamental matrix and recover four sets of camera poses
    F = compute_F(pts1, pts2)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs)

    # Step 2: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 3: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

    # Step 4: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w)

    # Step 5: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    # data = np.load('./resource/hw5/dsift_descriptor.npz')
    data = np.load('./dsift_descriptor.npz')
    desp1, desp2 = data['descriptors1'], data['descriptors2']
    disparity = dense_match(img_left_w, img_right_w, desp1, desp2)
    visualize_disparity_map(disparity)
