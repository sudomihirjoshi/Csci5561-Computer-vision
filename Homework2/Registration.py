import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
import math

def find_match(img1, img2):
    #img1 is template
    #img2 is target
    

    

    print("img1 shape : ", np.shape(img1))
    print("img2 shape : ", np.shape(img2))
    
    sift1 = cv2.SIFT_create()
    kp1, des1 = sift1.detectAndCompute(img1,None)


    

    # kp1 = sift1.detect(img1,None)
    # print(kp1[0].pt)
    
    
    # img=cv2.drawKeypoints(img1,kp1,img1)
    # img=cv2.drawKeypoints(img1,kp1,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite('sift_keypoints_test1.jpg',img)
    # cv2.imshow("img1",img)
    # cv2.waitKey(0)


    sift2 = cv2.SIFT_create()
    kp2, des2 = sift2.detectAndCompute(img2,None)

    print("kp1 shape : ", np.shape(kp1))
    print("kp2 shape : ", np.shape(kp2))
    print("des1 shape : ", np.shape(des1))
    print("des2 shape : ", np.shape(des2))
    # print(kp2)
    # img=cv2.drawKeypoints(img2,kp2,img2)
    # cv2.imshow("img2",img)
    # cv2.waitKey(0) 

    # print("des1 length: ", len(des1))
    # print("kp1 length : ", len(kp1))
    # print("kp2 length : ", len(kp2))


    # img=cv2.drawKeypoints(img2,kp2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite('sift_keypoints_test2.jpg',img)


    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(des1) # train on template. this is the smaller of the two



    distances, indices = nbrs.kneighbors(des2) #fit the template. 
    # these indices are of the template image. 

    # print("distances :", distances)
    # print("distance length  :", len(distances))
    # print("indices lenght  :", len(indices))

    print("indices shape : ", np.shape(indices))
    # print(" shape : ", np.shape(des2))
    # print(indices[0])

    
    # x1 = indices[][0]
    # print("shape of indices :", x1.shape )
    # x2 = []
    # for i in kp2:
    #     x2.append(i.pt)
    # print("shape of x1 :", x1.shape)
    

    # x2 = np.array(x2)
    # print("shape of x2 :", x2.shape)

    x1 = []
    x2 = []
    accepted_distances = []
    accepted_indices = []


    for pint in range(len(indices)):


        if distances[pint][0] == 0: #avoid divide by 0
            continue
        ratio = distances[pint][0] / distances[pint][1] # chota / motha should be < 0.7 we want them far apart
        if ratio < 0.6:

            x2.append(kp2[pint].pt)



            x1.append(kp1[indices[pint][0]].pt)
            accepted_indices.append(indices[pint][0]) #of template
            # accepted_indices_target.append(kp2[(indexof)])
            


    # print("accepted distances : ", accepted_distances)
    # print("accepted indices : ", accepted_indices)
    # print(len(accepted_distances))


    

    x1 = np.array(x1)
    x2 = np.array(x2)


    

    # To do
    return x1, x2







def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):


    print(" template shape = ", np.shape(x1))
    # x1[:, [1, 0]] = x1[:, [0, 1]]
    # x2[:, [1, 0]] = x2[:, [0, 1]]
    
    # print(np.shape(x2))
    
    
    
    old_maximum = 0
    # for current_iteration in range(1):
    for current_iteration in range(ransac_iter):
        rand_indices = np.random.choice(a=np.arange(len(x1)), size=3, replace=False)
        # print(rand_indices)
        chosen_points_x1 = x1[rand_indices].transpose()
        chosen_points_x2 = x2[rand_indices].transpose()

        # chosen_points_x1 = np.append(chosen_points_x1,[1,1,1])
        # chosen_points_x2 = np.append(chosen_points_x2,[1,1,1]) 
        ones = np.ones(3)
        chosen_points_x1 = np.vstack([chosen_points_x1,ones])
        chosen_points_x2 = np.vstack([chosen_points_x2,ones])

        # print(chosen_points_x1)
        # print(chosen_points_x2)
        try:
            # x_1 = inv(chosen_points_x1)
            A = np.matmul(chosen_points_x2,np.linalg.inv(chosen_points_x1))
        except np.linalg.LinAlgError:
            continue
        # print("A : ", A)

        inliers = 0
        for i in range(len(x1)):

            input_pt = np.append(x1[i],np.ones(1))
            output_pred = np.dot(A,input_pt)

            error = np.sqrt(np.square(x2[i][0]-output_pred[0]) + np.square(x2[i][1]-output_pred[1]) )
            if error < ransac_thr:
                inliers += 1

        if inliers > old_maximum:
            old_maximum = inliers
            good_A = A
            print("inliers :",inliers)

            # print(output_pred)

    print("good_A",good_A) # Last row of matrix should be 001
    

    A = good_A




    # To do
    return A




    

def warp_image(img, A, output_size):
    # print("img shape :" , img.shape)
    # print("output_size : ", output_size)

    img_warped = -1 * np.ones(output_size) # unmapped pixel values stay -1. Need to map all
    # A_inv = np.linalg.inv(A)
    # print(output_size)
    # for i in range(output_size[0]):
    #     for j in range(output_size[1]):
    
    all_possible_coordinates_x,all_possible_coordinates_y = np.meshgrid(range(output_size[1]),range(output_size[0]),  indexing='xy'    )
    # print("all _ x : ", all_possible_coordinates_x)
    # print("all _ y : ", all_possible_coordinates_y)


    all_possible_coordinates = np.vstack(((all_possible_coordinates_x.flatten()),(all_possible_coordinates_y.flatten())))
    # print("shape of alll coordinates :" , all_possible_coordinates)
    
    ones = np.ones(len(all_possible_coordinates[0])).flatten()
    
    all_possible_coordinates = np.vstack((all_possible_coordinates,ones))

    all_pixels_source = np.matmul(A, all_possible_coordinates)


    # height, width = img.shape
    mask = (np.linspace(0, len(img[0]) - 1, len(img[0])), np.linspace(0, len(img) - 1, len(img)))


    img_warped = interpolate.interpn(mask, img.T, np.transpose(all_pixels_source)[:,:2], "linear", False, 0) #default 0


    img_warped = img_warped.reshape(output_size)
    # out_pixel = np.array([j,i,1])

    # source_in_img = np.matmul(A,out_pixel)
            # interpolation
            # corners
            # left_up = img[math.floor(source_in_img[1]), math.floor(source_in_img[0])]
            # left_down = img[math.ceil(source_in_img[1]), math.floor(source_in_img[0])]
            # right_up = img[math.floor(source_in_img[1]), math.ceil(source_in_img[0])]
            # right_down = img[math.ceil(source_in_img[1]), math.ceil(source_in_img[0])]

            # left_gap = source_in_img[0] - math.floor(source_in_img[0])
            # right_gap = math.ceil(source_in_img[0]) - source_in_img[0]

            # up_gap = math.ceil(source_in_img[1]) - source_in_img[1]
            # down_gap = source_in_img[1] - math.floor(source_in_img[1])

            # pixel value is weighted mean of corners

            #img_warped[i,j] = left_up * right_gap * up_gap + left_down * right_gap * up_gap + right_up * up_gap * left_gap + right_down * left_gap * down_gap



            # if int(source_in_img[0]) < len(img) and int(source_in_img[1]) < len(img[0]):
            #     img_warped[i,j] = img[int(source_in_img[0]),int(source_in_img[1])]

    # To do
    # plt.imshow(img, cmap='gray')
    # plt.title('function cha argument')
    # plt.show()
    # plt.imshow(img_warped, cmap='gray')
    # plt.title('function cha output')
    # plt.show()



    return img_warped




##############################################################

def get_differential_filter(): #from assignment1

    # filter_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    # filter_y = np.transpose(filter_x)
    # confirm which one to use

    filter_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    filter_y = np.transpose(filter_x)

    # To do
    return filter_x, filter_y

def filter_image(im, filter): #from assignment1
    im_height,im_width = np.shape(im)
    im_filtered = np.zeros((im_height,im_width))

    pad_margin = len(filter)//2
    im_padded = np.pad(im, [(pad_margin, pad_margin), (pad_margin, pad_margin)], mode='constant', constant_values=0)

    for up_down in range(im_height):
        for left_right in range(im_width):

            # im_filtered[i][j] = filter[0][0]*im_padded[i-1,j-1] + filter[0][1]*im_padded[i-1,j] + filter[0][2]*im_padded[i-1,j+1] + filter[1][0]*im_padded[i,j-1] + filter[1][1]*im_padded[i,j] + filter[1][2]*im_padded[i,j+1] + filter[2][0]*im_padded[i+1,j-1] + filter[2][1]*im_padded[i+1,j] + filter[2][2]*im_padded[i+1,j+1]
            im_filtered[up_down,left_right] = np.dot(filter.flatten(order='C'),im_padded[up_down:up_down+len(filter),left_right:left_right+len(filter)].flatten(order='C'))
    # To do
    return im_filtered




def align_image(template, target, A):
    Itpl = template
    # total_iterations = 2
    # total_iterations = 1999
    # total_iterations = 760
    total_iterations = 190  # works the best
    A_refined = A 
    print("A type : ", A.dtype)
    epsilon = 0.005
    #Spuedocode:

    #1: Initialize p = p0 from input A.
    # p = (A[0,0]-1,A[0,1],A[0,2],A[1,0],A[1,1]-1,A[1,2]) #useless ?

    #2: Compute the gradient of template image, ∇Itpl
    sobel_x, sobel_y = get_differential_filter()
    dIdx = filter_image(template,sobel_x)
    dIdy = filter_image(template,sobel_y)

    # kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    # kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    # dIdx = cv2.filter2D(template, -1, kernelx)
    # dIdy = cv2.filter2D(template, -1, kernely)
    # suggested by TA

    #3: Compute the Jacobian ∂W/∂p at (x; 0).

    #4: Compute the steepest decent images ∇Itpl*(∂W/∂p)
    t_v = len(template)
    t_h = len(template[0])
    steepest_descent = np.zeros((t_v, t_h , 6))
    for v in range(t_v):
        for h in range(t_h):
            steepest_descent[v,h] =  np.array([dIdx[v,h],dIdy[v,h]])@ np.array([[v,h,1,0,0,0],[0,0,0,v,h,1]]) 
    
    # print("sttepest descent 0 : ", steepest_descent[0,0])

    #5: Compute the 6 × 6 Hessian H =Sigma([∇I(tpl)*∂W/∂p]T [∇I(tpl)∂W/∂p]
    Hessian = np.zeros((6,6))
    for v in range(t_v):
        for h in range(t_h):
            Hessian += np.outer(steepest_descent[v,h],steepest_descent[v,h]) 
    # print(Hessian)


    errors = []
    #6: while True do
    for iter in range(total_iterations):
        #7: Warp the target to the template domain Itgt(W(x; p)).
        Itgt = warp_image(target,A_refined,np.shape(template))

        print("Iter ", iter )
        #8: Compute the error image Ierr = Itgt(W(x; p)) − Itpl.
        Ierr = Itgt - Itpl

        errors.append(np.sqrt(np.sum( np.square(Ierr) )  )  )
        #9: Compute F =Sigma[∇I(tpl)∂W/∂p]T Ierr
        steepest_descent = np.transpose(steepest_descent.reshape(-1, 6))
        F = np.matmul(steepest_descent, Ierr.flatten())
        #for v in range(t_v):
            #for h in range(t_h):
                # print("Ierr = ", Ierr[v,h])
                #Ierr_column = np.array([[Ierr[v,h]],[Ierr[v,h]],[Ierr[v,h]],[Ierr[v,h]],[Ierr[v,h]],[Ierr[v,h]]])
                # print("steepest descent transpose [0,0] :", (np.transpose(steepest_descent[v,h])))
                # print("steepest descent transpose shape :", np.shape(np.transpose(steepest_descent[v,h])))
                # print("Ierr_column shape :", np.shape(Ierr_column))
                # print( v, h )
                #F +=( ( np.transpose(steepest_descent[v,h]) * Ierr[v,h]).reshape(6,1) )
            # print("row : " , v)
        #10: Compute ∆p = H−1F.
        detla_p = np.matmul(np.linalg.inv(Hessian),F) 
        # print("A array from p : ", np.array([[1 + detla_p[0], detla_p[1], detla_p[2]],[detla_p[3], 1 + detla_p[4], detla_p[5]],[0, 0, 1],]) )
        #11: Update W(x; p) ← W(x; p) ◦ W−1(x; ∆p) = W(W−1(x; ∆p); p)
        A_refined = np.matmul(A_refined, np.linalg.inv(np.array([[1 + detla_p[0], detla_p[1], detla_p[2]],[detla_p[3], 1 + detla_p[4], detla_p[5]],[0, 0, 1],],dtype=float)))
        #12: if ∥∆p∥ < ϵ then
        if (np.linalg.det(np.array([[1 + detla_p[0], detla_p[1], detla_p[2]],[detla_p[3], 1 + detla_p[4], detla_p[5]],[0, 0, 1],]))) < epsilon:
            break
            #13: break
            #14: end if
        #15: end while
    #16: Return A_refined made of p.

    # To do
    errors = np.array(errors)
    print("errors : ", errors)
    return A_refined,errors



def track_multi_frames(template, img_list):
    A_list = []
    x1, x2 = find_match(template, target_list[0]) #survati cha goshti main sarkhyach 
    A_og = align_image_using_feature(x1, x2, ransac_thr = 50, ransac_iter = 1000)



    for target in img_list:
        A,useless = align_image(template, target, A_og)
        A_list.append(A)

    # To do
    return A_list




def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image_using_feature(img1, img2, x1, x2, A, ransac_thr, img_h=500):
    x2_t = np.hstack((x1, np.ones((x1.shape[0], 1)))) @ A.T
    errors = np.sum(np.square(x2_t[:, :2] - x2), axis=1)
    mask_inliers = errors < ransac_thr
    boundary_t = np.hstack(( np.array([[0, 0], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]], [0, img1.shape[0]], [0, 0]]), np.ones((5, 1)) )) @ A[:2, :].T

    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    boundary_t = boundary_t * scale_factor2
    boundary_t[:, 0] += img1_resized.shape[1]
    plt.plot(boundary_t[:, 0], boundary_t[:, 1], 'y')
    for i in range(x1.shape[0]):
        if mask_inliers[i]:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'g')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'go')
        else:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'r')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'ro')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        # boundary_t = np.hstack((np.array([  [template.shape[1], 0], [0, 0],
        #                                 [0, template.shape[0]], [0, 0], [template.shape[1], template.shape[0]]]), np.ones((5, 1)))) @ A[:2, :].T

        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    print("Bbox_list[0] = ", bbox_list[0])
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./JS_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./JS_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    A = align_image_using_feature(x1, x2, ransac_thr = 50, ransac_iter = 1000)
    visualize_align_image_using_feature(template, target_list[0], x1, x2, A, ransac_thr=50, img_h=500)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.title('Professor ni lihilela')
    plt.axis('off')
    plt.show()
    # img_warped = warp_image(target_list[1], A, template.shape)
    # plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    # plt.title('Professor ni lihilela')
    # plt.axis('off')
    # plt.show()

    # A_refined = align_image(template, target_list[0], A)
    A_refined, errors = align_image(template, target_list[1], A)
    visualize_align_image(template, target_list[1], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)


