import cv2
import numpy as np
import matplotlib.pyplot as plt


# before submitting:
# sobel confirm vs differenciation 
# extract hog madhe image la stretch karaychay ka?
# use of np. functions allowed?
# Written report has what all?
# Zero division prevention needed? Arctan has ways to deal 




def get_differential_filter():

    # filter_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    # filter_y = np.transpose(filter_x)
    # confirm which one to use

    filter_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    filter_y = np.transpose(filter_x)

    # To do
    return filter_x, filter_y



def filter_image(im, filter):
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


def get_gradient(im_dx, im_dy):
    grad_mag = np.sqrt(np.square(im_dx)+np.square(im_dy))
    
    grad_angle = np.zeros((len(im_dx),len(im_dx[0])))
    for i in range(len(im_dx)):
        for j in range(len(im_dy)):
            if im_dx[i][j] == 0:
                im_dx[i][j] = 0.00001  # prevents division by zero #arctan2 checckout 
            grad_angle[i][j] = np.arctan(im_dy[i][j]/im_dx[i][j])
            if grad_angle[i][j] < 0 :
                grad_angle[i][j] = grad_angle[i][j] + np.pi

        
    # To do
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):

    M = len(grad_mag)//cell_size
    # print(len(grad_mag))
    # print(M)
    # print(len(grad_mag))
    
    # print(cell_size)
    N = len(grad_mag[0])//cell_size
    # print(N)
    bins = 6
    ori_histo = np.zeros((M,N,bins),dtype = float)

    for i_window in range(M):
        for j_window in range(N):
            mag_window = grad_mag[(cell_size*i_window):(cell_size*(i_window+1)),(cell_size*j_window):(cell_size*(j_window+1))]
            angle_window = grad_angle[(cell_size*i_window):(cell_size*(i_window+1)),(cell_size*j_window):(cell_size*(j_window+1))]
            # print(cell_size)
            # print(M)
            # print(cell_size*M)
            # print(grad_angle[0:8,0:8])
            # print(angle_window)

            for i in range(cell_size):
                for j in range(cell_size):

                    theta = angle_window[i][j]
                    if theta < (np.pi)/12:
                        ori_histo[i_window][j_window][0] += mag_window[i][j]
                    elif theta < (  (np.pi)/12    +         (np.pi)/6  ):
                        ori_histo[i_window][j_window][1] += mag_window[i][j]
                    elif theta < (  (np.pi)/12    +     2 * (np.pi)/6  ):
                        ori_histo[i_window][j_window][2] += mag_window[i][j]
                    elif theta < (  (np.pi)/12    +     3 * (np.pi)/6  ):
                        ori_histo[i_window][j_window][3] += mag_window[i][j]
                    elif theta < (  (np.pi)/12    +     4 * (np.pi)/6  ):
                        ori_histo[i_window][j_window][4] += mag_window[i][j]
                    elif theta < (  (np.pi)/12    +     5 * (np.pi)/6  ):
                        ori_histo[i_window][j_window][5 ] += mag_window[i][j]
                    else:
                        ori_histo[i_window][j_window][0] += mag_window[i][j]

    # To do
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    e = 0.001
    M = len(ori_histo)
    N = len(ori_histo[0])
    # print(M)
    # print(N)
    ori_histo_normalized = np.zeros((M-block_size+1, N-block_size+1 ,6*(block_size**2)),dtype = float)
    for i in range(len(ori_histo)-1):
        for j in range(len(ori_histo[0])-1):
            ori_histo_normalized[i][j] = np.concatenate((ori_histo[i][j],ori_histo[i][j+1],ori_histo[i+1][j],ori_histo[i+1][j+1]) )
            denominator = np.sqrt(sum(np.square(ori_histo_normalized[i][j])) + e**2)
            ori_histo_normalized[i][j] = ori_histo_normalized[i][j] / denominator
            # print(i,j,ori_histo_normalized[i][j])
    # To do
    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    # stretch to fit whole range karaycha ahe ka?

    cell_size = 8
    block_size = 2

    sobel_x,sobel_y = get_differential_filter()

    im_x = filter_image(im,sobel_x)
    im_y = filter_image(im,sobel_y)

    gradient_mag, gradient_angle = get_gradient(im_x,im_y)

    histogram = build_histogram(gradient_mag, gradient_angle,cell_size)
    hog_descriptor = get_block_descriptor(histogram,block_size).flatten() #flatten nahi kela tar exactly kuthe adaktay?
    hog = hog_descriptor.flatten()

    # To do

    # visualize to verify
    # visualize_hog(im, hog, 8, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()





def face_recognition(I_target, I_template):
    template_h, template_w = I_template.shape
    target_h , target_w = I_target.shape

    print(template_h, template_w)
    print(target_h , target_w)
    
    hog_template = extract_hog(I_template)
    hog_template_zero_mean = hog_template - np.mean(hog_template)
    hog_template_denominator = np.sqrt(sum(np.square(hog_template_zero_mean)))

    threshold_passed_boxes = []
    all_boxes = []
    heatmap_all = np.zeros((len(I_target[0]) - len(I_template[0]))//5)
                                                                # [::5]
    for box_corner_x in range(len(I_target) - len(I_template))[::5]:
        heatmap_row = []
        for box_corner_y in range(len(I_target[0]) - len(I_template[0]))[::5]:

            target_subset = I_target[box_corner_x:(box_corner_x + len(I_template)),box_corner_y:(box_corner_y + len(I_template[0]))]
            hog_target_subset = extract_hog(target_subset)
            hog_target_subset_zero_mean = hog_target_subset - np.mean(hog_target_subset)
            hog_target_denominator = np.sqrt(sum(np.square(hog_target_subset_zero_mean)))
            ncc_score = np.dot(hog_template_zero_mean ,hog_target_subset_zero_mean)/(hog_template_denominator*hog_target_denominator)
            all_boxes.append([box_corner_y,box_corner_x,ncc_score])
            heatmap_row.append(ncc_score)
            # print("ncc score: ", ncc_score)
            if ncc_score > 0.42:
                threshold_passed_boxes.append([box_corner_y,box_corner_x,ncc_score]) # throw away later, keep for visualization.
                # all_boxes.append([box_corner_y,box_corner_x,ncc_score])
        heatmap_all = np.vstack((heatmap_all,heatmap_row))

    plt.imshow(heatmap_all, cmap = 'hot', interpolation='nearest')
    plt.show

    bounding_boxes = []

   
    d = len(I_template)
    # print(len(threshold_passed_boxes))
    while len(threshold_passed_boxes) > 0:

        key = threshold_passed_boxes[0]
        bounding_boxes.append(key)
        repeated_boxes = []
        for box in threshold_passed_boxes:
            intersection = (min((key[0]+d),(box[0]+d)) - max(key[0], box[0])) * (min((key[1]+d),(box[1]+d)) - max(key[1], box[1]))
            if (intersection / (2*d*d - intersection)) >= 0.5:
                repeated_boxes.append(box)

        # simultaneous is not happening for some reason, check with TA
        # repeated_boxes = []
                
        for destroy in repeated_boxes:
            threshold_passed_boxes.remove(destroy)


    # bounding_boxes = all_boxes
    return np.array(bounding_boxes)


    # # hog_template = extract_hog(I_template)
    # print(bounding_boxes)

    # bounding_boxes = np.array(threshold_passed_boxes)
    # return  np.array(bounding_boxes)

def visualize_face_detection(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size 
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()




if __name__=='__main__':

    # im = cv2.imread('cameraman.tif', 0)
    # hog = extract_hog(im)

    I_target= cv2.imread('target.png', 0)
    #MxN image

    I_template = cv2.imread('template.png', 0)
    #mxn  face template

    bounding_boxes=face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png')
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
    #this is visualization code.




