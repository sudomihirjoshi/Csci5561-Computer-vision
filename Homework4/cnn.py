import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main


def get_mini_batch(im_train, label_train, batch_size):
    # TO DO
    # im_train = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
    im_train = im_train.T

    labels_encoded = []
    # label_train = label_train.T
    for i in label_train:
        labels_encoded.append(np.eye(10)[i])
    labels_encoded = np.array(labels_encoded).reshape(12000,10)
    # print("shape of encoded images = ", np.shape(labels_encoded))
    # print("labels = ", label_train[:5] )
    # print("labels encoded= ", labels_encoded[:5] )
    # print("length of im_train = ", len(im_train))
    random = np.random.permutation(len(im_train))

    # print("length = ",len(im_train))
    # print("length = ",len(im_train[0]))
    # print("length = ",len(label_train))

    im_train_permutated = im_train[random]
    label_train_permutated = labels_encoded[random]

    number_of_batches = int(len(im_train)/batch_size)
    # print("batch size = ", batch_size)
    # print("number of batches = ", number_of_batches)
    mini_batch_x = []
    mini_batch_y = []
    
    for i in range(number_of_batches):
        mini_batch_x.append(im_train_permutated[i*batch_size:(i+1)*batch_size].T)
        mini_batch_y.append(label_train_permutated[i*batch_size:(i+1)*batch_size].T)
    
    # print(np.shape(mini_batch_y))

    return mini_batch_x, mini_batch_y


# 


def fc(x, w, b):
    # TO DO
    # print("shape of x = ", np.shape(x))
    # print("shape of w = ", np.shape(w))
    # print("shape of b = ", np.shape(b))
    # print("shape of x transpose ", np.shape(x.T))
    x = x.reshape(-1,1)
    # w = w.reshape((np.max(b.shape),np.max(w.shape)))
    # x = x.reshape((np.max(x.shape),1))
    # b = b.reshape((np.max(b.shape),1))


    # a = (w @ x).reshape(10,1)
    a = (w @ x)
    # print("shape of a = " , np.shape(a.reshape(-1,1)))
    y = a + b


    # y = y.reshape(1,-1)
    # print("shape of a = ", np.shape(a))
    return y


def fc_backward(dl_dy, x, w, b, y):

    # TO DO

    dl_db = dl_dy

    dl_dx = np.dot(w.T,dl_dy)

    # dl_dw = np.dot(x,dl_dy)
    dl_dw = x.T * dl_dy
    # dl_dx = 
    return dl_dx, dl_dw, dl_db



def loss_euclidean(y_tilde, y):
    # TO DO
    l_root = np.linalg.norm((y_tilde - y),'fro')
    l = l_root ** 2
    dl_dy = 2*(y_tilde-y)
    return l, dl_dy


def loss_cross_entropy_softmax(x, y):
    # TO DO
    exp = np.exp(x)
    y_tilde = exp/np.sum(exp)
    y_tilde = np.array(y_tilde)
    y_tilde_log = np.log(y_tilde + 0.0000001)
    x = np.array(x)
    y = np.array(y)
    l = -1 * np.sum( np.multiply(y,y_tilde_log) )

    dl_dy = y_tilde - y



    return l, dl_dy


def relu(x):
    # TO DO
    # shape = np.shape(x)
    # x = x.reshape((-1,1))

    # y = np.maximum(0,x)
    y = np.maximum(0.1*x,x) # leaky relu

    # y = y.reshape(shape)
    return y


def relu_backward(dl_dy, x, y):
    # TO DO
    # print("backward relu madhe x cha shape = ", np.shape(x.ravel()))
    # # y = x.flatten()
    # print("backward relu madhe y cha shape = ", np.shape(y))
    # y[y<0] = 0
    # y[y>1] = dl_dy

    # dy_dx = np.where(x > 0,1,0)
    dy_dx = np.where(x > 0,1,0.1) #leaky relu
    dl_dx = dl_dy * dy_dx

    # for i in 
    # # if x < 0:
    # #     dl_dx = 0
    # # else:
    # #     dl_dx = dl_dy
    return dl_dx





def filter_image(im, filter): #from homework one
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


def filter_image_for_conv_back(im, filter): #from homework one
    im_height,im_width = np.shape(im)
    im_filtered = np.zeros((3,3))

    # pad_margin = len(filter)//2

    pad_margin = 1
    im_padded = np.pad(im, [(pad_margin, pad_margin), (pad_margin, pad_margin)], mode='constant', constant_values=0)

    for up_down in range(3):
        for left_right in range(3):

            # im_filtered[i][j] = filter[0][0]*im_padded[i-1,j-1] + filter[0][1]*im_padded[i-1,j] + filter[0][2]*im_padded[i-1,j+1] + filter[1][0]*im_padded[i,j-1] + filter[1][1]*im_padded[i,j] + filter[1][2]*im_padded[i,j+1] + filter[2][0]*im_padded[i+1,j-1] + filter[2][1]*im_padded[i+1,j] + filter[2][2]*im_padded[i+1,j+1]
            im_filtered[up_down,left_right] = np.dot(filter.flatten(order='C'),im_padded[up_down:up_down+len(filter),left_right:left_right+len(filter)].flatten(order='C'))
    # To do
    return im_filtered

def conv(x, w_conv, b_conv):
    # TO DO
    # print("convolution la input cha shape", np.shape(x) )

    x = x.reshape((14,14))
    y = np.zeros((14,14,3))
    for channel in range(3):

        kernel_conv = w_conv[:,:,0,channel]
        # print("conv wala kernel OG:", kernel_conv)
        kernel_conv = kernel_conv.reshape((3,3) )

        # print("filter wala kernel rehshaped :", kernel_conv)
        kernel_filtering = kernel_conv[::-1,::-1]
        # print("filter wala kernel :", kernel_filtering)
        # print("shape of b = ", b_conv.shape)
        y[:,:,channel] = filter_image(x,kernel_filtering) + b_conv[:,channel]
        # y[:,:,channel] = filter_image(x,kernel_filtering)




    
    return y



def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    # print("shape of x = ", np.shape(x))
    x = x.reshape((14,14))

    # dl_db = dl_dy   
    # print("dl_dy cha shape = ", np.shape(dl_dy))

    dl_dw = np.zeros_like(w_conv)
    dl_db = np.zeros_like(b_conv)

    # print("dl_dw cha shape = ", np.shape(dl_dw))

    for channel in range(3):
        # dl_dw[:,:,channel] = conv(x,dl_dy[:,:,1],0)
        conv_kernel = dl_dy[:,:,channel]
        conv_kernel = conv_kernel.reshape((14,14))
        filter_kernel = conv_kernel[::-1,::-1]

        dl_dw[:,:,0,channel] = filter_image_for_conv_back(x,filter_kernel)
        dl_db[:,channel] = np.sum(filter_kernel)


    return dl_dw, dl_db








def pool_2x2_2d(x):
    a = int(len(x)/2)
    b = int(len(x[0])/2)
    y = np.zeros((a,b))

    for i in range(a):
        for j in range(b):
            y[i,j] = np.max([ x[2*i][2*j],x[2*i][2*j+1],x[2*i+1][2*j],x[2*i+1][2*j+1]   ])
    return y 



def pool2x2(x):
    # TO DO
    shape = np.shape(x)
    y = np.zeros((int(shape[0]/2),int(shape[1]/2),shape[2]))

    for k in range(3):
        y[:,:,k] = pool_2x2_2d(x[:,:,k])
    
    return y



def pool2x2_backward(dl_dy, x, y):
    # TO DO
    # do we apply gradient only to the mx element or the entire block?
    arg_maxs = []
    dl_dx = np.zeros_like(x)
    # print("shape of x is :", np.shape(x))
    for k in range(3):
        for i in range(len(y)):
            for j in range(len(y[0])):
                i_add = 0
                j_add = 0
                points_to_consider = np.array([ x[2*i][2*j][k],x[2*i][2*j+1][k],x[2*i+1][2*j][k],x[2*i+1][2*j+1][k] ])
                max_arg = np.argmax(points_to_consider)
                # print("argmax = ", max_arg)
                if max_arg > 1:
                    i_add = 1
                if (max_arg % 2) == 1:
                    j_add = 1 

                dl_dx[2*i + i_add][2*j + j_add][k] += dl_dy[i][j][k]
                # arg_maxs.append(max_arg)
    # print("max args = ", arg_maxs)

    # dl_dx = np.repeat(dl_dy,2,axis=1)
    # dl_dx = np.repeat(dl_dx,2,axis=0)

    return dl_dx


##### Pool backwards wih gradient applied to all values of the 2X2 block
# def pool2x2_backward(dl_dy, x, y):
#     # TO DO
#     # do we apply gradient only to the mx element or the entire block?
    
#     dl_dx = np.repeat(dl_dy,2,axis=1)
#     dl_dx = np.repeat(dl_dx,2,axis=0)

#     return dl_dx








def flattening(x):
    # TO DO
    x = np.asarray(x)
    l = (np.size(x),1)
    y = x.reshape(l)

    return y


def flattening_backward(dl_dy, x, y):
    l = np.shape(x)
    dl_dx = dl_dy.reshape(l)
    # TO DO
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    # TO DO
    # print("mini batch y cha shape = ", len(mini_batch_y),len(mini_batch_y[0]),len(mini_batch_y[0][0]) )
    # print("mini batch x cha shape = ", len(mini_batch_x),len(mini_batch_x[0]),len(mini_batch_x[0][0]) )

    lr = 0.0005 #learning rate
    
    dr = 0.9 #decay rate

    number_of_batches = len(mini_batch_x)
    batch_size = len(mini_batch_x[0].T)


    w_size = (10,196)
    b_size = (10,1)

    

    w = np.random.normal(0,0.1,w_size)
    b = np.random.normal(0,0.1,b_size)

    number_of_iterations = 2000

    iterations = range(number_of_iterations)

    for iteration in iterations:
        if iteration%1000 == 0:
            lr = lr * dr

        dl_dw = np.zeros(w_size)
        dl_db = np.zeros(b_size)
        
        mini_batch_number = iteration % number_of_batches
        imgs = mini_batch_x[mini_batch_number].T
        # imgs = mini_batch_x[mini_batch_number].reshape(196,1)
        lbls = mini_batch_y[mini_batch_number].T
        for index in range(len(mini_batch_x[mini_batch_number].T)):
        # for img in mini_batch_x[mini_batch_number].T:
            img = imgs[index]

            # print("length of image= ",len(img))

            prediction_y = fc(img.T,w,b)
            
            # print("dimestions of prediction_y  = ", np.shape(prediction_y)) 

            # print("shape of y prediction = ",np.shape(prediction_y))

            # print("mini batch number cha shape = ", np.shape(mini_batch_y[mini_batch_number]))
            # print("actual label cha shape = ", np.shape(lbls[index]))
            loss,dl_dy = loss_euclidean(prediction_y,lbls[index].reshape(10,1))

            # dl_dy = 

            # print("dimestions of loss = ", np.shape(loss))

            dl_dx1, dl_dw1, dl_db1 = fc_backward(dl_dy,img.T,w,b,mini_batch_y)

            dl_dw = dl_dw + dl_dw1
            dl_db = dl_db + dl_db1

        # print("batch changed ######################## iteration number = ", iteration," done ")

        w = w - (lr/batch_size) * dl_dw
        b = b - (lr/batch_size) * dl_db


    return w, b



# def train_slp(mini_batch_x, mini_batch_y,lr,dr):
def train_slp(mini_batch_x, mini_batch_y):
    # TO DO
    # lr = 0.001 #learning rate
    
    # dr = 0.95 #decay rate
    lr = 0.0098
    dr = 0.995
    number_of_batches = len(mini_batch_x)
    batch_size = len(mini_batch_x[0].T)

    w_size = (10,196)
    b_size = (10,1)

    # 0.00979999999999999 0.9900000000000001 0.8355
    

    w = np.random.normal(0,0.1,w_size)
    b = np.random.normal(0,0.1,b_size)
    #2000 -> 82%
    #3000 -> 83%
    #4000 -> 84.9%
    #4500 -> 85.something%
    #5000 -> 85.8%
    #6000
    number_of_iterations = 5500 # best so far

    iterations = range(number_of_iterations)

    for iteration in iterations:
        if iteration%1000 == 0:
            lr = lr * dr

        dl_dw = np.zeros(w_size)
        dl_db = np.zeros(b_size)
        
        mini_batch_number = iteration % number_of_batches
        imgs = mini_batch_x[mini_batch_number].T
        # imgs = mini_batch_x[mini_batch_number].reshape(196,1)
        lbls = mini_batch_y[mini_batch_number].T
        for index in range(len(mini_batch_x[mini_batch_number].T)):
        # for img in mini_batch_x[mini_batch_number].T:
            img = imgs[index]

            # print("length of image= ",len(img))

            prediction_y = fc(img.T,w,b)
            
            # print("dimestions of prediction_y  = ", np.shape(prediction_y)) 

            # print("shape of y prediction = ",np.shape(prediction_y))

            # print("mini batch number cha shape = ", np.shape(mini_batch_y[mini_batch_number]))
            # print("actual label cha shape = ", np.shape(lbls[index]))
            loss,dl_dy = loss_cross_entropy_softmax(prediction_y,lbls[index].reshape(10,1))

            # dl_dy = 

            # print("dimestions of loss = ", np.shape(loss))

            dl_dx1, dl_dw1, dl_db1 = fc_backward(dl_dy,img.T,w,b,mini_batch_y)

            dl_dw = dl_dw + dl_dw1
            dl_db = dl_db + dl_db1

        # print("batch changed ######################## iteration number = ", iteration," done ")

        w = w - (lr/batch_size) * dl_dw
        b = b - (lr/batch_size) * dl_db

    return w, b



def train_mlp(mini_batch_x, mini_batch_y,lr = 0.012,num_iterations = 14500):
    # TO DO

    
    # dr = 0.95 #decay rate
    # lr = 0.0065
    dr = 0.99
    number_of_batches = len(mini_batch_x)
    batch_size = len(mini_batch_x[0].T)

    w1_size = (30,196)
    b1_size = (30,1)
    w2_size = (10,30)
    b2_size = (10,1)

    # best lr,iterations, accuracy =  0.012 14500 90.85
    

    w1 = np.random.normal(0,0.1,w1_size)
    b1 = np.random.normal(0,0.1,b1_size)


    w2 = np.random.normal(0,0.1,w2_size)
    b2 = np.random.normal(0,0.1,b2_size)
    #2000 -> 82%
    #3000 -> 83%
    #4000 -> 84.9%
    #4500 -> 85.something%
    #5000 -> 85.8%
    #6000
    # number_of_iterations = 12200 # best so far
    number_of_iterations = num_iterations

    iterations = range(number_of_iterations)

    for iteration in iterations:
        if iteration%1000 == 0:
            lr = lr * dr

        dl_dw1_total = np.zeros(w1_size)
        dl_db1_total = np.zeros(b1_size)
        dl_dw2_total = np.zeros(w2_size)
        dl_db2_total = np.zeros(b2_size)
        
        mini_batch_number = iteration % number_of_batches
        imgs = mini_batch_x[mini_batch_number].T
        # imgs = mini_batch_x[mini_batch_number].reshape(196,1)
        lbls = mini_batch_y[mini_batch_number].T
        for index in range(len(mini_batch_x[mini_batch_number].T)):
        # for img in mini_batch_x[mini_batch_number].T:
            img = imgs[index]

            # print("length of image= ",len(img))

            a1 = fc(img.T,w1,b1)
            z1 = relu(a1)

            prediction_y = fc(z1.T,w2,b2)
            
            # print("dimestions of prediction_y  = ", np.shape(prediction_y)) 

            # print("shape of y prediction = ",np.shape(prediction_y))

            # print("mini batch number cha shape = ", np.shape(mini_batch_y[mini_batch_number]))
            # print("actual label cha shape = ", np.shape(lbls[index]))
            loss,dl_dy = loss_cross_entropy_softmax(prediction_y,lbls[index].reshape(10,1))

            # dl_dy = 

            # print("dimestions of loss = ", np.shape(loss))

            dl_df_1, dl_dw2, dl_db2 = fc_backward(dl_dy, z1, w2, b2, mini_batch_y)
            # print("ek forward zhala")
            dl_da_1 = relu_backward(dl_df_1, a1, z1)
            # print("relu zhala")
            dl_dx, dl_dw1, dl_db1 = fc_backward(dl_da_1, img.T, w1, b1, a1)
            # print("ek backward zhala")
            # dl_dx2, dl_dw2, dl_db2 = fc_backward(dl_dy,z1.T,w2,b2,mini_batch_y)

            # dl_dx1, dl_dw1, dl_db1 = fc_backward(dl_dy,img.T,w,b,mini_batch_y)

            

            dl_dw1_total = dl_dw1_total + dl_dw1
            dl_db1_total = dl_db1_total + dl_db1
            dl_dw2_total = dl_dw2_total + dl_dw2
            dl_db2_total = dl_db2_total + dl_db2
            # dl_db = dl_db + dl_db1
            # dl_dw = dl_dw + dl_dw1
            # dl_db = dl_db + dl_db1

        # print("batch changed ######################## iteration number = ", iteration," done ")

        w1 = w1 - (lr/batch_size) * dl_dw1_total
        b1 = b1 - (lr/batch_size) * dl_db1_total
        w2 = w2 - (lr/batch_size) * dl_dw2_total
        b2 = b2 - (lr/batch_size) * dl_db2_total

    # return w, b

    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y,lr = 0.75,dr = 0.9, iters = 9900):
    # print("shape of mini batch x = ",np.shape(mini_batch_x))
    number_of_batches = len(mini_batch_x)
    # number_of_batches = len(mini_batch_x)
    batch_size = len(mini_batch_x[0].T)


    wc_size = (3,3,1,3)
    bc_size = (1,3)
    wf_size = (10,147)
    bf_size = (10,1)

    w_conv = np.random.normal(0,0.1,wc_size)
    b_conv = np.random.normal(0, 0.1,bc_size)
    w_fc = np.random.normal(0, 0.1,wf_size)
    b_fc = np.random.normal(0, 0.1,bf_size)

    # lr = 0.075 #0.01
    # dr = 0.99

    #  lr = 0.075
    # dr = 0.9 -> 90.1%

    number_of_iterations = iters

    iterations = range(number_of_iterations)
    loss_train = []
    for iteration in iterations:



        if iteration%1000 == 0:
            lr = lr * dr

        dl_dwc_total = np.zeros(wc_size)
        dl_dbc_total = np.zeros(bc_size)
        dl_dwf_total = np.zeros(wf_size)
        dl_dbf_total = np.zeros(bf_size)
        
        mini_batch_number = iteration % number_of_batches
        imgs = mini_batch_x[mini_batch_number].T
        # imgs = mini_batch_x[mini_batch_number].reshape(196,1)
        lbls = mini_batch_y[mini_batch_number].T
        for index in range(len(mini_batch_x[mini_batch_number].T)):
        # for img in mini_batch_x[mini_batch_number].T:
            img = imgs[index]
            img = img.reshape((14,14))
            # print("length of image= ",len(img))
            
            a1 = conv(img.T,w_conv,b_conv)
            z1 = relu(a1)
            # print("shape of pre pooling = ",np.shape(z1))
            p1 = pool2x2(z1)
            # print("shape of pre pooling = ",np.shape(z1))
            # print("shape of pooled p1 = ", np.shape(p1))

            a2 = flattening(p1)


            prediction_y = fc(a2.T,w_fc,b_fc)
            
            # print("dimestions of prediction_y  = ", np.shape(prediction_y)) 

            # print("shape of y prediction = ",np.shape(prediction_y))

            # print("mini batch number cha shape = ", np.shape(mini_batch_y[mini_batch_number]))
            # print("actual label cha shape = ", np.shape(lbls[index]))
            loss,dl_dy = loss_cross_entropy_softmax(prediction_y,lbls[index].reshape(10,1))
            loss_train.append(loss)
            
            # dl_dy = 

            # print("dimestions of loss = ", np.shape(loss))


            dl_df_1, dl_dwf, dl_dbf = fc_backward(dl_dy, a2, w_fc, b_fc, mini_batch_y)

            dl_df_c = flattening_backward(dl_df_1, p1, a2)
            # print("ek forward zhala")
            dl_dp = pool2x2_backward(dl_df_c,z1,p1)


            dl_da_1 = relu_backward(dl_dp, a1, z1)
            # print("relu zhala")
            # dl_dx, dl_dw1, dl_db1 = fc_backward(dl_da_1, img.T, w1, b1, a1)
            # print("ek backward zhala")
            # dl_dx2, dl_dw2, dl_db2 = fc_backward(dl_dy,z1.T,w2,b2,mini_batch_y)
            # print("conv_back cha input cha shape : ", np.shape(dl_da_1))
            # dl_dx1, dl_dw1, dl_db1 = fc_backward(dl_dy,img.T,w,b,mini_batch_y)
            dl_dwc,dl_dbc = conv_backward(dl_da_1,img,w_conv,b_conv,a1)
            

            dl_dwc_total = dl_dwc_total + dl_dwc
            dl_dbc_total = dl_dbc_total + dl_dbc
            dl_dwf_total = dl_dwf_total + dl_dwf
            dl_dbf_total = dl_dbf_total + dl_dbf
            # dl_db = dl_db + dl_db1
            # dl_dw = dl_dw + dl_dw1
            # dl_db = dl_db + dl_db1

        # print("batch changed ######################## iteration number = ", iteration," done ")

        w_fc = w_fc - lr * dl_dwf_total/batch_size
        b_fc = b_fc - lr * dl_dbf_total/batch_size
        w_conv = w_conv - lr * dl_dwc_total/batch_size
        b_conv = b_conv - lr * dl_dbc_total/batch_size

        if iteration%300 == 0:
            print("Finnished iteration number ", iteration)

        



    # TO DO
    return w_conv, b_conv, w_fc, b_fc






if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()



    # k = np.array([1,2,3,4,5,6,7,8])
    # k = k.T
    # k_r = k.reshape((4,2))
    # print("reshaped ", k_r)