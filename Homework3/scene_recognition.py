import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier # Allowed? confirm No
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath

from itertools import product # Allowed? confirm YES


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def compute_dsift(img,stride,size):

    # v = range(int(size/2),int((len(img)-size/2)), stride)
    # h = range(int(size/2),int((len(img[0])-size/2)), stride)
    v = range(int(size/2),int((len(img))), stride)
    h = range(int(size/2),int((len(img[0]))), stride)
    v = range(0,int((len(img))), stride)
    h = range(0,int((len(img[0]))), stride)
    # print("v :", v)
    # print("h :", h)
    
    # for vv in v:
    #     for hh in h:

    # all_x,all_y = np.meshgrid(v,h)

    # all_kps = np.vstack(all_x.flatten,all_y.flatten)
    sift = cv2.SIFT_create()
    
    # kp_locations = product(v,h).flatten()
    
    # x_location = kp_locations[]
    kp_input = []
    # for x in v:
    #     for y in h:
    #         kp_input.append([cv2.KeyPoint(x,y,size)])

    
    kp_input = [cv2.KeyPoint(x,y, size) for x in v for y in h]

    kp, dense_feature = sift.compute(img, kp_input)

    
    # kp_locations = 
    # key_points
    
    # dense_feature = np.vstack(dense_feature)
    dense_feature = dense_feature.flatten()
    # print("shape of one d_sift = ", dense_feature.shape())
    
    # To do
    return dense_feature
# test = np.array([[16,1,2,3],[2,1,2,3],[1,6,2,3],[1,1,10,3]])
# output_size = (2,2)
# # feature = cv2.resize(test,output_size)
# feature = test - np.mean(test)
# print("image : ",feature)

# feature = feature/np.linalg.norm(feature)
# print("image : ",feature)

    # To do

def get_tiny_image(img, output_size):
    # test = np.array([16,1,2,3],[2,1,2,3],[1,6,2,3],[1,1,10,3])
    feature = cv2.resize(img,output_size,interpolation = cv2.INTER_AREA) #Allowed? YES
    feature = feature - np.mean(feature) #0 mean
    # print("image : ",feature)
    feature = feature/np.linalg.norm(feature) #unit length

    # feature = feature/np.linalg.norm(test)
    # To do
    return feature.flatten()
# get_tiny_image


def predict_knn(feature_train, label_train, feature_test, k):
    
    

    # distances,indices = 
    
    # neigh = KNeighborsClassifier(k)
    # neigh.fit(feature_train, label_train)
    # label_test_pred = neigh.predict(feature_test) 
    # print("prediction of nearest neighbour = ", label_test_pred)
    # Not allowed

    nbrs = NearestNeighbors(n_neighbors=k).fit(feature_train,label_train)
    distances, indices = nbrs.kneighbors(feature_test)
    # print("indices and distance : ", distances,indices)

    indices = np.array(indices)
    # keepdims=True
    # mode_index = stats.mode(indices.T,keepdims=True)
    # label_train = np.array(label_train)
    # label_test_pred = label_train[mode_index[0]]
    # label_test_pred = label_test_pred.ravel()

    label_train = np.array(label_train)
    labels = label_train[indices]
    mode_label  = stats.mode(labels.T,keepdims=True)

    label_test_pred = mode_label[0].ravel()





    # print(label_test_pred[0:5:1])
    # print(len(label_train))
    # print(np.array(mode_index[0]).reshape(-1).shape)
    # label_train = np.array(label_train)
    # label_test_pred = label_train[np.array(mode_index[0]).ravel()]

    # label_test_pred = stats.mode(labels)



    


    # To do
    return label_test_pred
# labels = np.array([9,8,7,6,5,4,3,2,1,0])
# test_array = np.array([[7,2,7,2,7,3,4,5],[2,2,2,2,3,4,5,6]])
# mode = stats.mode(test_array.T, keepdims=True)
# print("mode : ",mode[0])
# indices = labels[mode[0]]
# print(indices)


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    output_size = (16,16)

    train_img_tiny_feature = []
    for train_img in img_train_list:
        img = cv2.imread(train_img)
        # print("dimention_of_image = ", img.shape())
        tiny_img = get_tiny_image(img,output_size)
        train_img_tiny_feature.append(tiny_img)
    
    test_img_tiny_feature = []
    for test_img in img_test_list:
        img = cv2.imread(test_img)
        tiny_img = get_tiny_image(img,output_size)
        test_img_tiny_feature.append(tiny_img)

    train_img_tiny_feature = np.array(train_img_tiny_feature)
    test_img_tiny_feature = np.array(test_img_tiny_feature)

    # for k in range(1,11,1):
    #     predicted_labels = predict_knn(train_img_tiny_feature,label_train_list,test_img_tiny_feature,k)

    k = 5
    predicted_labels = predict_knn(train_img_tiny_feature,label_train_list,test_img_tiny_feature,k)

    # confusion = np.zeros((len(label_classes),len(label_classes)))

    # sum = 0
    # correct = 0

    # for i in range(len(img_test_list)):
    #     a = label_classes.index(label_test_list[i])
    #     b = label_classes.index(predicted_labels[i])

        
        
    #     confusion[a,b] += 1
    #     sum += 1
    #     if a == b :
    #         correct += 1

    # accuracy = correct/sum

    # train_img_tiny_feature = np.array(train_img_tiny_feature)
    # test_img_tiny_feature = np.array(test_img_tiny_feature)
    confusion,accuracy = confusion_and_accuracy(label_test_list,predicted_labels,label_classes)
    print(k, accuracy)
    
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy

def confusion_and_accuracy(truth,prediction,label_classes):
    
    confusion = np.zeros((len(label_classes),len(label_classes)))

    sum = 0
    correct = 0

    for i in range(len(truth)):
        a = label_classes.index(truth[i])
        b = label_classes.index(prediction[i])

        
        
        confusion[a,b] += 1
        sum += 1
        if a == b:
            correct += 1

    accuracy = correct/sum
    return confusion,accuracy   


def build_visual_dictionary(dense_feature_list, dict_size):

    # dense_feature_list = np.vstack(dense_feature_list)
    
    kmeans = KMeans(dict_size, n_init=10, max_iter=200)
    kmeans.fit(dense_feature_list)


    print("len of cluster centers", len(kmeans.cluster_centers_))
    # To do
    return kmeans.cluster_centers_

def compute_bow(feature, vocab):
    nbrs = NearestNeighbors(n_neighbors=1).fit(vocab) #How many neighbours?
    distances, indices = nbrs.kneighbors(feature.reshape(1, -1))

    bow_feature = np.zeros_like(vocab[0])
    for i in indices:
        bow_feature[i] += 1

    bow_feature = bow_feature/np.linalg.norm(bow_feature)

    # To do
    return bow_feature

def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # STRIDE = 8
    # SIZE = 11
    # STRIDE = 20
    # SIZE = 20
    # STRIDE = 10
    # SIZE = 20
    STRIDE = 12
    SIZE = 20
    # STRIDE = 10
    # SIZE = 10
    minimum_size = (200,220)
    # build vocab using all training data
    train_img_dsift = []
    minimum_x = 1000
    minimum_y = 1000
    for train_img in img_train_list:
        img = cv2.imread(train_img)
        # print("GAdbad here ? : ", len(img))
        # print("GAdbad here ? : ",  len(img[0]))
        # print("GAdbad here ? : ", len(img[0][0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,minimum_size,interpolation = cv2.INTER_AREA)
        # if len(img) < minimum_x :
        #     minimum_x = len(img)
        # if len(img) < minimum_y :
        #     minimum_y = len(img[0])
        # print("shape of image : ", img.shape)
        dsift = compute_dsift(img,STRIDE,SIZE) ############
        # print("Dsift dimention = ", len(dsift))
        train_img_dsift.append(dsift)
    # print("minimum x = ", minimum_x)
    # print("minimum y = ", minimum_y)

    train_img_dsift = np.array(train_img_dsift)
    # print("shape of train_img_dsift : ", train_img_dsift.shape)
    visual_dictionary = build_visual_dictionary(train_img_dsift,dict_size=50)


    # find BOW of each sample in training
    train_img_bow = []
    for train_img in img_train_list:
        img = cv2.imread(train_img)
  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,minimum_size,interpolation = cv2.INTER_AREA)
        dsift = compute_dsift(img,STRIDE,SIZE)
        # train_img_dsift.append(dsift)
        
        img_bow = compute_bow(dsift,visual_dictionary)
        train_img_bow.append(img_bow)
        # print("train_img_bow shape = ", train_img_bow.shape())
        # train_img_bow.append(compute_bow(dsift,visual_dictionary))
    train_img_bow = np.array(train_img_bow)

    test_img_bow = []
    for test_img in img_test_list:
        img = cv2.imread(test_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,minimum_size,interpolation = cv2.INTER_AREA)
        if len(img) < minimum_x :
            minimum_x = len(img)
        if len(img) < minimum_y :
            minimum_y = len(img[0])
        dsift = compute_dsift(img,STRIDE,SIZE)
        # train_img_dsift.append(dsift)
        test_img_bow.append(compute_bow(dsift,visual_dictionary))
    print("minimum x = ", minimum_x)
    print("minimum y = ", minimum_y)
    test_img_bow = np.array(test_img_bow)
    # fill knn with training bow

    # nbrs = NearestNeighbors().fit(train_img_bow)
    # for k in range(5,27,2):
    #     prediction = predict_knn(train_img_bow,label_test_list,test_img_bow,k)
    # # # 
    #     confusion, accuracy = confusion_and_accuracy(label_test_list,prediction,label_classes)
    #     print(k,accuracy)
    # for k in range(17,21,1):
    #     prediction = predict_knn(train_img_bow,label_test_list,test_img_bow,k)
    # # # 
    #     confusion, accuracy = confusion_and_accuracy(label_test_list,prediction,label_classes)
    #     print(k,accuracy)
    k = 10
    prediction = predict_knn(train_img_bow,label_train_list,test_img_bow,k)
    # # k = 20 gives 39.33% accuracy
    # # k = 31 gives 38% accuracy
    # # k = 15 gives 32.7% accuracy
    # # # # 
    confusion, accuracy = confusion_and_accuracy(label_test_list,prediction,label_classes)
    
    print(accuracy)
    # To do
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy



def predict_svm(feature_train, label_train, feature_test, n_classes,c):
   
    svc = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C = c, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
    svc.fit(feature_train,label_train)

    label_test_pred = svc.predict(feature_test)

    # To do
    return label_test_pred

def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):

    STRIDE = 20
    SIZE = 20
    minimum_size = (200,220)

    train_img_dsift = []
    minimum_x = 1000
    minimum_y = 1000
    for train_img in img_train_list:
        img = cv2.imread(train_img)
        # print("GAdbad here ? : ", len(img))
        # print("GAdbad here ? : ",  len(img[0]))
        # print("GAdbad here ? : ", len(img[0][0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,minimum_size,interpolation = cv2.INTER_AREA)
        # if len(img) < minimum_x :
        #     minimum_x = len(img)
        # if len(img) < minimum_y :
        #     minimum_y = len(img[0])
        # print("shape of image : ", img.shape)
        dsift = compute_dsift(img,STRIDE,SIZE) ############
        # print("Dsift dimention = ", len(dsift))
        train_img_dsift.append(dsift)
    # print("minimum x = ", minimum_x)
    # print("minimum y = ", minimum_y)

    train_img_dsift = np.array(train_img_dsift)
    # print("shape of train_img_dsift : ", train_img_dsift.shape)
    visual_dictionary = build_visual_dictionary(train_img_dsift,dict_size=50)


    # find BOW of each sample in training
    train_img_bow = []
    for train_img in img_train_list:
        img = cv2.imread(train_img)
  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,minimum_size,interpolation = cv2.INTER_AREA)
        dsift = compute_dsift(img,STRIDE,SIZE)
        # train_img_dsift.append(dsift)
        img_bow = compute_bow(dsift,visual_dictionary)
        train_img_bow.append(img_bow)
        # print("train_img_bow shape = ", train_img_bow.shape())
        # train_img_bow.append(compute_bow(dsift,visual_dictionary))
    train_img_bow = np.array(train_img_bow)

    test_img_bow = []
    for test_img in img_test_list:
        img = cv2.imread(test_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,minimum_size,interpolation = cv2.INTER_AREA)
        if len(img) < minimum_x :
            minimum_x = len(img)
        if len(img) < minimum_y :
            minimum_y = len(img[0])
        dsift = compute_dsift(img,STRIDE,SIZE)
        # train_img_dsift.append(dsift)
        test_img_bow.append(compute_bow(dsift,visual_dictionary))
    print("minimum x = ", minimum_x)
    print("minimum y = ", minimum_y)
    test_img_bow = np.array(test_img_bow)
    # fill knn with training bow
    # for c in np.arange(0.1,10,0.1):
    #     prediction = predict_svm(train_img_bow,label_test_list,test_img_bow,15,c)

    #     confusion, accuracy = confusion_and_accuracy(label_test_list,prediction,label_classes)
    #     print(c, accuracy)
    # To do
    c = 1.1
    prediction = predict_svm(train_img_bow,label_train_list,test_img_bow,15,c)
    confusion, accuracy = confusion_and_accuracy(label_test_list,prediction,label_classes)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # To do: replace with your dataset path
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")
    
    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    
    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)




