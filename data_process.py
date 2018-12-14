import numpy as np
from scipy.io import mmread
from scipy.io import loadmat
from scipy.io import savemat
from scipy.sparse import csr_matrix
data_location  = "visterms_1000.mat"



class Process:
    def read(self, data_file_source):
        Label = loadmat(data_file_source)
        self.filenames = Label['filenames']
        self.images= Label['images']
        self.num_objects, self.num_feature = self.images.shape
        self.label_names = Label['label_names']
        self.label_serials = Label['label_serials']
        inds = self.label_serials.argsort()
        self.label_serials[0] = self.label_serials[0][inds] - 1
        self.label_names = self.label_names[inds]
        self.filenames = self.filenames[inds]
        self.classes = np.unique(self.label_serials)
        self.num_class = np.size(self.classes)
        self.class_sizes = np.zeros((self.num_class,1))
        self.class_start = np.zeros((self.num_class,1))
        for k in self.classes:
            self.class_sizes[k] = np.sum(self.label_serials == k)
            temp1, temp2  = np.nonzero(self.label_serials == k)
            self.class_start[k] = temp2[0]

    def write(self,nums, test_name, train_name):
        if (nums > self.num_class):
            print("Failed.")
        else:
            newclass = int(np.floor((np.random.random()* self.num_class)))
            class_visited = []
            class_visited.append(newclass)
            while len(class_visited) < nums:
                newclass = int(np.floor((np.random.random()* self.num_class)))
                while newclass in class_visited:
                    newclass = int(np.floor((np.random.random()* self.num_class)))
                class_visited.append(newclass)
            ##get the training data and the test data
            test = np.array([], dtype = int)
            train = np.array([],dtype = int)
            for i in class_visited:
                ind = np.array([j + int(self.class_start[i]) for j in range(int(self.class_sizes[i]))])
                np.random.shuffle(ind)
                ind_train = [j for j in range(int(np.floor(0.8* self.class_sizes[i])))]
                ind_test = [j for j in range(int(np.floor(0.8* self.class_sizes[i])) , int(self.class_sizes[i]))]
                ind_test = ind[ind_test]
                ind_train = ind[ind_train]
                test = np.append(test, ind_test)
                train = np.append(train, ind_train)
            self.filenames = np.array(self.filenames)            
            self.label_names = np.array(self.label_names)
            self.label_serials = np.array(self.label_serials)
            savemat(test_name, mdict={'filenames': self.filenames[0][test],'images' : self.images[test], 'label_names': self.label_names[0][test], 'label_serials' : self.label_serials[0][test]})
            savemat(train_name, mdict={'filenames': self.filenames[0][train],'images' : self.images[train], 'label_names': self.label_names[0][train], 'label_serials' : self.label_serials[0][train]})

if __name__ == '__main__':

    for nn in [5]:
        for idx in [1, 2, 3]:
            save_name_test = 'test'+str(nn)+'_'+str(idx);
            save_name_train = 'train'+str(nn)+'_'+str(idx);
            np.random.seed(idx+nn)
            print("Start processing")
            A = Process()
            A.read(data_location)
            print(A.label_names.shape)
            print(A.filenames.shape)
            A.write(nn,'newcompress/'+save_name_test, 'newcompress/'+save_name_train)
            print("Main Finished")