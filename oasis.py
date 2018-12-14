##This file recreate the method meationed in OASIS with some change


# feature_location  = "visterms_1000.market"
data_location  = "visterms_1000.mat"
agress = 0.1
reseed = 1
#6,5
num_steps= int(10e6)
num_perstep = int(10e5)
do_sym = False
sym_every = True


# loss_steps = np.zeros((1, num_steps))


from scipy.io import mmread
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
import numpy as np

#define gloabal variable for easy use
class OASIS:
    def __init__(self, data_file_source):
        # Reading data
        print("Start reading data！")
        #######no longer necessary ####
        # print("Reading Image Feature!")
        # Data = mmread(image_feature_file_source)
        # self.Full_Data = Data.todense()
        ###############################
        #reading from mat
        print("Reading Iamge imforation from mat")
        Label = loadmat(data_file_source)
        self.filenames = Label['filenames']
        self.images= Label['images']
        self.num_objects, self.num_feature = self.images.shape
        self.label_names = Label['label_names']
        self.label_serials = Label['label_serials']
        print("Finished reading")
        ###########################################
        #initialize W
        self.W = np.eye(self.num_feature, dtype = 'float64')
        #organize the data
        inds = self.label_serials.argsort()
        self.label_serials[0] = self.label_serials[0][inds] - 1
        self.label_names = self.label_names[inds]
        self.classes = np.unique(self.label_serials)
        self.num_class = np.size(self.classes)
        self.class_sizes = np.zeros((self.num_class,1))
        self.class_start = np.zeros((self.num_class,1))
        for k in self.classes:
            self.class_sizes[k] = np.sum(self.label_serials == k)
            temp1, temp2  = np.nonzero(self.label_serials == k)
            self.class_start[k] = temp2[0]
        print("Initialization Finshed")

    def process(self):
        num_bash = int(np.ceil(num_steps / num_perstep))
        for i in range(num_bash):
            print("Current num_bash" + str(i) + 'start')
            self.process_bash()
            print("Current num_bash" + str(i) + 'end')
            print('')


    def process_bash(self):
        for i in range(num_perstep):
            if(i % (num_perstep/100)  == 0):
                print('.')
                #select a random picture
            pic = int(np.floor(np.random.random() * self.num_objects))
            class_temp = self.label_serials[0][pic]
            pic_pos = int(self.class_start[class_temp] + np.floor(np.random.random() * self.class_sizes[class_temp]))
            pic_neg = int(np.floor(np.random.random() * self.num_objects))
            while(self.label_serials[0][pic_neg] == class_temp):
                pic_neg = int(np.floor(np.random.random() * self.num_objects))
            ###check should be delete latter!!!!
            # if( self.label_serials[0][pic_pos] == self.label_serials[0][pic] and 
            # self.label_serials[0][pic_neg] != self.label_serials[0][pic]):
            #     print('Correct')
            # else:
            #     print('False')
            delta = self.images[pic_pos] - self.images[pic_neg]
            loss = 1- self.images[pic] @ self.W @ (delta.T)
            V = self.images[pic].T @  (delta)
            if loss > 0:
                norm_V = linalg.norm(V)
                tau_val = loss/norm_V

                tau = float(min(agress, tau_val))
                self.W = self.W + tau* V


if __name__ == '__main__':
    np.random.seed(0)
    print("Start processing")
    Process = OASIS(data_location)
    Process.process()
    print(Process.W)
    print("Main Finished")

    