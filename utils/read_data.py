import numpy as np
import torch
import os
from PIL import Image
import torch.utils.data as Data
from utils.cli import get_parser
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/mnist_experiment_1')

# cml arguments
parser = get_parser()
args = parser.parse_args()

degrees = [0,90,180,270]
fums = [1,2,3]
grids = [args.grid]
version = 2

# cml arguments
parser = get_parser()
args = parser.parse_args()
grids = [args.grid]

def numeric_sort(file_name):
    file_name_parts = file_name.split('/')[-1].split('_')
    file_name_parts[-1] = file_name_parts[-1].split('.')[0]  
    return int(file_name_parts[0]), int(file_name_parts[1])

def read_data(train_or_test):
        # set Image file and label file path
    image_folder = args.project_root + f'/data/{train_or_test}_dataset/{grids[0]}mm/images'
    label_folder = args.project_root + f'/data/{train_or_test}_dataset/{grids[0]}mm/labels'
    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.jpg')]

    # Create empty list to store iamges and lables
    X = []  # store images
    y = []  # store lables

    # Iterate Images
    for image_path in image_files:

        # Get images file name
        image_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Build path for label file
        label_path = os.path.join(label_folder, f'{image_filename}.txt')

        # Read label file
        with open(label_path, 'r') as label_file:
            label = label_file.read().strip() 

        # Image preprocessing
        with Image.open(image_path) as img:
            img = np.array(img) 

        img = img.transpose((2, 0, 1))

        # put image and label into list
        X.append(img)
        y.append(label)

    return X,y


def get_data():
    
    X_train,y_train = read_data("train")
    X_test,y_test = read_data("test")

    # transfer X and y into numpy array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Normalise
    X_train = X_train / 255.0
    X_test = X_test/255.0

    def normalize_mean_std(image):
        mean = np.mean(image)
        stddev = np.std(image)
        normalized_image = (image - mean) / stddev
        return normalized_image

    X_train = normalize_mean_std(X_train)
    X_test = normalize_mean_std(X_test)


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')


    # #tensor to numpy
    X_train_tensor = torch.from_numpy(X_train)
    X_test_tensor = torch.from_numpy(X_test)
    y_train_tensor = torch.from_numpy(y_train)
    y_test_tensor = torch.from_numpy(y_test)

    #  #gpu environment: transfer into cuda
    if torch.cuda.is_available():
        X_train_tensor = X_train_tensor.cuda()
        X_test_tensor = X_test_tensor.cuda()
        y_train_tensor = y_train_tensor.cuda()
        y_test_tensor = y_test_tensor.cuda()
    
    return X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor

  
def get_dataloader(X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor,batch):
    
    # #Cominbe dataset
    train_dataset = Data.TensorDataset(X_train_tensor,y_train_tensor)
    val_dataset = Data.TensorDataset(X_test_tensor,y_test_tensor)

    # #Create dataset loader

    train_data_loader = Data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_data_loader = Data.DataLoader(val_dataset, batch_size=batch, shuffle=False)
    return train_data_loader,val_data_loader