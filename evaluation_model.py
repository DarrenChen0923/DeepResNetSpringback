from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import torch
from utils.cli import get_parser
from utils.read_data import get_data
from utils.read_data import get_dataloader
from models.resnet import ResNet

# cml arguments
parser = get_parser()
args = parser.parse_args()

## Parameters 
version = 2
grids = [args.grid]
num_epochs = 1000
batch = 64
learning_rate = 0.001
model_path =  args.project_root + f'/trained_models/{args.load_model}'
degrees = [0,90,180,270]
fums = [1,2,3]

batch = 64

X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor = get_data()
train_data_loader,val_data_loader = get_dataloader(X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor,batch)

if torch.cuda.is_available():
    model = ResNet().cuda()
else:
    model = ResNet()
model.load_state_dict(torch.load(model_path))
model.eval()
pre = model(X_test_tensor)

if torch.cuda.is_available():
    pre = pre.cpu().detach().numpy()
else:
    pre = pre.detach().numpy()


y_test_tensor = y_test_tensor.cpu()

mae = mean_absolute_error(y_test_tensor,pre)
mse = mean_squared_error(y_test_tensor,pre)
rmse = mean_squared_error(y_test_tensor,pre,squared=False)
r2=r2_score(y_test_tensor,pre)

print("MAE",mae)
print("MSE",mse)
print("RMSE",rmse)
print("R2",r2)