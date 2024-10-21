import torch
import os
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from models.resnet import ResNet
from utils.read_data import get_data
from utils.read_data import get_dataloader

from torch.utils.tensorboard import SummaryWriter

from utils.cli import get_parser

parser = get_parser()
args = parser.parse_args()
batch = 64

X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor = get_data()
train_data_loader,val_data_loader = get_dataloader(X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor,batch)

grids = [args.grid]
writer = SummaryWriter('runs/mnist_experiment_1')

if torch.cuda.is_available():
    model = ResNet().cuda()
else:
    model = ResNet()
criterion = nn.MSELoss()
learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(), lr = learning_rate) #lr

# Training

num_epochs = 1000
for epoch in range(num_epochs):
    for images, labels in train_data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        al_label = labels.unsqueeze(1)
        loss = criterion(outputs, al_label)
        loss.backward()
        optimizer.step()
        writer.add_scalar('training loss', loss, epoch)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


# saving the model
os.makedirs("trained_models",exist_ok=True)
torch.save(model.state_dict(), f'trained_models/deep_resnet_{grids[0]}mm.pth')

print(f"Model saved in {args.project_root}/trained_models/deep_resnet__{grids[0]}mm.pth!")

writer.close()
    # Validation on training performance
        
model.eval()
pre = model(X_train_tensor)
pre = pre.cpu().detach().numpy()
y_train_tensor = y_train_tensor.cpu().detach().numpy()

mae = mean_absolute_error(y_train_tensor,pre)
mse = mean_squared_error(y_train_tensor,pre)
rmse = mean_squared_error(y_train_tensor,pre,squared=False)
r2=r2_score(y_train_tensor,pre)

print("Training Performance:")
print("MAE",mae)
print("MSE",mse)
print("RMSE",rmse)
print("R2",r2)        