# Multi GPU Training using DataParallel

Consider the demo.py code for this tutorial

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import time

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 400)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 200)
        self.fc4 = nn.Linear(200, 100)
        self.fc5 = nn.Linear(100, 50)
        self.fc6 = nn.Linear(50, output_size)

    def forward(self, input):
        x = self.sigmoid(self.fc1(input))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return self.sigmoid(self.fc6(x))


if __name__ == '__main__':
    # Parameters and DataLoaders
    input_size = 1000
    output_size = 1
    batch_size = 2048
    data_size = 1000000

    model = Model(input_size, output_size)
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).cuda()

    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size), batch_size=batch_size, shuffle=True)

    optimizer = optim.SGD(params=model.parameters(), lr=1e-3)
    cls_criterion = nn.BCELoss()
    print("Starting training loop")
    start = time.perf_counter()
    for data in rand_loader:
        targets = torch.empty(data.size(0)).random_(2).view(-1, 1)

        if torch.cuda.is_available():
            input = Variable(data.cuda())
            with torch.no_grad():
                targets = Variable(targets.cuda())
        else:
            input = Variable(data)
            with torch.no_grad():
                targets = Variable(targets)

        output = model(input)

        optimizer.zero_grad()
        loss = cls_criterion(output, targets)
        loss.backward()
        optimizer.step()
    end = time.perf_counter()
    print("Time taken: ", end-start)
```



In this tutorial, we will run [DATAPARALLEL]([DataParallel &mdash; PyTorch 2.1 documentation](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)) in GPU training using pytorch dataparallel model 



![](C:\Users\mehta\AppData\Roaming\marktext\images\2023-11-06-19-09-31-image.png)



Make sure that, the inputs to the function DataParallel are

1. model

2. Make sure the inputs to the model are converted to device that you are going to be using, (in our case it is going to be CUDA)



Let's have a quick demo on NYU HPC as well for better understanding:

- The given code `demo.py` shows that:
  
  - class RandomDataset() function generates random dataset of configurable input size
  
  - We have model that does mapping

- If you see in the code, we have a model that checks if we have more than 1 GPUs then print the number of GPUs that are available and then there is function of DataParallel, which wraps the model, the code section that I was talking about was this:

```python
model = Model(input_size, output_size)
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).cuda()
```



- Now let's see how to run this demo.py using sbatch file

```shell
#!/bin/bash
#SBATCH --job-name=Multi-GPUDemo-1GP-misc
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --output=%x.result
#SBATCH --mem=20GB
#SBATCH --time=00:30:00

module load python/intel/3.8.6
module load anaconda3/2020.07

cd /scratch/cmn8525/HPML/Assignments/demo/
eval "$(conda shell.bash hook)"
conda activate ../../../hpml/

echo "Demo - Multi GPU Execution"
python demo.py
```

Two attributes that we need to define over here:

1. GRES attribute: It is used to describe about the number of GPUs that we are going to use

2. The partition attribute: This attribute is important because it is going to define, you want to perform comparative tests on CPU/ GPU. It allows you to specify which type of GPU you want to use



Further attributes:

3. CPUs per task: This attribute is related to cpu related tasks like prep in dataloading, etc.
