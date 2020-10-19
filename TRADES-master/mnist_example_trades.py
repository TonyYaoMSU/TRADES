from __future__ import print_function  # python2 python3 print
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from trades import trades_loss

# define a network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)     # 
        self.conv2 = nn.Conv2d(20, 50, 5, 1)    # 
        self.fc1 = nn.Linear(4 * 4 * 50, 500)  # 
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 28*28 -> 24*24
        x = F.max_pool2d(x, 2, 2) # 24*24 -> 12*12
        x = F.relu(self.conv2(x)) # 12*12 -> 8*8
        x = F.max_pool2d(x, 2, 2) # 8*8 -> 4*4
        x = x.view(-1, 4 * 4 * 50) # sample * flatten
        x = F.relu(self.fc1(x)) # first fc
        x = self.fc2(x) # second fc
        return x

# define train process

def train(args, model, device, train_loader, optimizer, epoch):
    model.train() # model.train() for training while model.learn() for testing
    for batch_idx, (data, target) in enumerate(train_loader):    # for i, (X,y) in enumerate(training data)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,    
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta,           # balance for TRADES
                           distance='l_inf')
        loss.backward()   #   calculate gradients
        optimizer.step()     # again our old friend: zero_grad -> backward -> step (update)
        if batch_idx % args.log_interval == 0:   # check how many times we should log in the concole our epoch num, ratio of trained data, 100% ratio, and loss
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

#define test process

def test(args, model, device, test_loader):
    model.eval()   # here we enter evaluation mode
    test_loss = 0 # loss
    correct = 0   # counter
    with torch.no_grad():   
        # model.eval() will notify all your layers [batchnorm or dropout] that you are in eval mode. torch.no_grad() impacts the autograd engine and deactivate it. 
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss <- negative log likelihood
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()    # pred.eq(target) to compute the ratio of correctness (1 - error rate)

    test_loss /= len(test_loader.dataset)      # Whole sum / total length


    # while training is based on mini batch, here test only on total loss and accuracy
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))   # .0f eg: 2.85 -> 3


def main():
    # Training settings

    # use python3 mnist_example_trades.py -h for later help
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')        # anable cuda for default
    parser.add_argument('--epsilon', default=0.1,
                        help='perturbation')
    parser.add_argument('--num-steps', default=10,
                        help='perturb number of steps')
    parser.add_argument('--step-size', default=0.02,
                        help='perturb step size')
    parser.add_argument('--beta', default=1.0,
                        help='regularization, i.e., 1/lambda in TRADES')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)  # seed for rand

    device = torch.device("cuda" if use_cuda else "cpu")  #old friend 

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}   # page-lock; pinned
    
    
    
    # load data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    # load model structure definition
    model = Net().to(device)


    # define your SGD(Adagrad?Adam?LBFGS?) optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)  # all the arguments in the parser

    for epoch in range(1, args.epochs + 1):        # from epoch 1 to epoch MAX
        train(args, model, device, train_loader, optimizer, epoch)    # each epoch we train
        test(args, model, device, test_loader) # and we test to see converge!
 
    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")   # save model for later use (maybe load for another 100 epochs? LOL)


if __name__ == '__main__':
    main()
