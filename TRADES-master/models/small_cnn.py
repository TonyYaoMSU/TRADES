from collections import OrderedDict
import torch.nn as nn

# for the use of train_trades_mnist

class SmallCNN(nn.Module):
    def __init__(self, drop=0.5):

        # later super..... c++.....
        super(SmallCNN, self).__init__()

        # initial channels <- balck& white pic has only one channel
        self.num_channels = 1
        self.num_labels = 10 # 10 label mnist

        activ = nn.ReLU(True) # 节省内存

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)), # 26*26
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)), # 24*24
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)), # 12*12
            ('conv3', nn.Conv2d(32, 64, 3)), # 10*10
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)), # 8*8
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)), # 4*4
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)), # 64 channel
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),  # half 0s
            ('fc2', nn.Linear(200, 200)), # 200 -> 200 fc
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):

                # Fills the input Tensor with values according to the method described in Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015), using a normal distribution. 
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: #Fills the input Tensor with the value val.
                    nn.init.constant_(m.bias, 0)


                # if it is conv then He else (1,1,1,......1,1,1,1,0) weights

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # specially initiate fc3 with (0,0,0,0,0,0......)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input):
        # first extract features
        features = self.feature_extractor(input)


        # then flat the conv features into one embedding vector and feed it into FC (MLP) fully connected layer
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits