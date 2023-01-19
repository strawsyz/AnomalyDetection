import torch
import torch.nn as nn
from torch.nn import functional as F

from NK import NK
from transformer import Transformer


class Learner2(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0):
        super(Learner2, self).__init__()
        self.filter1 = nn.LayerNorm(input_dim)
        self.filter2 = nn.LayerNorm(input_dim)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout(drop_p)

        self.fc1 = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 256)
        self.fc4 = nn.Linear(256, 32)
        self.fc5 = nn.Linear(32, 1)

        self.nk1 = NK(memory_rate=1.0, num_key_memory=10, max_memory_size=15,
                      threshold_caption_score=0.1)
        self.nk2 = NK(memory_rate=1.0, num_key_memory=10, max_memory_size=15,
                      threshold_caption_score=0.1)

        self.tf1 = Transformer(32, 32, 1, 1, 0)
        self.tf2 = Transformer(32, 32, 6, 8, 0)

    # def weight_init(self):
    #     for layer in self.classifier:
    #         if type(layer) == nn.Linear:
    #             nn.init.xavier_normal_(layer.weight)

    def forward(self, x, vars=None):
        x = x.float()
        x1 = self.relu2(x)

        # x1 = x[:, :1024]
        # x = x[:, 1024:]

        x = self.fc4((self.relu(self.fc3(self.dropout(self.relu(self.fc2(self.relu(self.fc1(x)))))))))
        # out = self.nk1(out)
        x = self.relu(x)
        batch_size = int(x.shape[0] / 64)
        if self.training: #batch_size > 0:
            x = torch.reshape(x, (batch_size, 64, 32))
            a_x = torch.reshape(x, (batch_size, 64, 32))[:, :32, :]
            n_x = torch.reshape(x, (batch_size, 64, 32))[:, 32:, :]
            a_output = self.tf1(a_x)
            n_output = self.tf1(n_x)
            output = torch.cat([a_output, n_output], dim=1)
            output = torch.reshape(output, (int(batch_size*64), 1))
            output = F.sigmoid(output)
            # output = F.relu(output)
            return output
        else:
            output = self.tf1(x)
            output = output.squeeze()
            output = F.sigmoid(output)
            # output = F.relu(output)
            return output
        # out = self.tf1(out)
        # out = self.fc5(self.relu(out))

        out2 = self.fc4(self.relu(self.fc3(self.dropout(self.relu(self.fc2(self.relu(self.fc1(x1))))))))
        # out2 = self.nk2(out2)
        ou2 = self.tf2(out2)
        # out2 = self.fc5(self.relu(out2))

        out = out + out2 * 0.2
        return torch.sigmoid(out)
