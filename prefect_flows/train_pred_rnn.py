"""
PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning

@inproceedings{wang2017predrnn,
  title={{PredRNN}: Recurrent Neural Networks for Predictive Learning Using Spatiotemporal {LSTM}s},
  author={Wang, Yunbo and Long, Mingsheng and Wang, Jianmin and Gao, Zhifeng and Yu, Philip S},
  booktitle={Advances in Neural Information Processing Systems},
  pages={879--888},
  year={2017}
}

@misc{wang2021predrnn,
      title={{PredRNN}: A Recurrent Neural Network for Spatiotemporal Predictive Learning}, 
      author={Wang, Yunbo and Wu, Haixu and Zhang, Jianjin and Gao, Zhifeng and Wang, Jianmin and Yu, Philip S and Long, Mingsheng},
      year={2021},
      eprint={2103.09504},
      archivePrefix={arXiv},
}

https://github.com/thuml/predrnn-pytorch
"""

import torch
import torch.nn as nn
from prefect import flow
import mlflow
from datetime import datetime
from torchinfo import summary
from torch.optim import Adam

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new

class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs['patch_size'] * configs['patch_size'] * configs['img_channel']
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs['img_width'] // configs['patch_size']
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs['filter_size'],
                                       configs['stride'], configs['layer_norm'])
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor#.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true#.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs['device'])
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs['device'])

        for t in range(self.configs['total_length'] - 1):
            # reverse schedule sampling
            if self.configs['reverse_scheduled_sampling'] == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs['input_length']:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs['input_length']] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs['input_length']]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 4, 3).contiguous()
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        return next_frames, loss
    
from train_conv_lstm import load_adj_matrices

@flow(log_prints=True)
def train_pred_rnn_flow():
    train_loader, test_loader = load_adj_matrices()

    with mlflow.start_run():
    
        params = {
    "num_layers": 1,
    "num_hidden": [64],
    "configs": {
        "patch_size": 1, 
        "img_channel": 1,
        "img_width": 67,
        "filter_size": 5,
        "stride": 1,
        "layer_norm": False,
        "device": 'mps',
        "total_length": 9,
        "input_length": 1,
        "reverse_scheduled_sampling": 1
    }
}
        mlflow.log_params(params)
        lr = 1e-4
        mlflow.log_params({'lr': lr})
        device = 'mps'
        
        model = RNN(params['num_layers'], params['num_hidden'], params['configs']).to(device)
        with open('model_summary.txt', 'w') as f:
            f.write(str(summary(model)))
        mlflow.log_artifact('model_summary.txt')
        
        optim = Adam(model.parameters(), lr=lr)

        num_epochs = 30

        for epoch in range(num_epochs):
            train_loss = 0
            for _, sequence in enumerate(train_loader):
                inputs = sequence[:, :-1, :, :, :].to(device)
                targets = sequence[:, 1:, :, :, :].to(device)

                
                outputs, loss = model(inputs, targets)
                if isinstance(outputs, list):
                    outputs = torch.stack(outputs)
                
                train_loss += loss.item()
                
                optim.zero_grad()
                loss.backward()
                optim.step()

            train_loss = train_loss/len(train_loader)
            mlflow.log_metric("train_loss", loss, step=epoch)
            
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for i, sequence in enumerate(test_loader):
                    inputs = sequence[:, :-1, :, :, :].to(device)
                    targets = sequence[:, 1:, :, :, :].to(device)

                    outputs, loss = model(inputs, targets)
                    if isinstance(outputs, list):
                        outputs = torch.stack(outputs)
                
                    test_loss += loss.item()
            test_loss = test_loss/len(test_loader)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        torch.save(model.state_dict(), f"model_{current_time}.pth")
        mlflow.log_artifact(f"model_{current_time}.pth")

if __name__ == '__main__':
    train_pred_rnn_flow()