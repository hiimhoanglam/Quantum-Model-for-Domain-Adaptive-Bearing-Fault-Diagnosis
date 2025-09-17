import torch
import torch.nn as nn
import torch.nn.functional as F
class CWRUWDCNNTime(nn.Module):
    """
    WDCNN (Time-domain) for 1D vibration (e.g., CWRU)
    Input shape: [B, 1, 2048]
    Head: FC(100) -> BN -> ReLU -> FC(num_classes)
    Note: Do NOT apply softmax in forward; use CrossEntropyLoss on logits.
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()

        # 1) Wide first conv: k=64, s=16, padding='same' -> [B, 16, 128]
        self.conv1 = nn.Conv1d(1, 16, kernel_size=64, stride=16, padding=32, bias=False)
        self.bn1   = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)   # -> [B, 16, 64]

        # 2) Conv k=3, s=1, same padding
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)   # -> [B, 32, 32]

        # 3) Conv k=3
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3   = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)   # -> [B, 64, 16]

        # 4) Conv k=3
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4   = nn.BatchNorm1d(64)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)   # -> [B, 64, 8]

        # 5) Conv k=3, NO padding (giảm width 2)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn5   = nn.BatchNorm1d(64)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)   # -> [B, 64, 3]

        # Head: 64*3 = 192 -> 100 -> num_classes
        self.fc1   = nn.Linear(64*3, 100, bias=False)
        self.bn_fc = nn.BatchNorm1d(100)
        self.fc2   = nn.Linear(100, num_classes)

        self._init_weights()
        
        self.dropout = nn.Dropout(p=0.4)
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Expect x: [B, 1, 2048]. 
        # Nếu dataloader của bạn đang cho [B, 2048, 1], hãy transpose trước: x = x.transpose(1, 2)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # [B, 16, 64]
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # [B, 32, 32]
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # [B, 64, 16]
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))  # [B, 64, 8]
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))  # [B, 64, 3]

        x = x.flatten(1)                                 # [B, 64*3=192]
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)               # [B, 100]
        logits = self.fc2(x)                             # [B, num_classes]
        return logits

    @torch.no_grad()
    def adabn_recalibrate(self, loader, device):
        """
        AdaBN: cập nhật running mean/var của BN bằng dữ liệu miền đích (unlabeled).
        Không cập nhật trọng số; chỉ BN stats.
        """
        was_training = self.training
        self.train()  # BN cập nhật running stats trong train mode
        # Freeze grad (chỉ forward để BN cập nhật running mean/var)
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d,)):
                m.momentum = 0.1  # optional: mặc định PyTorch
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            inputs = inputs.to(device, non_blocking=True)
            # Nếu inputs là [B, 2048, 1], chuyển sang [B, 1, 2048] cho Conv1d
            if inputs.dim() == 3 and inputs.shape[2] == 1:
                inputs = inputs.transpose(1, 2)
            _ = self(inputs)
        self.train(was_training)