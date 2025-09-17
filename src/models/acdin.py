import torch
import torch.nn as nn
import torch.nn.functional as F
class BasicConv1d(nn.Module):
    """Conv1d (+ optional BN) + ReLU"""
    def __init__(self, in_ch, out_ch, k, s=1, p=0, dilation=1, use_bn=False):
        super().__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=dilation, bias=not use_bn)
        self.bn = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

class InceptionACDIN(nn.Module):
    """
    Inception cho ACDIN:
      - Branch1: 1x1 -> C1
      - Branch2: 1x1 (reduce=Cm) -> 3x3 dilated (rate) -> C2
      - Branch3: 1x1 (reduce=Cm) -> 5x5 dilated (rate) -> C3
      - Branch4: MaxPool(k=3,s=1,p=1) -> 1x1 -> C4
    Tất cả conv trong module dùng padding 'same' để giữ nguyên chiều dài theo trục thời gian.
    """
    def __init__(self, in_ch, Cm, C1, C2, C3, C4, dilation_rate=2, use_bn=False):
        super().__init__()
        d = dilation_rate
        # Branch 1: 1x1
        self.b1 = BasicConv1d(in_ch, C1, k=1, use_bn=use_bn)

        # Branch 2: 1x1 reduce -> 3x3 dilated
        self.b2_reduce = BasicConv1d(in_ch, Cm, k=1, use_bn=use_bn)
        self.b2_conv = BasicConv1d(Cm, C2, k=3, p=d, dilation=d, use_bn=use_bn)  # pad = d for k=3

        # Branch 3: 1x1 reduce -> 5x5 dilated
        self.b3_reduce = BasicConv1d(in_ch, Cm, k=1, use_bn=use_bn)
        self.b3_conv = BasicConv1d(Cm, C3, k=5, p=2*d, dilation=d, use_bn=use_bn)  # pad = 2d for k=5

        # Branch 4: pool -> 1x1
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.b4 = BasicConv1d(in_ch, C4, k=1, use_bn=use_bn)

        self.out_ch = C1 + C2 + C3 + C4

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2_conv(self.b2_reduce(x))
        y3 = self.b3_conv(self.b3_reduce(x))
        y4 = self.b4(self.pool(x))
        return torch.cat([y1, y2, y3, y4], dim=1)  # concat theo kênh


class AuxClassifier(nn.Module):
    """
    Bộ phân loại phụ: GAP -> Dropout -> Linear(num_classes)
    Dùng cho các vị trí 'Softmax 3x1' trong bảng.
    """
    def __init__(self, in_ch, num_classes=3, p=0.7):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)   # GAP over time
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        # x: (B, C, T)
        z = self.pool(x).squeeze(-1)  # (B, C)
        z = self.drop(z)
        return self.fc(z)             # (B, num_classes)


# --- ACDIN model ---

class ACDIN(nn.Module):
    """
    ACDIN theo Table 5.
    - Dùng atrous conv (dilation=2) trong các Inception.
    - Có 2 auxiliary classifiers (sau các mốc ghi 'Softmax 3x1').
    - Has BN: chỉ Inception(5b) bật BN=True theo bảng; còn lại BN=False.
    """
    def __init__(self, in_ch=1, num_classes=3, aux_logits=True):
        super().__init__()
        self.aux_logits = aux_logits

        # --- Stem: Conv -> Pool -> Conv -> Pool
        # Row: Convolution 3/1 -> 5118 × 64 (gợi ý: conv đầu không padding để giảm 2 mẫu)
        self.conv1 = BasicConv1d(in_ch, 64, k=3, s=1, p=0, use_bn=False)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)

        # Row: Convolution 3/1 -> 1706 × 128 (giữ length -> padding=1)
        self.conv2 = BasicConv1d(64, 128, k=3, s=1, p=1, use_bn=False)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)

        # --- Inception blocks theo bảng ---

        # Inception*2 @ 568 × 192 | Cm=16, C1=64, C2=64, C3=32, C4=32 | dilate=2 | BN=N
        self.incp_3a = InceptionACDIN(128, Cm=16, C1=64, C2=64, C3=32, C4=32, dilation_rate=2, use_bn=False)
        self.incp_3b = InceptionACDIN(self.incp_3a.out_ch, Cm=16, C1=64, C2=64, C3=32, C4=32, dilation_rate=2, use_bn=False)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=3)

        # Inception*2 @ 189 × 192 (trong bảng là thêm một lần *2 nữa)
        self.incp_4a = InceptionACDIN(self.incp_3b.out_ch, Cm=16, C1=64, C2=64, C3=32, C4=32, dilation_rate=2, use_bn=False)

        # Inception*3 @ 189 × 160 | Cm=16, C1=48, C2=48, C3=32, C4=32
        self.incp_4b = InceptionACDIN(self.incp_4a.out_ch, Cm=16, C1=48, C2=48, C3=32, C4=32, dilation_rate=2, use_bn=False)
        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=3)

        # --- Aux classifier #1 (sau pool4: 63 × 160) ---
        if self.aux_logits:
            self.aux1 = AuxClassifier(in_ch=self.incp_4b.out_ch, num_classes=num_classes, p=0.7)

        # Inception*3 @ 63 × 256 | Cm=16, C1=64, C2=96, C3=64, C4=32
        self.incp_5a = InceptionACDIN(self.incp_4b.out_ch, Cm=16, C1=64, C2=96, C3=64, C4=32, dilation_rate=2, use_bn=False)
        self.pool5 = nn.MaxPool1d(kernel_size=3, stride=3)

        # Inception*2 @ 21 × 608 | Cm=16, C1=128, C2=128, C3=256, C4=96
        self.incp_5b = InceptionACDIN(self.incp_5a.out_ch, Cm=16, C1=128, C2=128, C3=256, C4=96, dilation_rate=2, use_bn=False)
        self.pool6 = nn.MaxPool1d(kernel_size=2, stride=2)

        # --- Aux classifier #2 (sau pool6: 10 × 608) ---
        if self.aux_logits:
            self.aux2 = AuxClassifier(in_ch=self.incp_5b.out_ch, num_classes=num_classes, p=0.7)

        # Inception(5a) @ 10 × 384 | Cm=16, C1=96, C2=96, C3=128, C4=64 | BN=N
        self.incp_6a = InceptionACDIN(self.incp_5b.out_ch, Cm=16, C1=96, C2=96, C3=128, C4=64, dilation_rate=2, use_bn=False)

        # Inception(5b) @ 10 × 384 | Cm=16, C1=96, C2=96, C3=128, C4=64 | BN=Y
        self.incp_6b = InceptionACDIN(self.incp_6a.out_ch, Cm=16, C1=96, C2=96, C3=128, C4=64, dilation_rate=2, use_bn=True)

        self.pool7 = nn.MaxPool1d(kernel_size=2, stride=2)  # -> 5 × 384

        # --- Head: GAP -> FC(num_classes)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.incp_6b.out_ch, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 1, T)
        x = self.conv1(x)   # -> (B, 64, T-2)
        x = self.pool1(x)   # stride 3
        x = self.conv2(x)   # keep length (pad=1)
        x = self.pool2(x)

        x = self.incp_3a(x)
        x = self.incp_3b(x)
        x = self.pool3(x)

        x = self.incp_4a(x)
        x = self.incp_4b(x)
        x = self.pool4(x)

        aux1_out = None
        if self.training and self.aux_logits:
            aux1_out = self.aux1(x)

        x = self.incp_5a(x)
        x = self.pool5(x)

        x = self.incp_5b(x)
        x = self.pool6(x)

        aux2_out = None
        if self.training and self.aux_logits:
            aux2_out = self.aux2(x)

        x = self.incp_6a(x)
        x = self.incp_6b(x)
        x = self.pool7(x)  # -> length ~5

        z = self.gap(x).squeeze(-1)  # (B, C)
        z = self.dropout(z)
        logits = self.fc(z)

        if self.training and self.aux_logits:
            return logits, aux1_out, aux2_out  # dùng loss phụ khi train
        return logits