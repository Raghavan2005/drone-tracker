"""DroneNet-Pico: ~500K parameter anchor-free detector optimized for drone detection."""

import torch
import torch.nn as nn


class ConvBnSiLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MicroBlock(nn.Module):
    """Depthwise-separable conv with residual."""
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return x + self.act(self.bn2(self.pw(self.act(self.bn1(self.dw(x))))))


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Stage 0: 416 -> 208
        self.stem = ConvBnSiLU(3, 16, 3, 2)
        # Stage 1: 208 -> 104
        self.stage1 = nn.Sequential(ConvBnSiLU(16, 32, 3, 2), MicroBlock(32))
        # Stage 2: 104 -> 52 (P3)
        self.stage2 = nn.Sequential(ConvBnSiLU(32, 64, 3, 2), MicroBlock(64), MicroBlock(64))
        # Stage 3: 52 -> 26 (P4)
        self.stage3 = nn.Sequential(ConvBnSiLU(64, 128, 3, 2), MicroBlock(128), MicroBlock(128))
        # Stage 4: 26 -> 13 (P5)
        self.stage4 = nn.Sequential(ConvBnSiLU(128, 256, 3, 2), MicroBlock(256))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)   # 52x52x64
        p4 = self.stage3(p3)  # 26x26x128
        p5 = self.stage4(p4)  # 13x13x256
        return p3, p4, p5


class PANLite(nn.Module):
    """Lightweight Path Aggregation Network (2 scales only)."""
    def __init__(self):
        super().__init__()
        # Top-down: P5 -> P4
        self.up_conv1 = ConvBnSiLU(256 + 128, 128, 1)
        self.up_conv2 = ConvBnSiLU(128, 128, 3)
        # Top-down: -> P3
        self.up_conv3 = ConvBnSiLU(128 + 64, 64, 1)
        self.up_conv4 = ConvBnSiLU(64, 64, 3)
        # Bottom-up: P3 -> P4
        self.down_conv1 = ConvBnSiLU(64, 64, 3, 2)
        self.down_conv2 = ConvBnSiLU(64 + 128, 128, 1)
        self.down_conv3 = ConvBnSiLU(128, 128, 3)

    def forward(self, p3, p4, p5):
        # Top-down
        up5 = nn.functional.interpolate(p5, size=p4.shape[2:], mode="nearest")
        n4 = self.up_conv2(self.up_conv1(torch.cat([up5, p4], dim=1)))

        up4 = nn.functional.interpolate(n4, size=p3.shape[2:], mode="nearest")
        n3 = self.up_conv4(self.up_conv3(torch.cat([up4, p3], dim=1)))  # 52x52x64

        # Bottom-up
        down3 = self.down_conv1(n3)
        d4 = self.down_conv3(self.down_conv2(torch.cat([down3, n4], dim=1)))  # 26x26x128

        return n3, d4


class DetectionHead(nn.Module):
    """Anchor-free decoupled detection head for one scale."""
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.cls_conv = ConvBnSiLU(in_ch, in_ch, 3)
        self.cls_pred = nn.Conv2d(in_ch, num_classes, 1)
        self.reg_conv = ConvBnSiLU(in_ch, in_ch, 3)
        self.reg_pred = nn.Conv2d(in_ch, 4, 1)
        self.obj_pred = nn.Conv2d(in_ch, 1, 1)

    def forward(self, x):
        cls_feat = self.cls_conv(x)
        cls_out = self.cls_pred(cls_feat)

        reg_feat = self.reg_conv(x)
        reg_out = self.reg_pred(reg_feat)
        obj_out = self.obj_pred(reg_feat)

        return torch.cat([reg_out, obj_out, cls_out], dim=1)


class DroneNetPico(nn.Module):
    def __init__(self, num_classes=5, input_size=416):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        self.backbone = Backbone()
        self.neck = PANLite()
        self.head_s = DetectionHead(64, num_classes)   # 52x52
        self.head_m = DetectionHead(128, num_classes)  # 26x26

        self.strides = [8, 16]
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        n3, d4 = self.neck(p3, p4, p5)

        out_s = self.head_s(n3)  # B, 5+C, 52, 52
        out_m = self.head_m(d4)  # B, 5+C, 26, 26

        if self.training:
            return [out_s, out_m]

        return self._decode([out_s, out_m])

    def _decode(self, outputs):
        """Decode raw outputs to [batch, num_detections, 5+num_classes] for inference."""
        decoded = []
        for i, out in enumerate(outputs):
            b, c, h, w = out.shape
            out = out.permute(0, 2, 3, 1).reshape(b, h * w, c)

            stride = self.strides[i]
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=out.device, dtype=out.dtype),
                torch.arange(w, device=out.device, dtype=out.dtype),
                indexing="ij"
            )
            grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
            grid = grid.unsqueeze(0)

            # Decode: cx, cy, w, h
            xy = (out[..., :2].sigmoid() + grid) * stride
            wh = out[..., 2:4].exp() * stride
            obj = out[..., 4:5].sigmoid()
            cls = out[..., 5:].sigmoid()

            decoded.append(torch.cat([xy, wh, obj, cls], dim=-1))

        return torch.cat(decoded, dim=1)  # B, 3380, 5+C


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = DroneNetPico(num_classes=5)
    print(f"Parameters: {count_parameters(model):,}")

    x = torch.randn(1, 3, 416, 416)

    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")

    model.train()
    outs = model(x)
    for i, o in enumerate(outs):
        print(f"Scale {i}: {o.shape}")
