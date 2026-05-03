import torch
import torch.nn.functional as F
import collections

class FrameStackPreprocessor:
    def __init__(self, n_frames=4, out_size=(112, 112)):
        self.n_frames = n_frames
        self.out_size = out_size
        self.frames = collections.deque(maxlen=n_frames)

    def reset(self):
        self.frames.clear()

    def __call__(self, frame, device=None):
        if frame is None:
            return None
        
        # Ensure we are on the correct device
        if device is not None:
            frame = frame.to(device, non_blocking=True)

        # Grayscale conversion using luminosity weights for speed/accuracy
        if frame.shape[1] == 3:
            # frame is [B, 3, H, W]
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=frame.device).view(1, 3, 1, 1)
            frame = (frame * weights).sum(dim=1, keepdim=True)

        # Combined normalization and resizing
        frame = F.interpolate(frame.float() / 255.0, size=self.out_size, mode='bilinear', align_corners=False)

        # Handle initial frames by filling the deque
        if len(self.frames) == 0:
            for _ in range(self.n_frames):
                self.frames.append(frame)
        else:
            self.frames.append(frame)

        # Stack frames along the channel dimension
        return torch.cat(list(self.frames), dim=1)
    
    def close(self):
        pass

base_preprocessor = FrameStackPreprocessor()

