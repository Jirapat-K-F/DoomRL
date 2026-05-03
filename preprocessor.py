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
            frame = frame.to(device)

        # Convert to grayscale if it's RGB (3 channels)
        if frame.shape[1] == 3:
            # Simple average or luminosity formula
            frame = frame.mean(dim=1, keepdim=True)

        # Normalize to [0, 1]
        frame = frame.float() / 255.0

        # Resize
        if frame.shape[-2:] != self.out_size:
            frame = F.interpolate(frame, size=self.out_size, mode='bilinear', align_corners=False)

        # Handle initial frames by filling the deque
        if len(self.frames) == 0:
            for _ in range(self.n_frames):
                self.frames.append(frame)
        else:
            self.frames.append(frame)

        # Stack frames along the channel dimension
        stacked_frames = torch.cat(list(self.frames), dim=1)
        return stacked_frames
    
    def close(self):
        pass

base_preprocessor = FrameStackPreprocessor()

