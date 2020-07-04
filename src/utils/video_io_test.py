import torchvision
from torchvision.datasets.video_utils import VideoClips

file_name = "/data1/DataSet/Kinetics/compress/val_256/crying/0Je91ZCyNgk.mkv"
print(torchvision.io.read_video_timestamps(file_name))