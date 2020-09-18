# Prepare Data

## Download Dataset

In this work, we explore three datasets.
First doownload them from [UCF101](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
), [HMDB51](https://www.crcv.ucf.edu/data/UCF101.php) and [Kinetics](https://deepmind.com/research/open-source/kinetics).

## Video to frames
Use ffmpeg to generate frames for videos as in [TSN](https://github.com/yjxiong/temporal-segment-networks).

I suggest you download these extracted frames from [two stream](https://github.com/jeffreyyihuang/two-stream-action-recognition) directly.

## Generate Lists
generate list is quite easy. The list is in the format
> video_frames_path num_frames class

like

> Data/hmdb51/Oceans11_eat_h_cm_np1_le_goo_2 77 11

Look **Construct file lists for training and validation** section in [TSN](https://github.com/yjxiong/temporal-segment-networks).

## Others