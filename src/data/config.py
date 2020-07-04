import TC.video_transformations.videotransforms as videotransforms
import TC.video_transformations.video_transform_PIL_or_np as video_transform
from TC.video_transformations.volume_transforms import ClipToTensor

from torchvision import transforms


def data_config(args):
    if args.dataset == 'ucf101':
        num_class = 101
        image_tmpl = "frame{:06d}.jpg"
    elif args.dataset == 'hmdb51':
        num_class = 51
        image_tmpl = "img_{:05d}.jpg"
    elif args.dataset == 'kinetics':
        num_class = 400
        image_tmpl = "img_{:05d}.jpg"
        args.root = "/data1/DataSet/Kinetics/compress/"
    elif args.dataset == 'something_something_v1':
        num_class = 174
        image_tmpl = "{:05d}.jpg"
    else:
        raise ValueError('Unknown dataset ' + args.dataset)
    return num_class, int(args.data_length), image_tmpl


def augmentation_config(args):
    if int(args.spatial_size) == 112:
        # print("??????????????????????????")
        resize_size = 128
    else:
        resize_size = 256
    train_transforms = transforms.Compose([
        #videotransforms.RandomCrop(int(args.spatial_size)),
        video_transform.RandomRotation(10),
        video_transform.Resize(resize_size),
        video_transform.RandomCrop(int(args.spatial_size)),
        video_transform.ColorJitter(0.5, 0.5, 0.25, 0.5),
        ClipToTensor(channel_nb=3),
        video_transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # videotransforms.ColorJitter(),
        # videotransforms.RandomHorizontalFlip()
    ])
    test_transforms = transforms.Compose([
                                        video_transform.Resize(resize_size),
                                        video_transform.CenterCrop(int(args.spatial_size)),
                                         ClipToTensor(channel_nb=3),
                                         video_transform.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])]
                                         )
    eval_transfroms = transforms.Compose([
                                            video_transform.Resize(resize_size),
                                            video_transform.CenterCrop(int(args.spatial_size)),
                                         ClipToTensor(channel_nb=3),
                                         video_transform.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])
    ]
                                         )
    return train_transforms, test_transforms, eval_transfroms
