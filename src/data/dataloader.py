import torch


def data_loader_init(args, data_length, image_tmpl, train_transforms, test_transforms, eval_transforms):
    if args.dataset == 'ucf101':
        from data.dataset import DataSet as DataSet
    elif args.dataset == 'hmdb51':
        from data.dataset import DataSet as DataSet
    elif args.dataset == 'kinetics':
        from data.video_dataset import VideoDataSet as DataSet
    else:
        Exception("unsupported dataset")
    train_dataset = DataSet(args, args.root, args.train_list, num_segments=1, new_length=data_length,
                      stride=args.stride, modality=args.mode, dataset=args.dataset, test_mode=False,
                      image_tmpl=image_tmpl if args.mode in ["rgb", "RGBDiff"]
                      else args.flow_prefix + "{}_{:05d}.jpg", transform=train_transforms)
    print("training samples:{}".format(train_dataset.__len__()))
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.workers, pin_memory=True)
    val_dataset = DataSet(args, args.root, args.val_list, num_segments=1, new_length=data_length,
                          stride=args.stride, modality=args.mode, test_mode=True, dataset=args.dataset,
                          image_tmpl=image_tmpl if args.mode in ["rgb", "RGBDiff"] else args.flow_prefix + "{}_{:05d}.jpg",
                          random_shift=False, transform=test_transforms)
    print("val samples:{}".format(val_dataset.__len__()))
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True)
    eval_dataset = DataSet(args, args.root, args.val_list, num_segments=1, new_length=data_length,
                          stride=args.stride, modality=args.mode, test_mode=True, dataset=args.dataset,
                          image_tmpl=image_tmpl if args.mode in ["rgb", "RGBDiff"] else args.flow_prefix + "{}_{:05d}.jpg",
                          random_shift=False, transform=eval_transforms, full_video=True)
    print("eval samples:{}".format(eval_dataset.__len__()))
    eval_data_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True)
    return train_data_loader, val_data_loader, eval_data_loader, train_dataset.__len__(), val_dataset.__len__(), eval_dataset.__len__()
