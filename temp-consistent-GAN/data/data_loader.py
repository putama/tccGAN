
def CreateDataLoader(opt, mode="train"):
    if mode == "test":
        opt.nThreads = 1
        opt.batchSize = 1
        opt.serial_batches = True
        opt.no_flip = True
        opt.dataset_mode = 'unaligned'
        print("Create testing dataset ...")
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
