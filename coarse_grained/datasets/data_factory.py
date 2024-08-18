from coarse_grained.config.base_config import Config
from coarse_grained.datasets.model_transforms import init_transform_dict
from coarse_grained.datasets.msrvtt_dataset import MSRVTTDataset
from coarse_grained.datasets.lsmdc_dataset import LSMDCDataset
from coarse_grained.datasets.didemo_dataset import DiDeMoDataset
from coarse_grained.datasets.tacos_cg_dataset import TACoSCoarseGrainedDataset
from torch.utils.data import DataLoader


class DataFactory:

    @staticmethod
    def get_data_loader(config: Config, split_type='train'):
        img_transforms = init_transform_dict(config.input_res)
        train_img_tfms = img_transforms['clip_train']
        test_img_tfms = img_transforms['clip_test']

        if config.dataset_name == "MSRVTT":
            if split_type == 'train':
                dataset = MSRVTTDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                                  shuffle=True, num_workers=config.num_workers)
            else:
                dataset = MSRVTTDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                                  shuffle=False, num_workers=config.num_workers)

        elif config.dataset_name == 'LSMDC':
            if split_type == 'train':
                dataset = LSMDCDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                                  shuffle=True, num_workers=config.num_workers)
            else:
                dataset = LSMDCDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                                  shuffle=False, num_workers=config.num_workers)

        elif config.dataset_name == "DiDeMo":
            if split_type == 'train':
                dataset = DiDeMoDataset(config, split_type, train_img_tfms)
                shuffle = True
                return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle,
                                  num_workers=config.num_workers)
            else:
                dataset = DiDeMoDataset(config, split_type, test_img_tfms)
                shuffle = False
                return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle,
                                  num_workers=config.num_workers)


        elif config.dataset_name == "TACoSCoarseGrained":

            if split_type == 'train':
                split_file = 'train.txt'
            elif split_type == 'test':
                split_file = 'test.txt'
            else:
                split_file = 'eval.txt'

            dataset = TACoSCoarseGrainedDataset(
                config,
                'tacos_cg.json',
                split_file,
                img_transforms=train_img_tfms if split_type == 'train' else test_img_tfms
            )

            return DataLoader(
                dataset, batch_size=config.batch_size,
                shuffle=(split_type == 'train'),
                num_workers=config.num_workers
            )

        else:
            raise NotImplementedError
