from importlib import import_module

# model name -> (module, class). The class is imported on demand so that one model's optional dependencies
# (torch_geometric for SMART, natten for Forecast-MAE, ...) are not needed to run another.
__all__ = {
    'autobot': ('.autobot_dataset', 'AutoBotDataset'),
    'wayformer': ('.wayformer_dataset', 'WayformerDataset'),
    'MTR': ('.MTR_dataset', 'MTRDataset'),
    'forecast': ('.fmae_dataset', 'FMAEDataset'),
    'MAE': ('.fmae_dataset', 'FMAEDataset'),
    'EMP': ('.EMP_dataset', 'EMPDataset'),
    'SMART': ('.SMART_dataset', 'SMARTDataset'),
}


def build_dataset(config, val=False):
    module_name, class_name = __all__[config.method.model_name]
    dataset_class = getattr(import_module(module_name, __package__), class_name)
    dataset = dataset_class(
        config=config, is_validation=val
    )
    return dataset
