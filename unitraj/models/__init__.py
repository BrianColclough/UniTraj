from importlib import import_module

# model name -> (module, class). The class is imported on demand so that one model's optional dependencies
# (CUDA ops for MTR, torch_geometric and waymo_open_dataset for SMART, natten for Forecast-MAE, ...) are not
# needed to run another.
__all__ = {
    'autobot': ('unitraj.models.autobot.autobot', 'AutoBotEgo'),
    'wayformer': ('unitraj.models.wayformer.wayformer', 'Wayformer'),
    'MTR': ('unitraj.models.mtr.MTR', 'MotionTransformer'),
    'MAE': ('unitraj.models.fmae.trainer_mae', 'TrainerMAE'),
    'forecast': ('unitraj.models.fmae.trainer_forecast', 'TrainerForecast'),
    'EMP': ('unitraj.models.emp.trainer_forecast', 'TrainerEMP'),
    'SMART': ('unitraj.models.smart.smart', 'SMART'),
}


def build_model(config):
    module_name, class_name = __all__[config.method.model_name]
    model_class = getattr(import_module(module_name), class_name)
    model = model_class(
        config=config
    )

    return model
