import os
import torch
import random
import numpy as np
import optuna
from coarse_grained.modules.loss import LossFactory
from coarse_grained.config.all_config import gen_log, AllConfig
from coarse_grained.datasets.data_factory import DataFactory
from coarse_grained.model.model_factory import ModelFactory
from coarse_grained.trainer.trainer_stochastic import Trainer
from coarse_grained.modules.metrics import t2v_metrics, v2t_metrics
from coarse_grained.modules.optimization import AdamW, get_cosine_schedule_with_warmup

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def objective(trial):
    # Initialize config
    config = AllConfig()

    # Define hyperparameters to tune
    config.clip_lr = trial.suggest_loguniform('clip_lr', 1e-7, 1e-5)
    config.noclip_lr = trial.suggest_loguniform('noclip_lr', 1e-6, 1e-4)
    config.batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    config.num_epochs = trial.suggest_int('num_epochs', 5, 50)
    config.weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 0.5)
    config.pooling_type = trial.suggest_categorical('pooling_type', ['topk', 'attention', 'mean'])
    config.embed_dim = trial.suggest_categorical('embed_dim', [256, 512, 768, 1024])  # Add embed_dim to tuning
    config.num_mha_heads = trial.suggest_categorical(
        'num_mha_heads', [h for h in range(1, config.embed_dim + 1) if config.embed_dim % h == 0]
    )
    config.attention_temperature = trial.suggest_uniform('attention_temperature', 0.01, 1.0)
    config.transformer_dropout = trial.suggest_uniform('transformer_dropout', 0.0, 0.5)
    config.seed = trial.suggest_int('seed', 0, 100)

    # Set other configurations as needed
    config.exp_name = "hyperparameter_tuning"  # Example experiment name

    # GPU setup
    if config.gpu is not None and config.gpu != '99':
        print('set GPU')
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise Exception('NO GPU!')

    # Logging setup
    msg = f'model pth = {config.model_path}'
    gen_log(model_path=config.model_path, log_name='log_trntst', msg=msg)
    msg = f'\nconfig={config.__dict__}\n'
    gen_log(model_path=config.model_path, log_name='log_trntst', msg=msg)
    gen_log(model_path=config.model_path, log_name='log_trntst', msg='record all training and testing results')

    # Seed setup
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Tokenizer setup
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)

    # Data I/O
    train_data_loader = DataFactory.get_data_loader(config, split_type='train')
    valid_data_loader = DataFactory.get_data_loader(config, split_type='test')
    model = ModelFactory.get_model(config)

    # Metric setup
    if config.metric == 't2v':
        metrics = t2v_metrics
    elif config.metric == 'v2t':
        metrics = v2t_metrics
    else:
        raise NotImplemented

    # Optimizer setup
    params_optimizer = list(model.named_parameters())
    clip_params = [p for n, p in params_optimizer if "clip." in n]
    noclip_params = [p for n, p in params_optimizer if "clip." not in n]

    optimizer_grouped_params = [
        {'params': clip_params, 'lr': config.clip_lr},
        {'params': noclip_params, 'lr': config.noclip_lr}
    ]
    optimizer = AdamW(optimizer_grouped_params, weight_decay=config.weight_decay)
    num_training_steps = len(train_data_loader) * config.num_epochs
    num_warmup_steps = int(config.warmup_proportion * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    loss = LossFactory.get_loss(config.loss)

    # Trainer setup
    trainer = Trainer(model=model,
                      metrics=metrics,
                      optimizer=optimizer,
                      loss=loss,
                      config=config,
                      train_data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=scheduler,
                      writer=None,
                      tokenizer=tokenizer)

    # Training
    trainer.train()

    # Evaluation: Return a combination of validation metrics (R@1, R@5, R@10)
    eval_results = trainer.evaluate()
    R1 = eval_results.get('R@1', 0)
    R5 = eval_results.get('R@5', 0)
    R10 = eval_results.get('R@10', 0)

    # Log the results
    with open("tuning_results.txt", "a") as f:
        f.write(f"Trial {trial.number}\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"R@1: {R1}, R@5: {R5}, R@10: {R10}\n\n")

    # Combine the metrics, for example by averaging them
    return (R1 + R5 + R10) / 3.0


def main():
    study = optuna.create_study(direction='maximize')  # Change to 'minimize' if optimizing a loss
    study.optimize(objective, n_trials=50)  # Adjust number of trials as needed

    # Print and save the best hyperparameters
    best_params = study.best_params
    print(f"Best Hyperparameters: {best_params}")
    with open("best_hyperparameters.txt", "w") as f:
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")

    # Log the best trial results
    best_trial = study.best_trial
    with open("tuning_results.txt", "a") as f:
        f.write("\nBest Trial:\n")
        for key, value in best_trial.params.items():
            f.write(f"{key}: {value}\n")
        f.write(
            f"R@1: {best_trial.user_attrs['R@1']}, R@5: {best_trial.user_attrs['R@5']}, R@10: {best_trial.user_attrs['R@10']}\n\n")


if __name__ == '__main__':
    main()
