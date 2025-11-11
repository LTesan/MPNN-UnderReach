import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from src.model.model import SimulatorGNN
from src.model.model_meshgraph import SimulatorMeshGraph 
from src.dataloader.datamodule import GraphDataModule
from src.utils.utils import set_run_directory
from src.model.callbacks import ModelSaveTopK
import wandb


torch.set_float32_matmul_precision('medium')
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_dir = 'data/PlasticCollision'


def train(config=None):
    
    # Override default hyperparameters with sweep config if provided
    with wandb.init(config=config):
        args = wandb.config

    # Set seed
    pl.seed_everything(args.seed, workers=True)

    # Data preparation
    data_module = GraphDataModule(dataset_dir=dataset_dir, batch_size=args.batch_size, ratio=args.ratio, dataset_type=args.model)

    # Model instantiation
    if args.model == 'gnn':
        simulator = SimulatorGNN(args, mp_steps=args.mp_steps, input_size=2, output_size=1, hidden=args.hidden, 
                                layers=args.layers, shared_mp=args.shared_mp, epochs=args.epochs, lr=args.lr, device=device, noise=args.noise)
        monitor='valid_n_step_rollout_rmse'

    elif args.model == 'poisson':
        simulator = SimulatorGNN(args, mp_steps=args.mp_steps, input_size=2, output_size=1, hidden=args.hidden,
                                layers=args.layers, shared_mp=args.shared_mp, epochs=args.epochs, lr=args.lr, device=device, noise=args.noise)
        monitor='valid_n_step_rollout_rmse'
        
    elif args.model =='meshgraph':
        simulator = SimulatorMeshGraph(args, mp_steps=args.mp_steps, input_size=7, edge_input_size=4, output_size=4, hidden=args.hidden,
                                layers=args.layers, shared_mp=args.shared_mp, epochs=args.epochs, lr=args.lr, device=device, noise=args.noise)
        monitor='valid_n_step_rollout_pos_rmse'

    # Callbacks
    chck_path, name = set_run_directory(args, safe_mode=False)
    wandb_logger = WandbLogger(name=name, project='test_swep')
    
    model_save_custom = ModelSaveTopK(dirpath=str(chck_path / 'models'), monitor=monitor, mode='min', topk=3)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Trainer
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                            max_epochs=args.epochs,
                            logger=wandb_logger,
                            callbacks=[lr_monitor, model_save_custom],
                            num_sanity_val_steps=0,
                            deterministic=True,
                            check_val_every_n_epoch=args.eval_freq,
                            enable_checkpointing=False,
                            
                         )
    # fit model
    trainer.fit(simulator, datamodule=data_module)
    wandb.save(str(chck_path / 'config.json'))
    trainer.test(simulator, datamodule=data_module)

    # test model
    simulator.load_checkpoint(str(chck_path / 'models' / 'topk1.pth'))
    trainer.test(simulator, datamodule=data_module)


if __name__ == '__main__':
    train()