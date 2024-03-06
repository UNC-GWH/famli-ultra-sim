import torch
import pytorch_lightning as pl
from options.train_options import TrainOptions
from data import create_dataset  # Or your custom DataModule if you have one
from models import CUTModelLightning
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger  # If you are using Neptune

if __name__ == '__main__':
    # Parse command-line options
    opt = TrainOptions().parse()

    # Create a dataset or a data module
    dataset = create_dataset(opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    # Initialize the model
    model = CUTModelLightning(opt)

    # Configure callbacks
    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints', monitor='val_loss', save_top_k=2, mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    # Configure Neptune Logger (Optional)
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ZTQwZDg4My00NWQwLTQ1MWYtYmM3Yi1lZjE2YjdiYTA5YTIifQ==",
        project_name="ImageMindAnalytics/CUT",
        experiment_name='Experiment-1',  # Optional
        params=vars(opt)  # Logging hyperparameters
    )

    # Setup Trainer
    trainer = pl.Trainer(
        max_epochs=opt.n_epochs + opt.n_epochs_decay,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=neptune_logger if opt.use_neptune else None,
        gpus=1 if torch.cuda.is_available() else None,
        progress_bar_refresh_rate=20  # or another number, depending on your preference
    )

    # Start Training
    trainer.fit(model, dataloader)
