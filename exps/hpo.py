from argparse import ArgumentParser
import os
import os.path as osp
import torch
import torch.nn as nn

import ray
from ray import tune
from ray.tune import Trainable, CLIReporter
from ray.tune.trial import ExportFormat

from exps.train import Trainer
from exps.utils import set_seed
from exps.utils.parsers import train_abs_parser


class SiameseTrainable(Trainable):
    def setup(self, config):
        self.trn = Trainer(config)
        self.best_val_loss = 0.

    @property
    def model(self):
        return self.trn.model

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = osp.join(tmp_checkpoint_dir, "checkpoint.pt")
        model = self.model.to("cpu")
        if isinstance(self.model, nn.DataParallel):
            model = model.module
        torch.save(model.state_dict(), checkpoint_path)
        self.model.to(self.trn.device)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "checkpoint.pt")
        self.model.to("cpu")
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(torch.load(checkpoint_path))
        else:
            self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.to(self.trn.device)

    def _export_model(self, export_formats, export_dir):
        if export_formats == [ExportFormat.MODEL]:
            path = osp.join(export_dir, "exported_model.pt")
            model = self.model.to("cpu")
            if isinstance(self.model, nn.DataParallel):
                model = model.module
            torch.save(model.state_dict(), path)
            self.model.to(self.trn.device)
            return {export_formats[0]: path}
        else:
            raise ValueError(f"unexpected formats: {export_formats}")

    def reset_config(self, new_config):
        self.trn.reset(new_config)
        self.best_val_miou = 0.
        self.config = new_config
        return True

    def step(self):
        train_loss = self.trn.epoch()
        val_loss = self.trn.evaluate()
        result = {"train_loss": train_loss, "val_loss": val_loss}

        if self.best_val_loss > val_loss:
            self.best_val_loss = val_loss
            result.update(should_checkpoint=True)

        return result


def hpo_parser():
    parser = ArgumentParser(parents=[train_abs_parser()])
    parser.add_argument(
        "--hpo_dir",
        type=str,
        default="ray_results",
        help="The folder where to store the hyperparameter optimization "
             "parameters."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="hpo_relab",
        help="The subfolder of the experiment where to store the "
             "hyperparameter optimization results."
    )
    parser.add_argument(
        "--fname_suffix",
        type=str,
        default="",
        help="The suffix appended to the filename for storing the results."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="The number of trials for the hyperparameter optimization."
    )
    parser.add_argument(
        "--cpus_per_trial",  # best 3
        type=int,
        default=1,
        help="The number of used cpus per trial."
    )
    parser.add_argument(
        "--gpus_per_trial",  # best 0.5
        type=float,
        default=0,
        help="The percentage of used gpus per trial."
    )
    return parser


if __name__ == "__main__":
    args = hpo_parser().parse_args()
    set_seed(args.seed)

    exp_name = args.exp_name
    local_dir = osp.abspath(args.hpo_dir)
    check_dir = osp.abspath(osp.join(".", "ray_checkpoints"))

    config = {"dataset": "pascalvoc",
              "data_dir": osp.abspath("./datasets"),
              "batch_size": args.batch_size,
              "lr": tune.loguniform(1e-6, 1e-2) if args.lr is None else args.lr,
              "weight_decay": tune.loguniform(1e-15, 1e-5) if args.weight_decay is None else args.weight_decay,
              "momentum": tune.choice([0.9, 0.95, 0.99]) if args.momentum is None else args.momentum,
              "verbose": args.verbose,
              "device": args.device,
              "device_ids": args.device_ids
              }

    ray.init(num_gpus=torch.cuda.device_count())

    def trial_name_id(trial):
        return str(trial.trial_id)

    def trial_dirname_id(trial):
        config = trial.config
        keys = ("lr", "weight_decay", "momentum", "batch_size")
        dirname_id = "_".join(
            (f"{key}={config[key]:04f}" for key in keys if key in config))
        return f"{trial.trial_id}_" + dirname_id

    reporter = CLIReporter(
        metric_columns=["train_loss", "val_loss", "training_iteration"])

    analysis = tune.run(
        SiameseTrainable,
        name=exp_name,
        metric="min-val_loss",
        mode="min",
        stop={"training_iteration": args.epochs},
        reuse_actors=True,
        resources_per_trial={"cpu": args.cpus_per_trial,
                             "gpu": args.gpus_per_trial},
        config=config,
        num_samples=args.num_samples,
        local_dir=local_dir,
        progress_reporter=reporter,
        export_formats=ExportFormat.MODEL,
        verbose=1,
        trial_name_creator=trial_name_id,
        trial_dirname_creator=trial_dirname_id,
        keep_checkpoints_num=1,
        checkpoint_score_attr="min-val_loss",
    )
