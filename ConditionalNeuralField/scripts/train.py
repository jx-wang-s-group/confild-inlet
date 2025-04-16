from os.path import exists
from os import mkdir
import sys
from basicutility import ReadInput as ri

# RI_PARAMS = {
#     "read_multiple": False,
#     "hidden_size": 256,
#     "readin_data_shape": "t h w c",
#     "batch_shape": "t h w c",
#     "dims": 2,
#     "multiGPU": 1,
#     "NF": {
#         "name": "SIRENAutodecoder_film",
#         "kwargs": {"num_hidden_layers": 10, "out_features": 4},
#     },
#     "lr": {"nf": 1.0e-4, "latents": 1.0e-5},
# }
# ri.TrainParamReader.update_defaults(RI_PARAMS)

from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from einops import rearrange

from cnf.utils.normalize import Normalizer_ts
from cnf.utils import readdata
from cnf import nf_networks
from functools import partial


class basic_set(Dataset):
    def __init__(self, fois, coord) -> None:
        super().__init__()
        self.fois = fois
        self.coords = coord

    def __len__(self):
        return self.fois.shape[0]

    def __getitem__(self, idx):
        return self.coords, self.fois[idx], idx


class LatentContainer(torch.nn.Module):
    """
    a model container that stores latents for multi GPU
    """

    def __init__(
        self,
        N_samples,
        N_features,
        dims,
    ):
        super().__init__()
        self.expand_dims = " ".join(["1" for _ in range(dims)])
        self.expand_dims = f"N f -> N {self.expand_dims} f"
        self.latents = torch.nn.Parameter(
            torch.zeros((N_samples, N_features), dtype=torch.float32)
        )

    def forward(self, batch_ids):
        return rearrange(self.latents[batch_ids], self.expand_dims)


def rMAE(prediction, target, dims=(1, 2)):
    return torch.abs(prediction - target).mean(dim=dims) / torch.abs(target).mean(
        dim=dims
    )


class trainer(object):

    def __init__(self, hyper_para: ri.basic_input, infer_mode=False, infer_dps=False) -> None:
        '''
        Initialize the training module for the Conditional Neural Field model.
        For propogating gradient through the model (e.g. in DPS), set self.nf.eval() 
        Args:
            hyper_para (basic_input): The hyperparameters for the model.
            infer_mode (bool, optional): Flag indicating whether the module is in inference mode. Defaults to False.
                
        '''
        self.world_size = hyper_para.multiGPU

        if not infer_mode:
            ###### read data - fois ######
            if hasattr(hyper_para, "load_data_fn"):
                if type(hyper_para.load_data_fn) == str:
                    load_data_fn = getattr(readdata, hyper_para.load_data_fn)
                    load_params = {}
                elif type(hyper_para.load_data_fn) == dict:
                    load_data_fn = getattr(readdata, hyper_para.load_data_fn["name"])
                    load_params = hyper_para.load_data_fn["kwargs"]
                fois = load_data_fn(hyper_para.data_path, **load_params)

            else:
                fois = np.load(f"{hyper_para.data_path}")

            assert (
                rearrange(
                    fois,
                    f"{hyper_para.readin_data_shape} -> {hyper_para.readin_data_shape}",
                )
                == fois
            ).all(), f"data shape is {tuple(fois.shape)}, which is inconsistant \
            with the fois_shape ({hyper_para.readin_data_shape}) specified in yaml file."

            fois = rearrange(
                fois, f"{hyper_para.readin_data_shape} -> {hyper_para.batch_shape}"
            )

            if "kwargs" in hyper_para.NF:
                assert (
                    hyper_para.NF["kwargs"]["out_features"] == fois.shape[-1]
                ), "NF_out_features is not consistent with fois shape"
            else:
                assert (
                    hyper_para.NF["out_features"] == fois.shape[-1]
                ), "NF_out_features is not consistent with fois shape"

            self.spatio_shape = fois.shape[1:-1]
            self.spatio_axis = list(
                range(fois.ndim if isinstance(fois, np.ndarray) else fois.dim())
            )[1:-1]

            ###### read data - coordinate ######
            if hasattr(hyper_para, "coor_path"):
                coord = np.load(f"{hyper_para.coor_path}")
                assert (
                    coord.shape[:-1] == self.spatio_shape
                ), "coordinate shape is not consistent with fois shape"
                assert (
                    coord.shape[-1] == hyper_para.dims
                ), "coordinate dimension is not consistent with dims in yaml file"
            else:
                Warning(
                    "No coordinate data is provided, using cartisian coordinate inferred from fois data"
                )
                coord = [np.linspace(0, 1, i) for i in self.spatio_shape]
                coord = np.stack(np.meshgrid(*coord, indexing="ij"), axis=-1)

            ###### convert to tensor ######
            fois = (
                torch.tensor(fois, dtype=torch.float32)
                if not isinstance(fois, torch.Tensor)
                else fois
            )
            self.N_samples = fois.shape[0]
            print(f"total training samples: {self.N_samples}")
            coord = (
                torch.tensor(coord, dtype=torch.float32)
                if not isinstance(coord, torch.Tensor)
                else coord
            )

        ###### normalizer ######
        self.in_normalizer = Normalizer_ts(**hyper_para.normalizer)
        self.out_normalizer = Normalizer_ts(**hyper_para.normalizer)

        if not exists(f"{hyper_para.save_path}") and (not infer_mode):
            mkdir(hyper_para.save_path)

        if exists(f"{hyper_para.save_path}/normalizer_params.pt"):
            print(
                f"loading normalizer parameters from {hyper_para.save_path}/normalizer_params.pt"
            )
            norm_params = torch.load(f"{hyper_para.save_path}/normalizer_params.pt")
            self.in_normalizer.params = norm_params["x_normalizer_params"]
            self.out_normalizer.params = norm_params["y_normalizer_params"]

        elif not infer_mode:
            print("No noramlization file found! Calculating normalizer parameters")
            self.in_normalizer.fit_normalize(coord.flatten(0, -2))
            self.out_normalizer.fit_normalize(fois.flatten(0, -2))

            print(
                f"Saving normalizer parameters to {hyper_para.save_path}/normalizer_params.pt"
            )
            torch.save(
                {
                    "x_normalizer_params": self.in_normalizer.get_params(),
                    "y_normalizer_params": self.out_normalizer.get_params(),
                },
                hyper_para.save_path + "/normalizer_params.pt",
            )
        else:
            raise FileNotFoundError(
                f"{hyper_para.save_path}/normalizer_params.pt does not exist"
            )

        if not infer_mode:
            normed_coords = self.in_normalizer.normalize(coord)
            normed_fois = self.out_normalizer.normalize(fois)

        ###### nf ######
        if "kwargs" not in hyper_para.NF:
            self.nf: torch.nn.Module = getattr(nf_networks, hyper_para.NF["name"])(
                in_coord_features=hyper_para.dims,
                in_latent_features=hyper_para.hidden_size,
                out_features=hyper_para.NF["out_features"],
                num_hidden_layers=hyper_para.NF["num_hidden_layers"],
                hidden_features=hyper_para.NF["hidden_features"],
            )
        else:
            self.nf: torch.nn.Module = getattr(nf_networks, hyper_para.NF["name"])(
                hyper_para.NF["kwargs"]
            )

        ###### latents ######
        if not infer_mode:
            self.latents = LatentContainer(
                self.N_samples, hyper_para.hidden_size, hyper_para.dims
            )

        self.hyper_para = hyper_para

        if not infer_mode:
            if hasattr(hyper_para, "dataset"):
                raise NotImplementedError

            else:
                self.dataset = basic_set(normed_fois, normed_coords)

            self.test_criteria = partial(
                getattr(sys.modules[__name__], hyper_para.test_criteria),
                dims=self.spatio_axis,
            )

        if infer_mode and not infer_dps:
            self.infer = torch.no_grad(self.infer)

    def infer(
        self,
        coord: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        coord = self.in_normalizer.normalize(coord)
        out = self.nf(coord, latents)
        return self.out_normalizer.denormalize(out)

    def train(self, fix_nf = False):
        self.epoches = self.hyper_para.epochs
        self.criterion = getattr(torch.nn, self.hyper_para.loss_fn)()
        self.lr = self.hyper_para.lr
        self.save_dict = {
            "save_path": self.hyper_para.save_path,
            "save_every": self.hyper_para.save_every,
        }
        optim_dict = self.optim_dict if hasattr(self, "optim_dict") else {}
        start_epoch = self.start_epoch if hasattr(self, "start_epoch") else 0

        if self.world_size > 1:
            print(f"Data Parallel training, using {self.world_size} GPUs")

            p = mp.spawn(
                self._single_trainer,
                args=(
                    self.nf,
                    self.latents,
                    self.criterion,
                    self.dataset,
                    self.hyper_para,
                    self.save,
                    self.world_size,
                    optim_dict,
                    start_epoch,
                    self.out_normalizer,
                    self.test_criteria,
                    fix_nf,
                ),
                nprocs=self.world_size,
                join=False,
            )
            p.join()

        else:
            print(f"Single GPU training")
            self._single_trainer(
                0,
                self.nf,
                self.latents,
                self.criterion,
                self.dataset,
                self.hyper_para,
                self.save,
                self.world_size,
                optim_dict,
                start_epoch,
                self.out_normalizer,
                self.test_criteria,
                fix_nf
            )

    @staticmethod
    def _single_trainer(
        rank,
        model,
        latents,
        criterion,
        dataset,
        hyper_para,
        savefn,
        world_size=1,
        optim_dict={},
        start_epoch=0,
        out_normalizer=None,
        test_criteria=None,
        fix_nf = False,
    ):

        model.to(rank)
        latents.to(rank)
        if world_size > 1:

            dist.init_process_group("nccl", rank=rank, world_size=world_size)

            model = DDP(model, device_ids=[rank])
            latents = DDP(latents, device_ids=[rank])

            train_loader = DataLoader(
                dataset,
                batch_size=hyper_para.batch_size,
                shuffle=False,
                sampler=DistributedSampler(dataset),
            )
            test_loader = DataLoader(
                dataset,
                batch_size=hyper_para.test_batch_size,
                shuffle=False,
                sampler=DistributedSampler(dataset),
            )
        else:
            train_loader = DataLoader(
                dataset, batch_size=hyper_para.batch_size, shuffle=True
            )
            test_loader = DataLoader(
                dataset, batch_size=hyper_para.test_batch_size, shuffle=False
            )

        if rank == 0:
            logger = SummaryWriter(hyper_para.save_path)

        optim_net_dec = torch.optim.Adam(model.parameters(), lr=hyper_para.lr["nf"])
        optim_states = torch.optim.Adam(
            latents.parameters(), lr=hyper_para.lr["latents"]
        )
        
        # scheduler_net_dec = torch.optim.lr_scheduler.ExponentialLR(optim_net_dec, gamma=0.999)#torch.optim.lr_scheduler.CosineAnnealingLR(optim_net_dec,T_max=100, eta_min=1e-6)
        # scheduler_states = torch.optim.lr_scheduler.ExponentialLR(optim_states, gamma=0.999)#torch.optim.lr_scheduler.CosineAnnealingLR(optim_states, T_max=100, eta_min=1e-6)

        if len(optim_dict) > 0:
            optim_net_dec.load_state_dict(optim_dict["optim_net_dec_dict"])
            optim_states.load_state_dict(optim_dict["optim_states_dict"])
        
        if fix_nf: model.eval()
        
        for i in tqdm(range(start_epoch, start_epoch + hyper_para.epochs)):
            if world_size > 1:
                train_loader.sampler.set_epoch(i)
            if i != 0 and not fix_nf:
                optim_net_dec.step()
                optim_net_dec.zero_grad()
            train_ins_error = []

            for batch_coords, batch_fois, idx in train_loader:

                batch_latent = latents(idx)
                batch_coords = batch_coords.to(rank)
                batch_fois = batch_fois.to(rank)

                batch_output = model(batch_coords, batch_latent)
                loss = criterion(batch_output, batch_fois)
                optim_states.zero_grad()
                loss.backward()
                optim_states.step()
                train_ins_error.append(loss)

            epoch_loss = torch.stack(train_ins_error).mean()  # .to(rank)
            
            # scheduler_net_dec.step()
            # scheduler_states.step()
            
            if world_size > 1:
                torch.distributed.reduce(
                    epoch_loss, op=torch.distributed.ReduceOp.AVG, dst=0
                )
            if rank == 0:
                mean_loss = epoch_loss.item()
                tqdm.write(str(mean_loss))
                logger.add_scalar("loss", mean_loss, i)

            if i % hyper_para.save_every == 0:  # and i > 0:

                test_error = []
                model.eval()
                latents.eval()
                with torch.no_grad():
                    if world_size > 1:
                        test_loader.sampler.set_epoch(i)
                    for test_coords, test_fois, idx in test_loader:
                        prediction = out_normalizer.denormalize(
                            model(test_coords.to(rank), latents(idx))
                        )
                        target = out_normalizer.denormalize(test_fois.to(rank))
                        error = test_criteria(prediction, target)
                        test_error.append(error.detach())

                test_error = torch.cat(test_error).mean(dim=0)
                if world_size > 1:
                    torch.distributed.reduce(
                        test_error, op=torch.distributed.ReduceOp.AVG, dst=0
                    )
                if not fix_nf: model.train()
                latents.train()

                if rank == 0:
                    for k in range(test_error.shape[-1]):
                        logger.add_scalar(f"test_error_{k}", test_error[k], i)

                    if world_size > 1:
                        savefn(
                            model.module, latents.module, i, optim_net_dec, optim_states
                        )
                    else:
                        savefn(model, latents, i, optim_net_dec, optim_states)

    def save(self, model, latents, epoch, optim_nf, optim_latent):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_states_dict": optim_latent.state_dict(),
                "optim_net_dec_dict": optim_nf.state_dict(),
                "hidden_states": latents.state_dict(),
                "epoch": epoch,
            },
            f"{self.hyper_para.save_path}/checkpoint_{epoch}.pt",
        )

    def load(self, checkpoint_id: int):

        if checkpoint_id == -1:
            import glob

            checkpoint_list = glob.glob(f"{self.hyper_para.save_path}/checkpoint_*.pt")
            checkpoint_list = [
                int(i.split("_")[-1].split(".")[0]) for i in checkpoint_list
            ]
            checkpoint_id = max(checkpoint_list)

        print(
            f"loading checkpoint from {self.hyper_para.save_path}/checkpoint_{checkpoint_id}.pt"
        )

        checkpoint = torch.load(
            f"{self.hyper_para.save_path}/checkpoint_{checkpoint_id}.pt"
        )
        self.nf.load_state_dict(checkpoint["model_state_dict"])
        if hasattr(self, "N_samples"):
            assert self.N_samples == checkpoint["hidden_states"]["latents"].shape[0]
        else:
            self.N_samples = checkpoint["hidden_states"]["latents"].shape[0]
        self.latents = LatentContainer(
            self.N_samples, self.hyper_para.hidden_size, self.hyper_para.dims
        )
        self.latents.load_state_dict(checkpoint["hidden_states"])

        self.optim_dict = {
            k: checkpoint[k] for k in ["optim_net_dec_dict", "optim_states_dict"]
        }
        self.start_epoch = checkpoint["epoch"]
        return self.nf, self.latents, self.optim_dict, checkpoint["epoch"]


if __name__ == "__main__":
    import sys
    import os

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    hp = ri.basic_input(sys.argv[1])
    mytrainer = trainer(hp)
    # mytrainer.load(-1)
    mytrainer.train()