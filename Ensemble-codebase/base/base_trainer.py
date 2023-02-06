import torch
import torch.nn
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from utils import load_state_dict, rename_parallel_state_dict

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.device_ids = device_ids
        self.model = model
        self.model = self.model.to(self.device)

        self.real_model = self.model
        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        self.criterion = criterion.to(self.device)
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.load_crt is not None:
            print("Loading from cRT pretrain: {}".format(config.load_crt))
            self._load_crt(config.load_crt)

        if config.resume is not None:
            state_dict_only = config._config.get("resume_state_dict_only", False)
            self._resume_checkpoint(config.resume, state_dict_only=state_dict_only)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
            elif best:
                self._save_checkpoint(epoch, save_best=True, best_only=True)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False, best_only=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config,
            'criterion': self.criterion.state_dict()
        }
        if not best_only:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...".format(best_path))

    def _load_crt(self, cRT_pretrain):
        """
        Load from cRT pretrain
        :param cRT pretrain path to the checkpoint of cRT pretrain
        """
        state_dict = torch.load(cRT_pretrain)['state_dict']
        ignore_linear = True

        rename_parallel_state_dict(state_dict)
        
        if ignore_linear:
            for k in list(state_dict.keys()):
                if k.startswith('backbone.linear'):
                    state_dict.pop(k)
                    print("Popped", k)
        load_state_dict(self.real_model, state_dict)
        for name, param in self.real_model.named_parameters():
            if not name.startswith('backbone.linear'):
                param.requires_grad_(False)
            else:
                print("Allow gradient on:", name)
        print("** Please check the list of allowed gradient to confirm **")

    def _resume_checkpoint(self, resume_path, state_dict_only=False):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)

        if not state_dict_only:
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            
            if 'monitor_best' in checkpoint:
                self.mnt_best = checkpoint['monitor_best']

            # load architecture params from checkpoint.
            if checkpoint['config']['arch'] != self.config['arch']:
                self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                    "checkpoint. This may yield an exception while state_dict is being loaded.")
        
        state_dict = checkpoint['state_dict']
        if state_dict_only:
            rename_parallel_state_dict(state_dict)

        # self.model.load_state_dict(state_dict)
        load_state_dict(self.model, state_dict)

        if not state_dict_only:
            if 'criterion' in checkpoint:
                load_state_dict(self.criterion, checkpoint['criterion'])
                self.logger.info("Criterion state dict is loaded")
            else:
                self.logger.info("Criterion state dict is not found, so it's not loaded.")

            # load optimizer state from checkpoint only when optimizer type is not changed.
            if 'optimizer' in checkpoint:
                if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
                    self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                        "Optimizer parameters not being resumed.")
                else:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
