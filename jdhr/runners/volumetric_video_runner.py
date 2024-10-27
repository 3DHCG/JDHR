# Default runner, no fancy business here
# Perform the training loop and log stuff out (tensor board etc.)
# Also responsible for saving the model and the optimizer states
# Sometimes performs validation and also writes things to tensorboard

# For type annotation
import time
import jittor as jt
import datetime

from jdhr.engine import cfg, args  # need this for initialization?
from jdhr.runners.schedulers import ExponentialLR
from jdhr.runners.recorders import TensorboardRecorder
from jdhr.runners.moderators import DatasetRatioModerator
from jdhr.runners.optimizers import ConfigurableOptimizer, Adam
from jdhr.models.volumetric_video_model import VolumetricVideoModel
from jdhr.dataloaders.datasets.volumetric_video_dataset import VolumetricVideoDataset
from jdhr.runners.evaluators.volumetric_video_evaluator import VolumetricVideoEvaluator
from jdhr.runners.visualizers.volumetric_video_visualizer import VolumetricVideoVisualizer
from jdhr.engine import RUNNERS, OPTIMIZERS, SCHEDULERS, RECORDERS, VISUALIZERS, EVALUATORS, MODERATORS  # controls the optimization loop of a particular epoch

from jdhr.utils.console_utils import *
from jdhr.utils.timer_utils import timer
from jdhr.utils.base_utils import dotdict
from jdhr.utils.data_utils import add_iter, to_cuda
from jdhr.utils.net_utils import save_model, load_model, load_network, save_npz


# The outer most training loop sets lr scheduler, constructs objects etc
# The inner loop call training, logs stuff


@RUNNERS.register_module()
class VolumetricVideoRunner:  # a plain and simple object controlling the training loop
    def __init__(self,
                 model: VolumetricVideoModel,  # the network to train
                 dataset: VolumetricVideoDataset, # enumerate this
                 val_dataset: VolumetricVideoDataset, # enumerate this
                 optimizer_cfg: dotdict = dotdict(type=ConfigurableOptimizer.__name__),
                 scheduler_cfg: dotdict = dotdict(type=ExponentialLR.__name__),

                 moderator_cfg: dotdict = dotdict(type=DatasetRatioModerator.__name__),
                 recorder_cfg: dotdict = dotdict(type=TensorboardRecorder.__name__),
                 visualizer_cfg: dotdict = dotdict(type=VolumetricVideoVisualizer.__name__),
                 evaluator_cfg: dotdict = dotdict(type=VolumetricVideoEvaluator.__name__),

                 epochs: int = 400,  # total: ep_iter * epoch number of iterations
                 decay_epochs: int = -1,  # if -1, use epochs, else give user more control
                 ep_iter: int = 500,  # number of iterations per epoch
                 eval_ep: int = 1,  # report validation stats
                 save_ep: int = 10,  # separately save networks (might be heavy on storage)
                 save_lim: int = 3,  # only this number of files will be kept
                 empty_cache_ep: int = 1e10,  # neven empty cache
                 save_latest_ep: int = 1,  # just in case, save regularly
                 log_interval: int = 1,  # 10ms, tune this if in realtime
                 record_interval: int = 1,  # ?ms, tune this if in realtime
                 strict: bool = True,  # strict loading of network and modules?

                 resume: bool = True,
                 test_only: bool = False,
                 exp_name: str = cfg.exp_name,  # name of the experiment
                 pretrained_model: str = '',  # load this model first
                 trained_model: str = f'data/trained_model/{cfg.exp_name}',  # MARK: global configuration
                 load_epoch: int = -1,  # load different epoch to start with

                 clip_grad_norm: float = -1,  # 1e-3,
                 clip_grad_value: float = -1,  # 40.0,
                 retain_last_grad: bool = False,  # setting this to true might lead to excessive VRAM usage
                 ignore_eval_error: bool = True,  # errors in evaluation will not affect training
                 record_images_to_tb: bool = True,  # when testing, save images to tensorboard
                 print_test_progress: bool = False,  # when testing, print a progress bar for indication
                 test_using_train_mode: bool = False,  # when testing, call model.train() instead of model.eval()
                 test_using_inference_mode: bool = args.type != 'train',  # MARK: global configuration

                 test_amp_cached: bool = True,
                 train_use_amp: bool = False,
                 test_use_amp: bool = False,
                 use_jit_trace: bool = False,  # almost will never work
                 use_jit_script: bool = False,  # almost will never work
                 use_torch_compile: bool = False,  # almost will never work
                 #  torch_compile_mode: str = None,
                 torch_compile_mode: str = 'max-autotune-no-cudagraphs',
                 # Debugging
                 collect_timing: bool = False,  # will lose 1 fps over copying
                 timer_sync_cuda: bool = True,  # will explicitly call torch.cuda.synchronize() before collecting
                 timer_record_to_file: bool = False,  # will write to a json file for collected analysis of the timing
                 ):
        self.model = model  # possibly already a ddp model?

        # Used in evaluation
        if not jt.rank:  # only build these in main process
            self.val_dataset = val_dataset # different dataloader for validation
            self.evaluator: VolumetricVideoEvaluator = EVALUATORS.build(evaluator_cfg)
            self.visualizer: VolumetricVideoVisualizer = VISUALIZERS.build(visualizer_cfg)
            self.recorder: TensorboardRecorder = RECORDERS.build(recorder_cfg, resume=resume)

        if not test_only:
            self.dataset = dataset
            self.optimizer: Adam = OPTIMIZERS.build(optimizer_cfg, named_params=((k, v) for k, v in model.named_parameters() if v.requires_grad))  # requires parameters
            self.scheduler: ExponentialLR = SCHEDULERS.build(scheduler_cfg, optimizer=self.optimizer, decay_iter=(epochs if decay_epochs < 0 else decay_epochs) * ep_iter)  # requires parameters
            self.moderator: DatasetRatioModerator = MODERATORS.build(moderator_cfg, runner=self, total_iter=epochs * ep_iter)  # after dataset init

        self.exp_name = exp_name
        self.epochs = epochs
        self.ep_iter = ep_iter
        self.eval_ep = eval_ep
        self.save_ep = save_ep
        self.save_lim = save_lim
        self.empty_cache_ep = empty_cache_ep
        self.save_latest_ep = save_latest_ep
        self.log_interval = log_interval
        self.record_interval = record_interval

        self.resume = resume
        self.strict = strict
        self.load_epoch = load_epoch
        self.trained_model = trained_model
        self.pretrained_model = pretrained_model

        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value
        self.retain_last_grad = retain_last_grad


        # Trace model for faster inference
        self.use_jit_script = use_jit_script
        self.use_jit_trace = use_jit_trace
        self.use_torch_compile = use_torch_compile
        self.torch_compile_mode = torch_compile_mode

        self.ignore_eval_error = ignore_eval_error
        self.record_images_to_tb = record_images_to_tb
        self.print_test_progress = print_test_progress
        self.test_using_train_mode = test_using_train_mode
        self.test_using_inference_mode = test_using_inference_mode

        # Setting VRAM limit on Windows might make the framerate more stable
        #try:
            #torch.cuda.set_per_process_memory_fraction(torch_vram_frac_limit)  # set vram usage limit to current device
        #except:
        #    pass

        # HACK: GLOBAL VARIABLE, when dumping config, should ignore this one
        cfg.runner = self
        # cfg\..* = # this search will find all global config assignment
        # We need to perform the dumping before the global config to keep things clean

        # Debugging
        self.collect_timing = collect_timing  # another fancy self.timer (different from fps counter)
        self.timer_sync_cuda = timer_sync_cuda  # this enables accurate time recording for each section, but would slow down the programs
        self.timer_record_to_file = timer_record_to_file

    @property
    def collect_timing(self):
        return not timer.disabled

    @property
    def timer_sync_cuda(self):
        return timer.sync_cuda

    @property
    def timer_record_to_file(self):
        return timer.record_to_file

    @collect_timing.setter
    def collect_timing(self, val: bool):
        timer.disabled = not val

    @timer_sync_cuda.setter
    def timer_sync_cuda(self, val: bool):
        timer.sync_cuda = val

    @timer_record_to_file.setter
    def timer_record_to_file(self, val: bool):
        timer.record_to_file = val
        if timer.record_to_file:
            log(yellow(f'Will record timing results to {blue(join(self.recorder.record_dir, f"{self.exp_name}.json"))}'))
            timer.exp_name = self.exp_name
            timer.record_dir = self.recorder.record_dir
            if not hasattr(timer, 'timing_record'):
                timer.timing_record = dotdict()

    @property
    def total_iter(self):
        return self.epochs * self.ep_iter

    def load_network(self):
        if self.pretrained_model:  # maybe load pretrain model
            epoch = load_network(model=self.model,  # only loading the network, without recorder?
                                 model_dir=self.pretrained_model,
                                 strict=self.strict,
                                 )  # loads the next epoch to use

        epoch = load_network(model=self.model,  # only loading the network, without recorder?
                             model_dir=self.trained_model,
                             resume=self.resume,
                             epoch=self.load_epoch,
                             strict=self.strict,
                             )  # loads the next epoch to use
        return epoch

    def load_model(self):
        if self.pretrained_model:  # maybe load pretrain model
            epoch = load_model(model=self.model,
                               optimizer=self.optimizer,
                               scheduler=self.scheduler,
                               moderator=self.moderator,
                               model_dir=self.pretrained_model,
                               strict=self.strict,
                               )  # loads the next epoch to use
        epoch = load_model(model=self.model,
                           optimizer=self.optimizer,
                           scheduler=self.scheduler,
                           moderator=self.moderator,
                           model_dir=self.trained_model,
                           resume=self.resume,
                           epoch=self.load_epoch,
                           strict=self.strict,
                           )  # loads the next epoch to use
        return epoch

    def save_network(self, epoch, latest: bool = True, **kwargs):
        try:
            save_model(model=self.model,
                       model_dir=self.trained_model,
                       save_lim=self.save_lim,
                       epoch=epoch,
                       latest=latest,
                       **kwargs,
                       )
        except RuntimeError as e:
            log(red(e))
            jt.gc()

    def save_npz(self, epoch, latest: bool = True, **kwargs):
        try:
            save_npz(model=self.model,
                     model_dir=self.trained_model,
                     epoch=epoch,
                     **kwargs,
                     )
        except RuntimeError as e:
            log(red(e))
            jt.gc()

    def save_model(self, epoch: int, latest: bool = True, **kwargs):
        try:
            save_model(model=self.model,
                       optimizer=self.optimizer,
                       scheduler=self.scheduler,
                       moderator=self.moderator,
                       model_dir=self.trained_model,
                       save_lim=self.save_lim,
                       epoch=epoch,
                       latest=latest,
                       **kwargs,
                       )
        except RuntimeError as e:
            log(red(e))
            jt.gc()

    # Single epoch testing api
    def test(self):  # from begin epoch
        epoch = self.load_network()
        self.test_epoch(epoch)

    # Epoch based runner
    def train(self):  # from begin epoch
        epoch = self.load_model()

        # The actual training for this epoch
        train_generator = self.train_generator(epoch, self.ep_iter)  # yield every ep iter

        # train the network
        for epoch in range(epoch, self.epochs):

            # Possible to make this a decorator?
            next(train_generator)  # avoid reconstruction of the dataloader

            # Leave some breathing room for other applications
            if (epoch + 1) % self.empty_cache_ep == 0:
                log(green(f'Emptying cuda memory cache'))
                jt.gc()

            # Saving stuff to disk
            #if (epoch + 1) % self.save_ep == 0 and not jt.rank:   #no need
            #    try:
            #        self.save_model(epoch, latest=False)
            #    except Exception as e:
            #        log(red('Error in model saving, ignored and continuing'))
            #        stacktrace()
            #        stop_prog()  # stop it, otherwise multiple lives

            if (epoch + 1) % self.save_latest_ep == 0 and not jt.rank:
                try:
                    self.save_model(epoch, latest=True)
                    self.save_npz(epoch, latest=True)  # for inference and smaller file size
                except Exception as e:
                    log(red('Error in model saving, ignored and continuing'))
                    stacktrace()
                    stop_prog()  # stop it, otherwise multiple lives

            # Perform validation run if required
            if (epoch + 1) % self.eval_ep == 0 and not jt.rank:
                try:
                    self.test_epoch(epoch + 1)  # will this provoke a live display?
                except Exception as e:
                    log(red('Error in validation pass, ignored and continuing'))
                    stacktrace()
                    stop_prog()  # stop it, otherwise multiple lives
                    if not self.ignore_eval_error:
                        raise e

    def train_epoch(self, epoch: int):
        train_generator = self.train_generator(epoch, self.ep_iter)
        for _ in train_generator: pass  # the actual calling

    # Iteration based runner
    def train_generator(self, begin_epoch: int, yield_every: int = 1):
        # Train for one epoch (iterator style)
        # Actual start of the execution
        epoch = begin_epoch  # set starting epoch

        self.model.train()  # set the network (model) to training mode (recursive to all modules)
        for k, v in self.model.named_parameters():
            v.requires_grad=True
        start_time = time.perf_counter()
        for index, batch in enumerate(self.dataset):  # control number of iterations explicitly
            iter = begin_epoch * self.ep_iter + index  # epoch actually is only for logging here
            batch = add_iter(batch, iter, self.total_iter)  # is this bad naming

            batch = to_cuda(batch)  # cpu -> cuda

            data_time = time.perf_counter() - start_time
            timer.record('data transfer')

            # Model forwarding
            #with torch.cuda.amp.autocast(enabled=self.train_use_amp):  # maybe perform AMP
            output: dotdict = self.model(batch)  # random dict storing various forms of output
            loss: jt.Var = output.loss.mean()  # final optimizable loss variable to backward
            image_stats: dotdict = output.image_stats  # things to report to recorder (all image tensors (float32))
            scalar_stats: dotdict = output.scalar_stats  # things to report to logger and recorder (all scalars)
            timer.record('model forwarding')

            # Optimization step

            if hasattr(self.model, 'decorate_grad'): self.model.decorate_grad(self, batch)  # perform some action on the gradients
            elif hasattr(self.model, 'module') and hasattr(self.model.module, 'decorate_grad'): self.model.module.decorate_grad(self, batch)  # perform some action on the gradients

            
            self.optimizer.step(loss)
            self.scheduler.step()
            self.moderator.step()
            
            timer.record('optimization step')

            # Records data and batch forwarding time
            end_time = time.perf_counter()
            batch_time = end_time - start_time
            start_time = end_time  # note that all logging and profiling time are accumuated into data_time
            if (iter + 1) % self.log_interval == 0 and not jt.rank:

                # For recording onto the tensorboard
                scalar_stats = dotdict(
                    {
                        k: (v.mean().item() if isinstance(v, jt.Var) or isinstance(v, np.ndarray) else v)
                        for k, v in scalar_stats.items()
                    }
                )  # MARK: sync (for accurate batch time)

                lr = self.optimizer.param_groups[0]['lr']  # TODO: skechy lr query, only lr of the first param will be saved

                scalar_stats.data = data_time
                scalar_stats.batch = batch_time
                scalar_stats.lr = lr


                self.recorder.iter = iter
                self.recorder.epoch = epoch
                self.recorder.update_scalar_stats(scalar_stats)
                if self.record_images_to_tb: self.recorder.update_image_stats(image_stats)  # NOTE: recording images is slow

                # For logging onto the console
                eta_seconds = self.recorder.scalar_stats.batch.global_avg * (self.total_iter - self.recorder.iter)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                log_stats = dotdict()
                log_stats.eta = eta_string
                log_stats.update(self.recorder.log_stats)

                # Render table to screen
                display_table(log_stats)  # render dict as a table (live, console, table)

                # Actually uploading information to tensorboard if needed
                if (iter + 1) % self.record_interval == 0:
                    try: self.recorder.record(self.dataset.split.name)  # actual writing to the tensorboard logger
                    except Exception as e:
                        log(red('Error in recording, ignored and continuing'))
                        stacktrace()
                        stop_prog()  # stop it, otherwise multiple lives

            # Maybe save model or perform validation
            if yield_every > 0 and (iter + 1) % yield_every == 0:
                yield output
                self.model.train()

            # Actual start of the execution
            if (iter + 1) % self.ep_iter == 0: epoch = epoch + 1

            # Do profiling if appicable
            timer.record('logging & recording')
        jt.gc()

    def test_epoch(self, epoch: int):
        test_generator = self.test_generator(epoch, -1)  # nevel yield (special logic)
        for _ in test_generator: pass  # the actual calling

    def test_generator(self, epoch: int, yield_every: int = 1):
        # validation for one epoch

        self.model.eval()#train(self.test_using_train_mode)  # set the network (model) to training mode (recursive to all modules)
        for index, batch in enumerate(tqdm(self.val_dataset, disable=not self.print_test_progress)):
            iter = epoch * self.ep_iter - 1  # some indexing trick
            batch = add_iter(batch, iter, self.total_iter)  # is this bad naming
            batch = to_cuda(batch)  # cpu -> cuda, note that DDP will move all cpu tensors to cuda as well
            if batch.meta.camera_index%20 != 0: # eval on every 20 cameras
                continue
            with jt.no_grad():
                a=time.perf_counter()
                output: dotdict = self.model(batch)
                b=time.perf_counter()
                print("FPS:",1/(b-a))
                scalar_stats = self.evaluator.evaluate(output, batch)
                image_stats = self.visualizer.visualize(output, batch)

            self.recorder.iter = iter
            self.recorder.epoch = epoch
            self.recorder.update_scalar_stats(scalar_stats)
            if self.record_images_to_tb: self.recorder.update_image_stats(image_stats)
            self.recorder.record(self.val_dataset.split.name + "_" + "FRAME")  # per frame records, to make things cleaner

            if yield_every > 0 and (iter + 1) % yield_every == 0:
                # break  # dataloader could be infinite
                yield output
                self.model.eval()

            #profiler_step()  # record a step for the profiler, extracted logic
        scalar_stats = self.evaluator.summarize()
        #image_stats = self.visualizer.summarize()
        self.recorder.update_scalar_stats(scalar_stats)
        #if self.record_images_to_tb: self.recorder.update_image_stats(image_stats)
        self.recorder.record(self.val_dataset.split.name)
