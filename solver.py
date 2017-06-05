import os
import symbol
from datetime import datetime


class Solver(object):
    def __init__(self, config):
        try:
            self.config = config
            ''' environment parameters
            '''
            self.use_cpu = config.getboolean('env', 'use_cpu')
            self.gpus = config.get('env', 'gpus')
            self.kv_store = config.get('env', 'kv_store')
            self.ctx = mx.cpu() if self.use_cpu is True else [
                mx.gpu(int(i)) for i in self.gpus.split(',')]
            self.multi_thread = config.getboolean('env', 'multi_thread')

            ''' network parameters
            '''
            self.network = config.get('network', 'network')
            self.label_num = config.getint('network', 'label_num')
            self.aspp = config.getint('network', 'aspp')
            self.aspp_stride = config.getint('network', 'aspp_stride')
            self.cell_width = config.getint('network', 'cell_width')
            self.ignore_label = config.getint('network', 'ignore_label')
            self.atrous_type = config.get('network', 'atrous_type')
            self.relu_type = config.get('network', 'relu_type')
            self.bn_use_global_stats = config.getboolean('network', 'bn_use_global_stats')

            ''' model parameters
            '''
            self.num_epochs = config.getint('model', 'num_epochs')
            self.model_dir = config.get('model', 'model_dir')
            self.save_model_prefix = config.get('model', 'save_model_prefix')
            self.checkpoint_interval = config.getint('model', 'checkpoint_interval')
            # SGD parameters
            self.lr = config.getfloat('model', 'lr')
            self.lr_policy = config.get('model', 'lr_policy')
            self.lr_factor = config.getfloat('model', 'lr_factor')
            self.lr_factor_epoch = config.getfloat('model', 'lr_factor_epoch')
            self.momentum = config.getfloat('model', 'momentum')
            self.weight_decay = config.getfloat('model', 'weight_decay')
            # fine tuning
            self.load_model_dir = config.get('model', 'load_model_dir')
            self.load_model_prefix = config.get('model', 'load_model_prefix')
            self.load_epoch = config.getint('model', 'load_epoch')

            ''' data parameters
            '''
            self.data_dir = config.get('data', 'data_dir')
            self.label_dir = config.get('data', 'label_dir')
            self.train_list = config.get('data', 'train_list')
            self.use_val = config.getboolean('data', 'use_val')
            if self.use_val:
                self.val_list = config.get('data', 'val_list')
            self.batch_size = config.getint('data', 'batch_size')
            self.ds_rate = config.getint('data', 'ds_rate')

            # TODO: training tricks
            # self.scale_factors = [float(scale.strip()) for scale in config.get('data', 'scale_factors').split(',')]
            # self.crop_shape = tuple([int(l.strip()) for l in config.get('data', 'crop_shape').split(',')])
            # self.use_mirror = config.getboolean('data', 'use_mirror')
            # self.use_random_crop = config.getboolean('data', 'use_random_crop')
            # self.use_color_cast = config.getboolean('data', 'use_color_cast')
            if self.use_color_cast:
                self.color_cast_range = [int(r.strip()) for r in config.get('data', 'color_cast_range').split(',')]
            else:
                self.color_cast_range = None
            self.random_bound = tuple([int(l.strip()) for l in config.get('data', 'random_bound').split(',')])

            ''' inference
            '''
            self.train_size = 0
            with open(self.train_list, 'r') as f:
                for line in f:
                    self.train_size += 1
            self.epoch_size = self.train_size / self.batch_size
            self.data_shape = [tuple(list([self.batch_size, 3, self.crop_shape[0], self.crop_shape[1]]))]
            self.label_shape = [tuple([self.batch_size, (self.crop_shape[1]*self.crop_shape[0]/self.cell_width**2)])]
            self.data_name = ['data']
            self.label_name = ['seg_loss_label']
            self.symbol = None
            self.arg_params = None
            self.aux_params = None

        except ValueError:
            print 'Config parameter error'

    # build up symbol, parameters and auxiliary parameters
    def get_model(self):
        # get symbol from symbol.py
        self.symbol = symbol.get_symbol(self, self.label_num, self.ignore_label, self.aspp, self.aspp_stride,
                                        self.atrous_type, self.bn_use_global_stats, self.relu_type)

        # load model
        if self.load_model_prefix is not None and self.load_epoch > 0:
            self.symbol, self.arg_params, self.aux_params = \
                mx.model.load_checkpoint(os.path.join(self.load_model_dir, self.load_model_prefix), self.load_epoch)

    def fit(self):
        # kvstore
        if self.kv_store is 'local' and (
                self.gpus is None or len(self.gpus.split(',')) is 1):
            kv = None
        else:
            kv = mx.kvstore.create(self.kv_store)

        # setup module, including symbol, params and aux
        self.get_model()

        optimizer_params = {}
        # sgd optimizer
        if self.lr_factor < 1 and self.lr_factor_epoch > 0:
            optimizer_params['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
                step=max(int(self.epoch_size * self.lr_factor_epoch), 1),
                factor=self.lr_factor)
        optimizer_params['learning_rate'] = self.lr
        optimizer_params['momentum'] = self.momentum
        optimizer_params['wd'] = self.weight_decay
        optimizer_params['rescale_grad'] = 1.0 / self.batch_size

        # directory for saving models
        model_path = os.path.join(self.model_dir, self.save_model_prefix)
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        model_full_path = os.path.join(model_path, datetime.now().strftime('%Y_%m_%d_%H:%M:%S'))
        if not os.path.isdir(model_full_path):
            os.mkdir(model_full_path)

        module = mx.module.Module(self.symbol, context=self.ctx, data_names=self.data_name, label_names=self.label_name)

        # module.fit(
        #     train_data=train_data,
        #     kvstore=kv,
        #     optimizer=self.optimizer,
        #     optimizer_params=optimizer_params,
        #     initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        #     arg_params=self.arg_params,
        #     aux_params=self.aux_params,
        #     allow_missing=True,
        #     begin_epoch=self.load_epoch,
        #     num_epoch=self.num_epochs,
        # )
        # self.symbol.save('%s-symbol.json' % os.path.join(model_full_path, self.save_model_prefix))
