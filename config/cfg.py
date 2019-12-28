from collections import OrderedDict

cfg = OrderedDict()
cfg['use_gpu'] = True
cfg['scut_fbp_dir'] = '/home/xulu/DataSet/SCUT-FBP/Crop'
cfg['hotornot_dir'] = '/home/xulu/DataSet/eccv2010_beauty_data/hotornot_face'
cfg['cv_index'] = 1
cfg['scutfbp5500_base'] = '/home/xulu/DataSet/SCUT-FBP5500'

cfg['batch_size'] = 64
cfg['epoch'] = 400
cfg['random_seed'] = 40
