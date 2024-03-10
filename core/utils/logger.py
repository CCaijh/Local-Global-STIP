import logging
import os
import sys
import os.path as osp
def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if if_train:
            fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='w')
        else:
            fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


'''
    #初始化
    from utils.logger import setup_logger
    logger = setup_logger("reid_baseline", output_dir, if_train=True)    
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)
    
    #其他文件
    import logging
    logger = logging.getLogger("reid_baseline.train")
    logger.info('start training')
    

'''