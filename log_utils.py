import logging
import time

logger = logging.getLogger()

def init_logger(level='info', log_file=None):
    str2level = {'info': logging.INFO,
                 'debug': logging.DEBUG}
    logger = logging.getLogger()
    logger.setLevel(str2level[level])
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def user_friendly_time_since(sec):
    s = int(time.time()) - int(sec)
    if s < 60:
        return "{}s".format(s)

    m = s // 60
    s = s % 60
    if m < 60:
        return "{}m {}s".format(m, s)

    h = m // 60
    m = m % 60
    if h < 24:
        return "{}h {}m {}s".format(h, m, s)

    d = h // 24
    h = h % 24
    return "{}d {}h {}m {}s".format(d, h, m, s)
