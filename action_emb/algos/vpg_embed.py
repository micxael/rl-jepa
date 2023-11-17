import numpy as np
import torch
import time
import random
from . import vpg_algos as pg
from action_emb.algos.vpg_utils.buffer import VPGFlexBuffer

from ..environments.utils import Space
from .shared_utils import config_env
from ..utils.logger import EpochLogger
from .VPG_training_base import run_epoch, pretrain_SAS_buffer


def VPG_embed(cnf):
    if 'threads' in cnf.keys():
        torch.set_num_threads(cnf['threads'])
        print("num threads used", cnf['threads'])

    # Setup the logger
    logger = EpochLogger(output_dir=cnf['logger']['output_dir'])
    logger.save_config(config=locals(), cnf=cnf)

    torch.manual_seed(cnf['seed'])
    np.random.seed(cnf['seed'])
    random.seed(cnf['seed'])

    env, obs_dim, act_dim = config_env(cnf)
    
    # If we use custom Spaces, we want .n instead of .shape
    if isinstance(env.action_space, Space):
        act_dim = env.action_space.n

    cnf_train = cnf['training']

    # Create actor-critic module
    ac_fn = getattr(pg, cnf['model']['ac_fn'])
    ac = ac_fn(env.observation_space, env.action_space, cnf) # VPGActorCriticEmbed

    # Set up the experience buffer
    buffer = VPGFlexBuffer(
        obs_dim, act_dim, cnf_train['buffer_size'], cnf_train['gamma'], cnf_train['lam']
    )

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    start_time = time.time()
    state, ep_ret, ep_len = env.reset(), 0, 0
    episode_num = 0
    total_env_interacts = 0
    ep_ret_all = []

    # To catch potential value error in the training process without terminating the parent process
    try:
        # Set up all the pre-training stuff and run pre-training if enabled
        if (
            cnf['embed']['embed_module'] == 'SASEmbeddingModule'
            or cnf['embed']['embed_module'] == 'AutoEncoderEmbeddingModule'
        ):
            pretrain_SAS_buffer(
                cnf, obs_dim, act_dim, cnf_train, ac, episode_num, env, logger
            )
        else:
            raise ValueError("Unknown embedding module given!")

        for epoch in range(cnf_train['num_epochs']):
            terminal = False
            episode_num, total_env_interacts = run_epoch(
                episode_num,
                total_env_interacts,
                ep_ret_all,
                cnf_train['steps_per_epoch'],
                env,
                ac,
                logger,
                buffer,
                cnf_train
            )

            ac.update(buffer, logger)

            if epoch % cnf['logger']['log_every_n_epochs'] == 0:
                # Log info about epoch
                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('Episodes', episode_num)
                logger.log_tabular('EpRet', with_min_and_max=True)
                logger.log_tabular('EpLen', average_only=True)
                logger.log_tabular('VVals', with_min_and_max=True)
                logger.log_tabular('TotalEnvInteracts', total_env_interacts)
                logger.log_tabular('LossPi', average_only=True)
                logger.log_tabular('LossV', average_only=True)
                logger.log_tabular('DeltaLossPi', average_only=True)
                logger.log_tabular('DeltaLossV', average_only=True)
                
                logger.log_tabular('Time', time.time() - start_time)
                logger.dump_tabular()
            # Save model
            if (epoch % cnf_train['save_freq'] == 0) or (epoch == cnf_train['num_epochs'] - 1):
                logger.save_state({'env': env}, None)

        # ac.save_embeddings(logger, "_done") # TODO: needed?
    except Exception as e:
        print(e)
        return 0, 0

    return np.mean(ep_ret_all[-100:]), np.mean(ep_ret_all)


def run_vpg_embed(cnf):
    VPG_embed(cnf)
