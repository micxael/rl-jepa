import random
import time

import numpy as np
import torch

from . import sac as sac
from .sac.sac_base import SACReplayBuffer
from .shared_utils import config_env
from .VPG_training_base import pretrain_SAS_buffer
from ..environments.utils import Space
from ..utils.logger import EpochLogger


def mask_observation(obs, mask_percent):
    num_elements = obs.numel()
    num_elements_to_mask = int(num_elements * mask_percent)
    mask_indices = torch.randperm(num_elements)[:num_elements_to_mask]
    masked_obs = obs.clone().view(-1)  
    masked_obs[mask_indices] = 0 
    return masked_obs.view_as(obs)



def SAC_embed(cnf):
    if "threads" in cnf.keys():
        torch.set_num_threads(cnf["threads"])
        print("num threads used", cnf["threads"])

    # Setup the logger
    logger = EpochLogger(output_dir=cnf["logger"]["output_dir"])
    logger.save_config(config=locals(), cnf=cnf)

    torch.manual_seed(cnf["seed"])
    np.random.seed(cnf["seed"])
    random.seed(cnf["seed"])

    env, obs_dim, act_dim = config_env(cnf)
    test_env, obs_dim, act_dim = config_env(cnf)

    sequence_length = int(cnf["training"]["num_obs_samples"])
    masking_percentage = int(cnf["training"]["masking_percentage"])

    # If we use custom Spaces, we want .n instead of .shape
    if isinstance(env.action_space, Space):
        act_dim = env.action_space.n

    cnf_train = cnf["training"]

    # Create actor-critic module
    ac_fn = getattr(sac, cnf["model"]["ac_fn"])
    ac = ac_fn(env.observation_space, env.action_space, cnf)

    # Set up the experience buffer
    cnf_train = cnf["training"]

    buffer = SACReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_emb_dim=cnf["embed"]["a_embed_dim"],
        size=cnf_train["buffer_size"],
        sequence_length = sequence_length,
    )

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    start_time = time.time()
    episode_num = 0
    total_env_interacts = 0
    ep_ret_all = []

    # To catch potential value error in the training process without terminating the parent process
    try:
        # Set up all the pre-training stuff and run pre-training if enabled
        if (
            cnf["embed"]["embed_module"] == "SASEmbeddingModule"
            or cnf["embed"]["embed_module"] == "AutoEncoderEmbeddingModule"
        ):
            pretrain_SAS_buffer(
                cnf, obs_dim, act_dim, cnf_train, ac, episode_num, env, logger
            )
        else:
            raise ValueError("Unknown embedding module given!")

        for epoch in range(cnf_train["num_epochs"]):
            episode_num, total_env_interacts = run_sac_epoch(
                episode_num,
                total_env_interacts,
                ep_ret_all,
                cnf_train["steps_per_epoch"],
                env,
                ac,
                logger,
                buffer,
                cnf_train,
                sequence_length,
                masking_percentage,
            )
            ac.update(buffer, logger)
            for j in range(cnf_train["num_test_episodes"]):
                o, d, ep_ret_test, ep_len_test = test_env.reset(), False, 0, 0
                obs_sequence = np.tile(o, (sequence_length, 1))
                while not (d or (ep_len_test == cnf_train["max_ep_len"])):
                    for i in range(obs_sequence.shape[0]):
                        obs_sequence[i] = mask_observation(torch.from_numpy(obs_sequence[i]),masking_percentage).numpy()
                    flattened_sequence = obs_sequence.flatten()
                    a, action_embed = ac.test_step(
                        torch.as_tensor(flattened_sequence, dtype=torch.float32)
                    )
                    next_o, r, d, _ = test_env.step(a)
                    ep_ret_test += r
                    ep_len_test += 1

                    obs_sequence = np.roll(obs_sequence, -1, axis=0)
                    obs_sequence[-1] = next_o
                    o = next_o
                logger.store(TestEpRet=ep_ret_test, TestEpLen=ep_len_test)
            if epoch % cnf["logger"]["log_every_n_epochs"] == 0:
                # Log info about epoch
                logger.log_tabular("Epoch", epoch)
                logger.log_tabular("Episodes", episode_num)
                logger.log_tabular("EpRet", with_min_and_max=True)
                logger.log_tabular("EpLen", average_only=True)
                logger.log_tabular("TestEpRet", with_min_and_max=True)
                logger.log_tabular("TestEpLen", average_only=True)
                logger.log_tabular("TotalEnvInteracts", total_env_interacts)
                logger.log_tabular("LossPi", average_only=True)
                logger.log_tabular("LossQ", average_only=True)
                logger.log_tabular("QVals", with_min_and_max=True)
                logger.log_tabular("Time", time.time() - start_time)
                logger.dump_tabular()

            # Save model
            if (epoch % cnf_train["save_freq"] == 0) or (
                epoch == cnf_train["num_epochs"] - 1
            ):
                logger.save_state({"env": env}, None)

        ac.save_embeddings(logger, "_done")
    except Exception as e:
        print(e)
        return 0, 0
    return np.mean(ep_ret_all[-100:]), np.mean(ep_ret_all)


def run_sac_epoch(
    episode_num,
    total_env_interacts,
    ep_ret_all,
    steps_per_epoch,
    env,
    ac,
    logger,
    buffer,
    cnf_train,
    sequence_length,
    masking_percentage,
):
    """
    Runs an episode in the environment and stores the data.
    The agent may be updated in the process as per the configuration (update every n steps).
    """
    
    o, ep_ret, ep_len = env.reset(), 0, 0
    obs_sequence = np.tile(o, (sequence_length, 1))
    for t in range(steps_per_epoch):
            
        obs_sequence = np.roll(obs_sequence, -1, axis=0)
        obs_sequence[-1] = o
        for i in range(obs_sequence.shape[0]):
            obs_sequence[i] = mask_observation(torch.from_numpy(obs_sequence[i]),masking_percentage).numpy()
            
        flattened_sequence = obs_sequence.flatten()

        action, action_embed = ac.step(
            torch.as_tensor(flattened_sequence, dtype=torch.float32),
            episode_num,
            buffer,
            logger,
        )
        next_o, r, d, _ = env.step(action)
        ep_ret += r
        ep_len += 1
        d = False if ep_len == cnf_train["max_ep_len"] else d
        buffer.store(
            obs=flattened_sequence,
            act=action,
            act_emb=action_embed,
            rew=r,
            next_obs=next_o,
            done=d,
        )
        o = next_o

        timeout = ep_len == cnf_train["max_ep_len"]
        terminal = d or timeout
        epoch_ended = t == steps_per_epoch - 1
        if terminal or epoch_ended:
            if terminal:
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                episode_num += 1
            total_env_interacts += ep_len
            ep_ret_all.append(ep_ret)
            o, ep_ret, ep_len = env.reset(), 0, 0

    return episode_num, total_env_interacts


def run_sac_embed(cnf):
    SAC_embed(cnf)
