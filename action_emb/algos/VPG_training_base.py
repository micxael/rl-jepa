import torch
from action_emb.algos.vpg_utils.buffer import VPGFlexBuffer
import os
import numpy as np

def mask_observation(obs, mask_percent):
    num_elements = obs.numel()
    num_elements_to_mask = int(num_elements * mask_percent)
    mask_indices = torch.randperm(num_elements)[:num_elements_to_mask]
    masked_obs = obs.clone().view(-1)  
    masked_obs[mask_indices] = 0 
    return masked_obs.view_as(obs)


def run_epoch(
    episode_num,
    total_env_interacts,
    ep_ret_all,
    steps_per_epoch,
    env,
    ac,
    logger,
    buffer,
    cnf_train,
):
    """
    Runs an episode in the environment and stores the data.
    The agent may be updated in the process as per the configuration (update every n steps).
    """
    terminal = False
    timeout = False
    o, ep_ret, ep_len = env.reset(), 0, 0
    for t in range(steps_per_epoch):
        a, v, logp, action_embed = ac.step(
            torch.as_tensor(o, dtype=torch.float32), episode_num, buffer, logger
        )
        next_o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        buffer.store(o, a, r, next_o, d, logp, v, action_embed)
        logger.store(VVals=v)
        o = next_o

        if cnf_train["max_ep_len"] is not None:
            timeout = ep_len == cnf_train["max_ep_len"]
        else:
            timeout = False
        terminal = d or timeout
        epoch_ended = t == steps_per_epoch - 1
        if terminal or epoch_ended:
            if timeout or epoch_ended:
                _, v, _, _ = ac.step(
                    torch.as_tensor(o, dtype=torch.float32), episode_num, buffer, logger
                )
            else:
                v = 0
            buffer.finish_path(v)
            if terminal:
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                episode_num += 1
            total_env_interacts += ep_len
            ep_ret_all.append(ep_ret)
            o, ep_ret, ep_len = env.reset(), 0, 0

    return episode_num, total_env_interacts


def run_pretrain_episode(episode_num, env, ac, logger, buffer, cnf_train):
    terminal = False
    timeout = False
    sequence_length = int(cnf_train["num_obs_samples"])
    masking_percentage = float(cnf_train["masking_percentage"])
    o, ep_ret, ep_len = env.reset(), 0, 0
    obs_sequence = np.tile(o, (sequence_length, 1))
    while not terminal:
        for i in range(obs_sequence.shape[0]):
            obs_sequence[i] = mask_observation(torch.from_numpy(obs_sequence[i]),masking_percentage).numpy()
        flattened_sequence = obs_sequence.flatten()
        a = ac.step(
            torch.as_tensor(flattened_sequence, dtype=torch.float32), episode_num, buffer, logger
        )
        next_o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        obs_sequence = np.roll(obs_sequence, -1, axis=0)
        obs_sequence[-1] = next_o

        buffer.store(flattened_sequence, a, r, next_o, d, 0, 0, 0)
        o = next_o

        timeout = ep_len == cnf_train["max_ep_len"]
        terminal = d or timeout
    buffer.finish_path(0)

    return ep_ret, ep_len


def pretrain_SAS_buffer(cnf, obs_dim, act_dim, cnf_train, ac, episode_num, env, logger):
    if cnf["embed"]["load_emb_path"] and os.path.exists(cnf["embed"]["load_emb_path"]):
        print("Embedding path for loading given:", cnf["embed"]["load_emb_path"])
        ac.load_embeddings()

    if cnf["embed"]["pre_train"]["is_on"]:
        print("------Pre-training embedder...")
        cnf_pretrain = cnf["embed"]["pre_train"]
        # Note: Use this buffer here as it is easier to use with the embedder for pre-training
        pre_train_buffer = VPGFlexBuffer(obs_dim, act_dim, cnf_pretrain["samples"])
        ac.embedder_pretrain = True
        while len(pre_train_buffer) < cnf_pretrain["samples"]:
            run_pretrain_episode(
                episode_num, env, ac, logger, pre_train_buffer, cnf_train
            )
        ac.pretrain_embedder_update(pre_train_buffer, logger)
        ac.embedder_pretrain = False
        print("------Pre-training of embedder done------")
    
    if cnf["embed"]["save_pretrain_embeddings"]:
        ac.save_embeddings(logger)
        print("------Embeddings have been saved------")
