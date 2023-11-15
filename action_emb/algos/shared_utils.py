import gym
import action_emb.environments.all_envs_imported as envs
import torch
import torch.utils.data as du
from gym.spaces import Box, Discrete
from action_emb.environments.utils import Space


def config_env(cnf):
    # Instantiate environment
    if cnf["env"]["own_env"]:
        env_fn = getattr(envs, cnf["env"]["env_fn"])
        env = env_fn(**cnf["env"]["env_config"])
    else:
        env = gym.make(cnf["env"]["env_fn"])
    env.seed(cnf["seed"])

    if isinstance(env.action_space, Box):
        act_dim = env.action_space.shape[0]
    elif isinstance(env.action_space, Discrete):
        act_dim = env.action_space.n
    elif isinstance(env.action_space, Space):
        act_dim = env.action_space.shape[0]
    else:
        raise ValueError("Action Space Class is not known")
    if isinstance(env.observation_space, Discrete):
        obs_dim = env.observation_space.n
    else:
        obs_dim = env.observation_space.shape[0]

    return env, obs_dim, act_dim


def config_optims(pi_params, vf_params, cnf_train):
    pi_optim_fn = getattr(torch.optim, cnf_train["pi_optim"]["class"])
    vf_optim_fn = getattr(torch.optim, cnf_train["vf_optim"]["class"])
    pi_optimizer = pi_optim_fn(
        pi_params, lr=cnf_train["pi_lr"], **cnf_train["pi_optim"]["args"]
    )
    vf_optimizer = vf_optim_fn(
        vf_params, lr=cnf_train["vf_lr"], **cnf_train["vf_optim"]["args"]
    )
    return pi_optimizer, vf_optimizer


def compute_loss_v(data, ac):
    obs, ret = data["obs"], data["ret"]
    pred = ac.v(obs)
    loss = ((pred - ret) ** 2).mean()
    return loss


def compute_loss_pi(data, ac, clip_ratio):
    obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]
    act = act.squeeze()

    # Policy loss
    pi, logp = ac.pi(obs, act)
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    ent = pi.entropy().mean().item()
    clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

    return loss_pi, pi_info


def create_dataloader(experience_dict, batch_size):
    print(experience_dict["act_emb"][0])
    if (
        not isinstance(experience_dict["act_emb"][0], torch.Tensor)
        and experience_dict["act_emb"][0] == False
    ):
        dataset = EpochDataset(
            obs=experience_dict["obs"],
            act=experience_dict["act"],
            adv=experience_dict["adv"],
            logp=experience_dict["logp"],
            ret=experience_dict["ret"],
        )
    else:
        dataset = EpochDataset(
            obs=experience_dict["obs"],
            act=experience_dict["act"],
            adv=experience_dict["adv"],
            logp=experience_dict["logp"],
            ret=experience_dict["ret"],
            a_embed=experience_dict["act_emb"],
        )

    return du.DataLoader(dataset, batch_size=batch_size)


class EpochDataset(du.Dataset):
    def __init__(self, obs, act, adv, logp, ret, a_embed=None, batch_first=True):
        self.batch_first = batch_first
        self.obs = obs
        self.act = act
        self.adv = adv
        self.logp = logp
        self.ret = ret
        self.a_embed = a_embed

    def __getitem__(self, item):
        if self.batch_first:
            obs = self.obs[item]
            act = self.act[item]
            adv = self.adv[item]
            logp = self.logp[item]
            ret = self.ret[item]
            if self.a_embed is not None:
                return (obs, act, adv, logp, ret, self.a_embed[item])
            else:
                return (
                    obs,
                    act,
                    adv,
                    logp,
                    ret,
                )
        else:
            obs = self.obs[:, item]
            act = self.act[:, item]
            adv = self.adv[:, item]
            logp = self.logp[:, item]
            ret = self.ret[item]
            if self.a_embed is not None:
                return (obs, act, adv, logp, ret, self.a_embed[:, item])
            else:
                return (
                    obs,
                    act,
                    adv,
                    logp,
                    ret,
                )

    def __len__(self):
        if isinstance(self.obs, list):
            return len(self.obs)
        if self.batch_first:
            return self.obs.shape[0]
        else:
            return self.obs.shape[1]
