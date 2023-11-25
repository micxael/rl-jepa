# RL-JEPA


## Execution
```
// Suggested using a conda environment

conda create -n "xxxxx" python=3.10

pip install -r requirements.txt

python setup.py build && python setup.py install

python bin/run_sac_embed configs/template_sac_embed.yml
python bin/run_vpg_embed configs/template_vpg_embed.yml
python bin/run_ppo_embed configs/template_ppo_embed.yml
```

## Reference

This code has been adapted from that of Pritz et al., 2021
