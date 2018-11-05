{
    "env": "round2",
    "skip": 3,
    "random_start": 0,
    "gamma": 0.95,
    "horizon": 334,
    "model": {
        "fcnet_hiddens": [512],
        "fcnet_activation": "relu",
        "fcnet_layer_normalization": true
    },
    "actor_hiddens": [256],
    "actor_hidden_activation": "relu",
    "actor_layer_normalization": true,
    "critic_hiddens": [256],
    "critic_hidden_activation": "relu",
    "critic_layer_normalization": true,
    "n_step": 1,
    "schedule_max_timesteps": 2000000,
    "timesteps_per_iteration": 1000,
    "exploration_fraction": 0.1,
    "exploration_final_eps": 0.02,
    "lr_decay": true,
    "noise_scale": 2.5,
    "param_noise": true,
    "exploration_theta": 0.15,
    "exploration_sigma": 0.2,
    "target_network_update_freq": 1000,
    "tau": 1.0,
    "buffer_size": 6000000,
    "prioritized_replay": true,
    "prioritized_replay_alpha": 0.6,
    "prioritized_replay_beta": 0.4,
    "prioritized_replay_eps": 1e-6,
    "compress_observations": false,
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "use_huber": false,
    "huber_threshold": 1.0,
    "l2_reg": 1e-6,
    "grad_norm_clipping": 40.0,
    "learning_starts": 200000,
    "train_batch_size": 1024,
    "sample_batch_size": 25,
    "max_weight_sync_delay": 50,
    "stop_criteria": 9900
}
