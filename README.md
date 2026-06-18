# lewm-flow

Attempts to use flow-matching to steer the LeWM encoder and predictor to adjust to multi-embodiment and multi-task settings where the flow matching action denoiser is task and embodiment-specific but the encoder and predictor are independent of it.

## Ideas

1. The data quality needs to be further improved, particularly in the context of the reward for tactile movement
2. Currently we only learn from actual teleoperated data and not able to learn from human data in LeWorldModel, maybe that can also be possible at scale using VLMs (VLA-JEPA).
3. Optical flow, motion tracking and DINO representations are a way to get human and robot videos close to each other.
4. Unisteer and the combined approach has something interesting, we can inject specific kinds of features into the model from the flow matching head just by the presence of it.
5. And arguably, the flow matching action head can be used to steer not just for embodiments but also for tasks across different layers of the network rather than just projecting the trajectories.
6. Flow matching steering the model is almost the same as context engineering but from the inside of the model.
7. The SFT dataset should be as diverse as possible even at the expense of the full robustness of the base model.
8. As long as the flow matching one can mould it into whatever we want there's no issue.
9. LoRA does the exact same thing by injecting small, trainable rank decomposition matrices.
10. So potentially the flow matching model could feed into some kind of LoRA before going to the actual model?
11. We're currently using pycapacity, but there's other kinds of signals like pointnet, clip, etc.
12. The world model should have absolutely nothing to do with the reward of the state, that should solely be assessed by the flow matcher training/tuning.
13. We need more evidence for knowing that the world model won't suffer from the sim-to-real map, maybe relying on DINO entirely isn't that bad after all.
14. The world model needs to be as crude as possible while being useful as a verifier for remaining untouched regardless of embodiment, sim-to-real, etc. making use of perturbations.
15. The flow matching model should be trained with a clear reward focused on whether we've accomplished the task while ensuring progressive rewards for improvement.
16. The flow matching model is essentially playing the role of the \pi that translates high-level goals into low-level actions, and arguably also steering vectors.
17. The flow matching model kind of replaces the MSAT in RLDX-1.
18. SimDist focuses on control-loop decoupling between behaviour planning and environment dynamics to fix the Sim-to-Real gap whereas VLA-JEPA focuses on the Human-to-Robot gap by decoupling semantics and physics.
19. The predictor could be like inferix with block diffusion rather than completely autoregressive and the steering from the flow matching action denoiser could go to specific blocks.
20. The information steered could be about future possibilities as well of how the scene could evolve, like World Pilot.
