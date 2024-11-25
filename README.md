University project work for master course "**Project Deep Learning für selbstfahrendes Kraftfahrzeug**"

In this project we used imitation learning to train an agent for solving the OpenAI Gym CarRacing environment.
The agent was based on a custom convolutional network which was trained on manually generated samples from
the environment. The base model was later compressed with different pruning approaches and the fine-tuned
pruned networks were tested in the environment. We consistently achieved comparable results to the base model
even when parameter count was reduced by 80%. Our best model achieved an average test reward of 854 (with a
standard deviation of 52) over 20 episodes in the environment.

Ulm University Winter Semester 20/21
