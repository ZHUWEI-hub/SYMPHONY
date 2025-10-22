
# SYMPHONY: Synergistic Multi-agent Planning with Heterogeneous Language Model Assembly

This repository is the official implementation of [SYMPHONY: Synergistic Multi-agent Planning with Heterogeneous Language Model Assembly
](https://openreview.net/forum?id=7Spt8cAJq0&noteId=7Spt8cAJq0). 

![Overview](./img/overview.png)

# Requirements
Each of the three tasks corresponds to a separate folder: hotpot, webshop, and programming.
For each task, the implementation code and the required dependencies (requirements.txt) are located within the respective folder.
The entry point for running each program is the run.py file inside its corresponding folder. All three tasks are developed and tested with Python 3.10.

## Model Configuration
- If you are using a locally deployed model, make sure the model is set up in advance and provides an OpenAI-compatible API. Tools like [FastChat](https://github.com/lm-sys/FastChat) can be used as a reference for local deployment.
- If you are using a remote API-based model, ensure that the API endpoint and the API key are properly prepared.

Once the model is ready, configure it in the model.py file located in each task folder.

## HotpotQA

#### Setup

To get started:

1. Clone this repo and move to the HotPotQA directory:
```bash
cd SYMPHONY/hotpot
```

2. Install the module dependencies into your environment:
```bash
pip install -r requirements.txt
```

3. In the model.py file of each task, configure the model's API endpoint and API key accordingly.

4. Set the scripts and run paper experiments
```bash
python run.py
```

- ``--n_generate_sample``: number of times to prompt during expansion/sampling
- ``--iterations``: maximum number of trajectories to sample


## WebShop

#### Setup

To get started:

1. Clone this repo and move to the WebShop directory:
```bash
cd SYMPHONY/webshop
```

2. Install WebShop from source and run environment instance locally. Follow the instructions here (https://github.com/princeton-nlp/WebShop)

3. Install the module dependencies into your environment:
```bash
pip install -r requirements.txt
```

4. In the model.py file of each task, configure the model's API endpoint and API key accordingly.


5. Change localhost in lats.py to your local port running WebShop

6. Set the scripts and run paper experiments
```bash
python run.py
```

- ``--n_generate_sample``: number of times to prompt during expansion/sampling
- ``--iterations``: maximum number of trajectories to sample




## MBPP
#### Setup

To get started:

1. Clone this repo and move to the HotPotQA directory:
```bash
cd SYMPHONY/programming
```

2. Install the module dependencies into your environment:
```bash
pip install -r requirements.txt
```

3. In the model.py file of each task, configure the model's API endpoint and API key accordingly.

4. Set the scripts and run paper experiments
```bash
python run.py
```

- ``--n_generate_sample``: number of times to prompt during expansion/sampling
- ``--iterations``: maximum number of trajectories to sample


## Acknowledgments

This project builds upon and benefits from the following open-source projects:

- **[ReAct](https://github.com/ysymyth/ReAct)**: Synergizing Reasoning and Acting in Language Models
- **[Reflexion](https://github.com/noahshinn/reflexion)**: Language Agents with Verbal Reinforcement Learning
- **[LATS](https://github.com/lapisrocks/LanguageAgentTreeSearch)**: Language Agent Tree Search Unifies Reasoning Acting and Planing in Language Models

We are grateful for their excellent work and open-source contributions.

