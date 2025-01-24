# langchain-robot-chatbot
A chatbot for generating robot actions using LangChain, Streamlit and Isaac Sim.

## Requirements

#### Stacks
<p>
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white">
<img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white">
</p>

<p>
<img src="https://img.shields.io/badge/OpenAi-412991?style=for-the-badge&logo=openai&logoColor=white">
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/ROS-22314E?style=for-the-badge&logo=ros&logoColor=white">
<img src="https://img.shields.io/badge/IsaacSim-76B900?style=for-the-badge&logo=nvidia&logoColor=white">
</p>

For building and running the application you need:

- [python 3.12.8](https://www.python.org/)
- [Miniconda](https://docs.anaconda.com/miniconda/)

## Setup
### Conda create by requirements.txt

```sh
conda create -n langchain-robot-chatbot python==3.12.8
conda activate langchain-robot-chatbot
```

### Conda env create by environment.yml
```sh
conda env create -n langchain-robot-chatbot -f environment.yml
conda activate langchain-robot-chatbot
```

## Installation

```sh
git clone https://github.com/namaewa-im/langchain-robot-chatbot.git
cd langchain-robot-chatbot
pip install -r requirements.txt
```
