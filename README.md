\documentclass[12pt,a4paper]{article}
vfdvdfvfdffdfdf
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{subcaption}

\title{Enhanced Decision Transformer for Offline Reinforcement Learning}
\author{Your Name}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
This project builds upon the Decision Transformer framework for offline Reinforcement Learning (RL), introducing architectural enhancements and experiments on custom Atari datasets. We modify both the Transformer-based policy (in \texttt{gym/decision\_transformer/models/}) and a baseline MLP model (in \texttt{gym/decision\_transformer/models/mlp\_bc.py}) to test whether these improvements yield better performance. Additionally, we leverage datasets from Atari environments, such as the \texttt{ALEGalaxian-v5} dataset (found in the \texttt{atari/} directory), to evaluate the model's ability to generalize. 

This document provides an overview of the enhancements, datasets, training procedure, and preliminary results.
\end{abstract}

\section{Introduction}
The Decision Transformer~\cite{chen2021decision} reframes RL as a sequence modeling problem. Given a desired return-to-go, it uses a Transformer model (inspired by GPT architectures) to predict actions that achieve this return. The state, action, and return tokens are stacked and fed into a causal transformer.

Our contributions are:
\begin{itemize}
    \item **Enhanced MLP Baseline:** In \texttt{gym/decision\_transformer/models/mlp\_bc.py}, we increase the depth, add normalization layers, and switch to smoother activation functions (e.g., GELU) to improve stability and accuracy.
    \item **Deeper Decision Transformer:** In \texttt{gym/decision\_transformer/models/decision\_transformer.py}, we add more layers and attention heads, incorporate dropout and additional layer normalization, and adjust hyperparameters to handle complex datasets.
    \item **Custom Atari Datasets:** Under \texttt{atari/}, we experiment with datasets like \texttt{ALEGalaxian-v5}. By modifying preprocessing and segmenting trajectories differently, we test whether the improved models can generalize better than the original Decision Transformer.
\end{itemize}

\section{Background}
Consider a trajectory:
\[
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T, a_T, r_T),
\]
where $r_t$ is the reward at time $t$, and the return-to-go from time $t$ is:
\[
R_t = \sum_{t'=t}^T r_{t'}.
\]

The Decision Transformer conditions on $(R_{t}, s_t, a_{t-1}, \ldots)$ to predict $a_t$. Stacking the tokens as $(R_1, s_1, a_1, R_2, s_2, a_2, \dots)$, we use a causal self-attention mask so that the model only attends to past tokens:
\[
p(a_t | R_1, s_1, a_1, \ldots, R_t, s_t) = \text{Transformer}(R_{\le t}, s_{\le t}, a_{< t}).
\]

\section{Enhancements in the \texttt{gym} and \texttt{atari} Directories}
The directory structure is as follows:
\begin{itemize}
    \item \texttt{atari/}: Scripts and code for Atari data handling, including `ALEGalaxian-v5` environment datasets.
    \item \texttt{gym/}: Implementations of the Decision Transformer and MLP BC models, training scripts, and utilities for loading custom datasets.
\end{itemize}

\subsection{Enhanced MLP Model}
The MLP model now features:
\begin{itemize}
    \item Multiple hidden layers with \texttt{GELU} activation:
    \[
    x_{l+1} = \text{GELU}(W_l x_l + b_l),
    \]
    offering smoother gradients and potentially better convergence.
    \item Layer normalization and dropout between layers to reduce internal covariate shift and overfitting.
\end{itemize}

\subsection{Enhanced Decision Transformer}
The Transformer model includes:
\begin{itemize}
    \item More transformer layers (e.g., $n_{\text{layer}}=6$) and attention heads to increase representational capacity.
    \item Dropout layers in embeddings and output representations to improve generalization.
    \item Additional \texttt{LayerNorm} applied after the transformer block and before predictions.
\end{itemize}

\section{Custom Atari Datasets}
The Atari datasets, such as \texttt{ALEGalaxian-v5}, provide a range of trajectories. We:
\begin{itemize}
    \item Modify trajectory splitting and return distributions.
    \item Introduce variable-length trajectories and altered state pre-processing.
    \item Evaluate whether enhancements help the model adapt to these changes and achieve desired returns more consistently.
\end{itemize}

\section{Training and Preliminary Results}
Training is done via a \texttt{Trainer} class:
\[
\text{loss} = \|a_{\text{pred}} - a_{\text{target}}\|_2^2,
\]
where $a_{\text{pred}}$ is the predicted action and $a_{\text{target}}$ is the ground truth from the dataset.

Early experiments show:
\begin{itemize}
    \item Faster convergence and lower losses with the enhanced MLP.
    \item More stable training with the deeper Decision Transformer on modified Atari datasets.
\end{itemize}

Further evaluation and tuning are ongoing.

\section{Conclusion}
By enhancing both the MLP baseline and the Decision Transformer architecture, and by testing on modified Atari datasets, we aim to push the limits of sequence-modeling-based RL. Initial results are promising, and ongoing work will focus on further optimization and evaluation.

\bibliographystyle{plain}
\begin{thebibliography}{1}

\bibitem{chen2021decision}
Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch.
\newblock Decision Transformer: Reinforcement Learning via Sequence Modeling.
\newblock {\em arXiv preprint arXiv:2106.01345}, 2021.

\end{thebibliography}

\end{document}
