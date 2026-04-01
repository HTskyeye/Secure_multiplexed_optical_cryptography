# Secure Multiplexed Optical Cryptography via a Diffractive Neural Network

We introduce a defense-enhanced architecture that integrates a free-space optical neural network for ciphertext generation. The system’s core innovation lies in the integration of blank channel into the plaintext ensemble, which fundamentally reshapes the cryptographic solution space to be hyper-sensitive to key perturbations, thereby counteracting brute-force attacks. Through rigorous simulation and experimental validation, our system achieves remarkable decryption fidelity across eight multiplexed valid channels. Extensive cryptanalysis, including known-plaintext attacks and key space enumeration, confirms the system’s robustness, revealing that the proposed defense-enhanced strategy significantly reduces information leakage.

## Inovation

- Neural-augmented optical framework
- Multi-channel encryption
- Blank channel integration

![hhhhh](.\images\Fig1.png)

## Usage

1. Encryption and decryption process

   Run `./Multiplexed_optical_cryptography/main.py` to train the network

   Run `./Multiplexed_optical_cryptography/eval.py` to evaluate the network

2. Ciphertext-only attack

   Run `./Ciphertext_only_attack/retrieval.m` to simulate ciphertext-only attack



