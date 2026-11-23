
"""
Roda as simulações do campeonato e salva os resultados.

Inclui:
- comparação entre as etapas do modelo;
- simulação final com BMA;
- análise dos jogos restantes;
- cenários de rebaixamento e Libertadores;
- testes de sensibilidade dos parâmetros.

Uso: python analise_v2.py --n 1000000
"""

import argparse
import csv
import json
import os
import time
from collections import Counter

import numpy as np
import simular_v2 as S
from simular_v2 import Cfg, rodar, TIMES, IDX, ALVO, RESTANTES


os.makedirs("resultados", exist_ok=True)

# Número de jogos usados no backtest para calcular os pesos do BMA
NBT = 220

# Parâmetros usados como referência nas simulações
CENTRAL = {
    "sigma": 0.02,
    "tau": 0.18,
    "rho": -0.15,
    "nu": 1.4,
}

inicio = time.time()


def log(*args):
    """Mostra o tempo decorrido junto com a mensagem."""
    tempo = time.time() - inicio
    print(f"[{tempo:6.0f}s]", *args, flush=True)


def cfgs_bma(beta=0.5, **overrides):
    """Monta as configurações e seus pesos a partir do backtest."""
    with open("backtest3_config.json", encoding="utf-8") as arquivo:
        configs = json.load(arquivo)

    scores = np.array([c["logscore"] * NBT for c in configs])
    pesos = np.exp(beta * (scores - scores.max()))
    pesos /= pesos.sum()

    resultado = []

    for config, peso in zip(configs, pesos):
        if peso <= 2e-4:
            continue

        modelo = Cfg(
            sigma=config["sigma"],
            tau=config["tau"],
            rho=config["rho"],
            nu=config["nu"],
            **overrides,
        )
        resultado.append((modelo, peso))

    return resultado


def p_rel(resultado):
    """Probabilidade de terminar entre os quatro últimos."""
    caiu = resultado["rank"] >= 16
    return caiu.mean(axis=0)


def ic(p, n):
    """Intervalo aproximado de confiança para a estimativa."""
    return 1.96 * np.sqrt(p * (1 - p) / n)


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1_000_000)
parser.add_argument("--nl", type=int, default=200_000)
args = parser.parse_args()

R = {}
