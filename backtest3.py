
"""
Refino dos parâmetros do modelo e comparação com o backtest estático.

Testa diferentes valores de sigma, tau, rho e nu,
mantendo kappa em zero nesta etapa.
"""

import itertools
import json
import time

import numpy as np

from carregar import carregar
from dinamico import *
from dados_tabela import TABELA


# Dados e configuração inicial
jogos = carregar()
times = list(TABELA)
base = Base(jogos, times)

SIGMA = [0.005, 0.01, 0.02, 0.03]
TAU = [0.12, 0.18, 0.25]

# Combinações de rho, kappa e nu
COMBINACOES = [
    (rho, 0.0, nu)
    for rho in (-0.10, -0.15, -0.20)
    for nu in (1.2, 1.3, 1.4, 1.5)
]

# Janelas usadas no backtest
ORIGENS = list(range(130, 343, 10))

rng = np.random.default_rng(3)
resultados = {}
inicio = time.time()


for sigma, tau in itertools.product(SIGMA, TAU):
    metricas = {config: [[], [], []] for config in COMBINACOES}

    for origem in ORIGENS:
        estado = base.montar(origem)
        ajuste = ajustar(base, estado, sigma, sigma, tau, 0.0)

        jogos_teste = jogos[origem:origem + 10]
        previsoes = prever(
            base,
            ajuste,
            jogos_teste,
            COMBINACOES,
            rng=rng,
        )

        for config in COMBINACOES:
            logscore, logloss, rps = metricas_func = metricas_jogo(
                previsoes[config],
                jogos_teste,
            )

            valores = (logscore, logloss, rps)

            for i, valor in enumerate(valores):
                metricas[config][i].append(valor)

    for config in COMBINACOES:
        resultados[(sigma, tau) + config] = tuple(
            np.concatenate(valores)
            for valores in metricas[config]
        )


print(f"{len(resultados)} configurações em {time.time() - inicio:.0f}s")


# Comparação com o modelo estático
referencia = np.load("bt_ref_ls.npy")

linhas = []

for config, valores in resultados.items():
    logscore = valores[0].mean()
    logloss = valores[1].mean()
    rps = valores[2].mean()

    diferenca = (valores[0] - referencia).mean()
    erro = (valores[0] - referencia).std(ddof=1) / np.sqrt(len(referencia))

    linhas.append(
        (config, logscore, logloss, rps, diferenca, erro)
    )

linhas.sort(key=lambda linha: -linha[1])


print(
    f"{'sigma':>6}{'tau':>6}{'rho':>7}{'kappa':>7}"
    f"{'nu':>5}{'logscore':>10}{'logloss':>9}"
    f"{'RPS':>8}{'Δ vs estático':>16}"
)

for config, logscore, logloss, rps, diferenca, erro in linhas[:12]:
    sigma, tau, rho, kappa, nu = config

    print(
        f"{sigma:>6}{tau:>6}{rho:>7}{kappa:>7}{nu:>5}"
        f"{logscore:>10.4f}{logloss:>9.4f}{rps:>8.4f}"
        f"{diferenca:>+10.4f} ±{erro:.4f}"
    )


def mostrar_perfil(indice, nome, valores):
    print(f"\nPerfil de {nome}:")

    for valor in valores:
        melhor = max(
            (linha for linha in linhas if linha[0][indice] == valor),
            key=lambda linha: linha[1],
        )

        print(
            f"  {nome}={valor:<6} "
            f"logscore={melhor[1]:.4f}  "
            f"logloss={melhor[2]:.4f}  "
            f"RPS={melhor[3]:.4f}"
        )


mostrar_perfil(0, "sigma", SIGMA)
mostrar_perfil(1, "tau", TAU)
mostrar_perfil(2, "rho", (-0.10, -0.15, -0.20))
mostrar_perfil(4, "nu", (1.2, 1.3, 1.4, 1.5))


# Pesos pseudo-BMA
scores = np.array([linha[1] * len(referencia) for linha in linhas])

pesos = np.exp(scores - scores.max())
pesos /= pesos.sum()

configs_saida = []

for linha, peso in zip(linhas, pesos):
    config = linha[0]

    configs_saida.append(
        dict(
            sigma=config[0],
            tau=config[1],
            rho=config[2],
            kappa=config[3],
            nu=config[4],
            logscore=float(linha[1]),
            logloss=float(linha[2]),
            rps=float(linha[3]),
            peso=float(peso),
        )
    )

with open("backtest3_config.json", "w", encoding="utf-8") as arquivo:
    json.dump(configs_saida, arquivo, indent=1, ensure_ascii=False)


print("\nPesos pseudo-BMA (top 8):")

for linha, peso in zip(linhas[:8], pesos[:8]):
    sigma, tau, rho, kappa, nu = linha[0]

    print(
        f"  sigma={sigma} tau={tau} rho={rho} "
        f"kappa={kappa} nu={nu}  peso={peso:.3f}"
    )
