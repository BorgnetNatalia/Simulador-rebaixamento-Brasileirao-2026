
"""
Modelo dinâmico de gols para o Brasileirão.

A força de cada equipe varia ao longo da temporada, com um passeio
aleatório para ataque e defesa. A previsão considera também a incerteza
dos parâmetros e a evolução esperada até a próxima partida.

Os parâmetros são estimados por máxima a posteriori, usando Newton-Raphson
com matrizes esparsas. A Hessiana é usada para aproximar a covariância
posterior.

Os hiperparâmetros são avaliados por validação walk-forward.
"""

import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import splu
from scipy.special import gammaln

from dist import pmf_conjunta


# Placar de 0 a 12 gols
NG = 13


class Base:
    """Indexa os times e organiza os jogos para montar o modelo."""

    def __init__(self, jogos, times):
        self.times = list(times)
        self.n = len(self.times)
        self.idx = {time: i for i, time in enumerate(self.times)}
        self.jogos = jogos

    def montar(self, m):
        """Monta a matriz do modelo usando os m primeiros jogos."""
        jogos = self.jogos[:m]
        n = self.n

        # Número de partidas já disputadas por cada equipe
        cont = np.zeros(n, dtype=int)
        kc = np.zeros(m, dtype=int)
        kf = np.zeros(m, dtype=int)

        casa = np.array([self.idx[g["casa"]] for g in jogos])
        fora = np.array([self.idx[g["fora"]] for g in jogos])

        for r in range(m):
            kc[r] = cont[casa[r]]
            cont[casa[r]] += 1

            kf[r] = cont[fora[r]]
            cont[fora[r]] += 1

        # Posição inicial dos estados de ataque e defesa
        acumulado = np.concatenate([[0], np.cumsum(cont)[:-1]])

        offA = 2 + n + acumulado
        offD = 2 + n + cont.sum() + acumulado

        P = 2 + n + 2 * cont.sum()

        gols_casa = np.array([g["gc"] for g in jogos])
        gols_fora = np.array([g["gv"] for g in jogos])

        rows, cols, vals = [], [], []

        def add(linha, coluna, valor):
            rows.append(linha)
            cols.append(coluna)
            vals.append(valor)

        for r in range(m):
            linha_casa = 2 * r
            linha_fora = 2 * r + 1

            # Gols da equipe da casa
            add(linha_casa, 0, 1)
            add(linha_casa, 1, 1)
            add(linha_casa, 2 + casa[r], 1)

            add(linha_casa, offA[casa[r]] + kc[r], 1)
            add(linha_casa, offD[fora[r]] + kf[r], -1)

            # Gols da equipe visitante
            add(linha_fora, 0, 1)
            add(linha_fora, offA[fora[r]] + kf[r], 1)
            add(linha_fora, offD[casa[r]] + kc[r], -1)

        X = sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(2 * m, P),
        )

        y = np.empty(2 * m)
        y[0::2] = gols_casa
        y[1::2] = gols_fora

        return {
            "X": X,
            "y": y,
            "P": P,
            "cont": cont,
            "offA": offA,
            "offD": offD,
            "ic": casa,
            "iff": fora,
            "kc": kc,
            "kf": kf,
            "m": m,
        }

    def precisao(self, d, sig_a, sig_d, tau, s_h):
        """Monta a matriz de precisão dos priors dos parâmetros."""
        n = self.n
        cont = d["cont"]
        offA = d["offA"]
        offD = d["offD"]
        P = d["P"]

        rows, cols, vals = [], [], []
        linha_atual = 0

        def adicionar_linha(colunas, valores):
            nonlocal linha_atual

            for coluna, valor in zip(colunas, valores):
                rows.append(linha_atual)
                cols.append(coluna)
                vals.append(valor)

            linha_atual += 1

        # Priors de mu e do efeito geral de mando
        adicionar_linha([0], [0.2])
        adicionar_linha([1], [0.2])

        # Efeito de mando específico de cada equipe
        for i in range(n):
            adicionar_linha(
                [2 + i],
                [1 / max(s_h, 1e-3)],
            )

        # Priors iniciais e evolução dos estados de ataque e defesa
        for i in range(n):
            for offset, sigma in ((offA[i], sig_a), (offD[i], sig_d)):
                adicionar_linha([offset], [1 / tau])

                for k in range(1, cont[i]):
                    adicionar_linha(
                        [offset + k - 1, offset + k],
                        [-1 / sigma, 1 / sigma],
                    )

        L = sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(linha_atual, P),
        )

        return (L.T @ L).tocsc()


def ajustar(base, d, sig_a, sig_d, tau=0.35, s_h=0.0, iters=40):
    """Estima os parâmetros e aproxima a covariância posterior."""
    X = d["X"]
    y = d["y"]
    P = d["P"]

    Q = base.precisao(d, sig_a, sig_d, tau, s_h)

    th = np.zeros(P)
    th[0] = np.log(1.3)

    for _ in range(iters):
        lam = np.exp(X @ th)

        gradiente = X.T @ (y - lam) - Q @ th
        hessiana = (X.T @ sp.diags(lam) @ X + Q).tocsc()

        lu = splu(hessiana)
        passo = lu.solve(gradiente)

        th += passo

        if np.abs(passo).max() < 1e-8:
            break

    # Hessiana no ponto final
    lam = np.exp(X @ th)
    hessiana = (X.T @ sp.diags(lam) @ X + Q).tocsc()
    lu = splu(hessiana)

    loglik = np.sum(
        y * np.log(lam) - lam - gammaln(y + 1)
    )

    return {
        "th": th,
        "lu": lu,
        "H": hessiana,
        "Q": Q,
        "d": d,
        "sig_a": sig_a,
        "sig_d": sig_d,
        "tau": tau,
        "s_h": s_h,
        "loglik": float(loglik),
    }


def indices_finais(base, fit):
    """Extrai os parâmetros atuais e sua covariância posterior."""
    d = fit["d"]
    n = base.n

    # Parâmetros globais e efeitos de mando
    indices = [0, 1] + [2 + i for i in range(n)]

    # Último estado de ataque e defesa de cada equipe
    indices += [
        int(d["offA"][i] + d["cont"][i] - 1)
        for i in range(n)
    ]

    indices += [
        int(d["offD"][i] + d["cont"][i] - 1)
        for i in range(n)
    ]

    indices = np.array(indices)

    # Seleciona os estados finais da matriz completa
    E = np.zeros((d["P"], len(indices)))
    E[indices, np.arange(len(indices))] = 1

    cov = fit["lu"].solve(E)[indices, :]
    cov = (cov + cov.T) / 2

    return indices, fit["th"][indices], cov


def pmf_tabela_old(lh, la, rho):
    """Distribuição conjunta de placares com correção de Dixon-Coles."""
    gols = np.arange(NG)

    def log_pois(lam):
        return (
            -lam[..., None]
            + gols * np.log(lam[..., None])
            - gammaln(gols + 1)
        )

    p_casa = np.exp(log_pois(lh))
    p_fora = np.exp(log_pois(la)

    P = p_casa[..., :, None] * p_fora[..., None, :]

    if rho != 0:
        P[..., 0, 0] *= 1 - lh * la * rho
        P[..., 0, 1] *= 1 + lh * rho
        P[..., 1, 0] *= 1 + la * rho
        P[..., 1, 1] *= 1 - rho

    return P
