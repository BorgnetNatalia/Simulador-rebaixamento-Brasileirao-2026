
"""
Distribuições de gols e ajustes na probabilidade dos placares.

Compara Poisson e Conway-Maxwell-Poisson (CMP), permitindo controlar
a dispersão dos gols, a dependência entre os resultados das equipes
e a frequência de empates.
"""

import numpy as np
from scipy.special import gammaln


# Placar considerado nas simulações: de 0 a 12 gols
NG = 13

_gols = np.arange(NG)
_suporte = np.arange(60)
_cache = {}


def _mean_com(loglam, nu):
    """Calcula a média da distribuição CMP para um dado parâmetro."""
    log_peso = (
        _suporte * loglam[..., None]
        - nu * gammaln(_suporte + 1)
    )

    # Evita problemas numéricos com exponenciais muito grandes
    log_peso -= log_peso.max(axis=-1, keepdims=True)

    pesos = np.exp(log_peso)

    return (
        (pesos * _suporte).sum(axis=-1)
        / pesos.sum(axis=-1)
    )


def _tabela_lam(nu):
    """Prepara a tabela de conversão entre média e parâmetro lambda."""
    if nu not in _cache:
        medias = np.linspace(0.03, 6.0, 600)

        inferior = np.full_like(medias, -9.0)
        superior = np.full_like(medias, 9.0)

        # Busca binária para encontrar o lambda correspondente a cada média
        for _ in range(60):
            meio = (inferior + superior) / 2
            media_calc = _mean_com(meio, nu)

            superior = np.where(
                media_calc > medias,
                meio,
                superior,
            )

            inferior = np.where(
                media_calc <= medias,
                meio,
                inferior,
            )

        _cache[nu] = (medias, (inferior + superior) / 2)

    return _cache[nu]


def pmf_marg(m, nu=1.0):
    """Retorna as probabilidades de gols para uma média esperada."""
    m = np.asarray(m, dtype=float)

    # Quando nu = 1, a distribuição se reduz à Poisson
    if nu == 1.0:
        return np.exp(
            -m[..., None]
            + _gols * np.log(m[..., None])
            - gammaln(_gols + 1)
        )

    # Para outros valores de nu, usa a distribuição CMP
    medias, lambdas = _tabela_lam(nu)
    loglam = np.interp(m, medias, lambdas)

    log_normalizacao = np.log(
        np.exp(
            _suporte * loglam[..., None]
            - nu * gammaln(_suporte + 1)
        ).sum(axis=-1)
    )

    return np.exp(
        _gols * loglam[..., None]
        - nu * gammaln(_gols + 1)
        - log_normalizacao[..., None]
    )


def peso_conjunto(lh, la, x, y, rho, kappa):
    """Calcula os ajustes de dependência e de frequência de empates."""
    peso = np.ones(np.broadcast(lh, x).shape)

    # Correção de Dixon-Coles para placares baixos
    peso = np.where(
        (x == 0) & (y == 0),
        1 - lh * la * rho,
        peso,
    )

    peso = np.where(
        (x == 0) & (y == 1),
        1 + lh * rho,
        peso,
    )

    peso = np.where(
        (x == 1) & (y == 0),
        1 + la * rho,
        peso,
    )

    peso = np.where(
        (x == 1) & (y == 1),
        1 - rho,
        peso,
    )

    # Ajuste adicional para os placares empatados
    peso *= np.where(x == y, 1 + kappa, 1.0)

    return peso


def pmf_conjunta(lh, la, rho=0.0, kappa=0.0, nu=1.0):
    """Monta a distribuição conjunta dos placares das duas equipes."""
    p_casa = pmf_marg(lh, nu)
    p_fora = pmf_marg(la, nu)

    # Começa pela hipótese de independência entre os gols
    P = p_casa[..., :, None] * p_fora[..., None, :]

    # Ajuste de Dixon-Coles
    if rho != 0:
        P[..., 0, 0] *= 1 - lh * la * rho
        P[..., 0, 1] *= 1 + lh * rho
        P[..., 1, 0] *= 1 + la * rho
        P[..., 1, 1] *= 1 - rho

    # Aumenta ou reduz a probabilidade dos empates
    if kappa != 0:
        gols = np.arange(NG)
        P[..., gols, gols] *= 1 + kappa

    # Normaliza para que a soma das probabilidades seja igual a 1
    return P / P.sum(axis=(-1, -2), keepdims=True)
