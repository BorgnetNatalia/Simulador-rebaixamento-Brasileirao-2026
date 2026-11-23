"""
Simulador Monte Carlo v2026 rebaixamento no Brasileirão Série A 2026.

Camadas (cada uma pode ser ligada/desligada para medir sua contribuição):
  1. Força dinâmica bayesiana (espaço de estados) + incerteza de parâmetros (Laplace) sorteada por campeonato.
  2. Passeio aleatório da forma jogo a jogo (+ saltos raros opcionais) durante as 3 rodadas restantes.
  3. Gols ~ Conway-Maxwell-Poisson (subdispersão) com dependência de Dixon-Coles (placares baixos).
  4. Incentivos ENDÓGENOS: em cada rodada simulada, um clube "sem nada a disputar" (a faixa de posições
     matematicamente possível não cruza nenhuma fronteira: título, G4, Libertadores, Sul-Americana, Z4)
     joga com menos intensidade. Rodízio do Palmeiras (finais da Libertadores e da Copa do Brasil).
  5. Nº de vagas na Libertadores via tabela = variável aleatória (final da Libertadores 28/11 e
     rebaixamento do Grêmio, finalista da Copa do Brasil, que perde a vaga se cair).
  6. Critérios de desempate do regulamento: pontos, vitórias, saldo, gols pró, confronto direto
     (empate entre 2 clubes), sorteio (cartões não modelados).
"""
import numpy as np
from dataclasses import dataclass, field
from carregar import carregar
from dist import pmf_marg, NG
from dinamico import Base, ajustar, indices_finais
from dados_tabela import TABELA

J_PLAY = carregar()
TIMES = list(TABELA); NT = len(TIMES); IDX = {t: i for i, t in enumerate(TIMES)}
ALVO = ["Internacional", "Corinthians", "Grêmio", "Vitória", "Mirassol"]

# (mandante, visitante, estágio)  estágio 1 = restante da 36ª, 2 = 37ª, 3 = 38ª
RESTANTES = [
    ("Fluminense", "Mirassol", 1), ("Internacional", "Vitória", 1), ("Santos", "Grêmio", 1),
    ("Botafogo", "São Paulo", 1), ("Bragantino", "Vasco", 1), ("Atlético-MG", "Corinthians", 1),
    ("Athletico-PR", "Remo", 1),
    ("Chapecoense", "Palmeiras", 2), ("São Paulo", "Remo", 2), ("Fluminense", "Cruzeiro", 2),
    ("Botafogo", "Bahia", 2), ("Coritiba", "Flamengo", 2), ("Vitória", "Santos", 2),
    ("Corinthians", "Grêmio", 2), ("Mirassol", "Athletico-PR", 2), ("Internacional", "Bragantino", 2),
    ("Atlético-MG", "Vasco", 2),
    ("Flamengo", "Chapecoense", 3), ("Vasco", "Vitória", 3), ("Santos", "Botafogo", 3),
    ("Palmeiras", "Coritiba", 3), ("Bragantino", "Fluminense", 3), ("Cruzeiro", "Internacional", 3),
    ("Remo", "Corinthians", 3), ("Grêmio", "Mirassol", 3), ("Athletico-PR", "São Paulo", 3),
    ("Bahia", "Atlético-MG", 3),
]
ROT_JOGOS = {("Chapecoense", "Palmeiras"), ("Palmeiras", "Coritiba")}   # rodízio: finais da Libertadores/CDB
M = len(RESTANTES)
H_ = np.array([IDX[a] for a, _, _ in RESTANTES]); A_ = np.array([IDX[b] for _, b, _ in RESTANTES])
ST_ = np.array([s for _, _, s in RESTANTES])
G_IDX = {t: i for i, t in enumerate(TIMES)}


def estado_atual():
    pts = np.zeros(NT, int); w = np.zeros(NT, int); gd = np.zeros(NT, int); gf = np.zeros(NT, int)
    H0 = np.zeros((NT, NT), np.int8)
    for g in J_PLAY:
        c, f = IDX[g["casa"]], IDX[g["fora"]]; x, y = g["gc"], g["gv"]
        pc = 3 if x > y else 1 if x == y else 0; pf = 3 if y > x else 1 if x == y else 0
        pts[c] += pc; pts[f] += pf; w[c] += x > y; w[f] += y > x
        gd[c] += x - y; gd[f] += y - x; gf[c] += x; gf[f] += y
        H0[c, f] += pc; H0[f, c] += pf
    rem = np.zeros((3, NT), int)           # jogos que cada time ainda tem a partir de cada estágio
    for s in (1, 2, 3):
        for h, a, st in zip(H_, A_, ST_):
            if st >= s: rem[s - 1, h] += 1; rem[s - 1, a] += 1
    return pts, w, gd, gf, H0, rem


PTS0, W0, GD0, GF0, H2H0, REM = estado_atual()


@dataclass
class Cfg:
    sigma: float = 0.02; tau: float = 0.18; rho: float = -0.15; nu: float = 1.4
    param_incerto: bool = True      # sorteia parâmetros da posteriori (senão usa a média)
    jump_prob: float = 0.0; jump_sd: float = 0.08
    incentivos: bool = True; delta_a: float = 0.06; delta_d: float = 0.06; delta_sd: float = 0.04
    rodizio: float = 0.10; rodizio_sd: float = 0.05
    p_lib: float = 0.5; p_regra: float = 0.9      # P(Palmeiras campeão da Libertadores); P(regra Conmebol vale p/ Grêmio)
    confronto_direto: bool = True


_FITS = {}
def posterior(sigma, tau):
    k = (sigma, tau)
    if k not in _FITS:
        B = Base(J_PLAY, TIMES); d = B.montar(len(J_PLAY)); f = ajustar(B, d, sigma, sigma, tau, 0.0)
        S, th, Sg = indices_finais(B, f); n = NT
        sel = np.array([0, 1] + list(range(2 + n, 2 + 3 * n)))
        _FITS[k] = (th[sel], Sg[np.ix_(sel, sel)])
    return _FITS[k]


_CDF = {}
def cdf_tabela(nu):
    if nu not in _CDF:
        gm = np.arange(0.02, 6.0, 0.01); _CDF[nu] = (gm, np.cumsum(pmf_marg(gm, nu), -1).clip(max=1.0))
    return _CDF[nu]


def sorteia_gols(lh, la, rho, nu, rng):
    gm, cdf = cdf_tabela(nu)
    def marg(m):
        i = np.clip(((m - gm[0]) / 0.01 + 0.5).astype(int), 0, len(gm) - 1)
        return (rng.random(m.shape)[..., None] > cdf[i]).sum(-1)
    x = marg(lh); y = marg(la)
    if rho != 0:
        wmax = np.maximum(1 - rho * lh * la, 1 - rho)       # ρ<0: maiores pesos em (0,0) e (1,1)
        pend = np.ones(x.shape, bool)
        for _ in range(40):
            w = np.ones(x.shape)
            w = np.where((x == 0) & (y == 0), 1 - lh * la * rho, w); w = np.where((x == 0) & (y == 1), 1 + lh * rho, w)
            w = np.where((x == 1) & (y == 0), 1 + la * rho, w);      w = np.where((x == 1) & (y == 1), 1 - rho, w)
            ok = rng.random(x.shape) < w / wmax
            pend &= ~ok
            if not pend.any(): break
            x = np.where(pend, marg(lh), x); y = np.where(pend, marg(la), y)
    return x, y


def incentivos(pts, wins, rem, conj_t):
    """Máscara (N,20): True se o clube ainda tem algo a disputar (alguma fronteira cruzável)."""
    lo = pts * 100 + wins; hi = (pts + 3 * rem) * 100 + (wins + rem)
    best = 1 + (lo[:, None, :] > hi[:, :, None]).sum(-1)
    worst = (hi[:, None, :] >= lo[:, :, None]).sum(-1)      
    vivo = np.zeros(pts.shape, bool)
    for t in conj_t: vivo |= (best <= t) & (worst > t)
    return vivo


def classificar(pts, w, gd, gf, H2H, rng, cd=True):
    """Posição final (N,20) com o desempate: pts, vitórias, saldo, gols pró, confronto direto, sorteio."""
    N = pts.shape[0]
    key = pts * 1e9 + w * 1e7 + (gd + 300) * 1e4 + gf
    ordem = np.argsort(-(key + rng.random(key.shape) * 0.5), axis=1)
    ks = np.take_along_axis(key, ordem, 1)
    empat = (ks[:, 1:] == ks[:, :-1])
    n_tie_fronteira = int(empat[:, 15].sum())              # empate entre 16º e 17º nos 3 primeiros critérios
    n_cd = 0
    if cd:
        for r in np.where(empat.any(1))[0]:
            o = ordem[r].copy(); k = ks[r]; i = 0
            while i < NT:
                j = i
                while j + 1 < NT and k[j + 1] == k[i]: j += 1
                if j == i + 1:                                
                    a, b = o[i], o[j]
                    if H2H[r, a, b] != H2H[r, b, a]:
                        if H2H[r, b, a] > H2H[r, a, b]: o[i], o[j] = b, a
                        if i == 15: n_cd += 1
                i = j + 1
            ordem[r] = o
    rank = np.empty_like(ordem); rank[np.arange(N)[:, None], ordem] = np.arange(NT)[None]
    return rank, n_tie_fronteira, n_cd


def rodar_chunk(cfg: Cfg, N, seed, guardar=True):
    rng = np.random.default_rng(seed)
    th, Sg = posterior(cfg.sigma, cfg.tau)
    if cfg.param_incerto:
        L = np.linalg.cholesky(Sg + 1e-9 * np.eye(len(th))); par = th + rng.standard_normal((N, len(th))) @ L.T
    else:
        par = np.tile(th, (N, 1))
    mu, ha, A, D = par[:, 0], par[:, 1], par[:, 2:2 + NT].copy(), par[:, 2 + NT:].copy()
    pts = np.tile(PTS0, (N, 1)); w = np.tile(W0, (N, 1)); gd = np.tile(GD0, (N, 1)); gf = np.tile(GF0, (N, 1))
    H2H = np.tile(H2H0, (N, 1, 1))
    if cfg.incentivos:
        da = np.clip(rng.normal(cfg.delta_a, cfg.delta_sd, N), 0, None); dd = np.clip(rng.normal(cfg.delta_d, cfg.delta_sd, N), 0, None)
        rot = np.clip(rng.normal(cfg.rodizio, cfg.rodizio_sd, N), 0, None)
    else:
        da = dd = rot = np.zeros(N)
    p_i = rng.beta(20 * cfg.p_lib, 20 * (1 - cfg.p_lib), N)             
    x_lib = rng.random(N) < p_i
    resultados = np.zeros((N, M), np.int8); vivos_est = []
    for s in (1, 2, 3):
        # INCENTIVO
        if cfg.incentivos:
            rem = REM[s - 1]
            base_t = [1, 4, 16]
            if s < 3: conj = base_t + [6, 7, 8, 12, 13, 14]; vivo = incentivos(pts, w, rem, conj)  
            else:
                v_a = incentivos(pts, w, rem, base_t + [6, 7, 12, 13]); v_b = incentivos(pts, w, rem, base_t + [7, 8, 13, 14])
                vivo = np.where(x_lib[:, None], v_b, v_a)
            vivos_est.append(vivo)
        else:
            vivo = np.ones((N, NT), bool)
        idx = np.where(ST_ == s)[0]
        # EVOLU;'AO
        jog = np.zeros(NT, bool); jog[H_[idx]] = True; jog[A_[idx]] = True
        for i in np.where(jog)[0]:
            eA = rng.normal(0, cfg.sigma, N); eD = rng.normal(0, cfg.sigma, N)
            if cfg.jump_prob > 0:
                jA = rng.random(N) < cfg.jump_prob; jD = rng.random(N) < cfg.jump_prob
                eA += jA * rng.normal(0, cfg.jump_sd, N); eD += jD * rng.normal(0, cfg.jump_sd, N)
            A[:, i] += eA; D[:, i] += eD
        # GOL ESPERADO
        h, a = H_[idx], A_[idx]
        morto = ~vivo
        rh = np.array([RESTANTES[k][:2] in ROT_JOGOS and RESTANTES[k][0] == "Palmeiras" for k in idx])
        ra = np.array([RESTANTES[k][:2] in ROT_JOGOS and RESTANTES[k][1] == "Palmeiras" for k in idx])
        # rodízio do Palmeiras só nas rodadas 37 e 38 (jogo: Chapecoense x Palmeiras e Palmeiras x Coritiba)
        lh = mu[:, None] + ha[:, None] + A[:, h] - D[:, a] - da[:, None] * morto[:, h] + dd[:, None] * morto[:, a]
        la_ = mu[:, None] + A[:, a] - D[:, h] - da[:, None] * morto[:, a] + dd[:, None] * morto[:, h]
        lh = lh - rot[:, None] * rh + rot[:, None] * ra          
        la_ = la_ - rot[:, None] * ra + rot[:, None] * rh
        x, y = sorteia_gols(np.exp(lh), np.exp(la_), cfg.rho, cfg.nu, rng)
        # ATUALIZA A TAB
        for j, k in enumerate(idx):
            c, f = H_[k], A_[k]; xv, yv = x[:, j], y[:, j]
            pc = np.where(xv > yv, 3, np.where(xv == yv, 1, 0)); pf = np.where(yv > xv, 3, np.where(xv == yv, 1, 0))
            pts[:, c] += pc; pts[:, f] += pf; w[:, c] += xv > yv; w[:, f] += yv > xv
            gd[:, c] += xv - yv; gd[:, f] += yv - xv; gf[:, c] += xv; gf[:, f] += yv
            H2H[:, c, f] += pc.astype(np.int8); H2H[:, f, c] += pf.astype(np.int8)
            resultados[:, k] = np.where(xv > yv, 2, np.where(xv == yv, 1, 0))
    rank, n_tf, n_cd = classificar(pts, w, gd, gf, H2H, rng, cfg.confronto_direto)
    gre_cai = rank[:, IDX["Grêmio"]] >= 16
    regra = rng.random(N) < cfg.p_regra
    n_t = 6 + x_lib.astype(int) + (gre_cai & regra).astype(int)
    out = dict(rank=rank.astype(np.int8), pts=pts.astype(np.int16), res=resultados, x_lib=x_lib, n_t=n_t.astype(np.int8),
               n_tie_fronteira=n_tf, n_cd=n_cd, N=N)
    if vivos_est: out["vivos"] = np.stack([v.mean(0) for v in vivos_est])
    return out


def rodar(cfgs_pesos, N, seed=2026, chunk=25000):
    """cfgs_pesos: lista [(Cfg, peso)]; alocação proporcional determinística de simulações por configuração."""
    pesos = np.array([p for _, p in cfgs_pesos], float); pesos /= pesos.sum()
    n_c = np.floor(pesos * N).astype(int); n_c[np.argmax(pesos)] += N - n_c.sum()
    saidas = []; ss = np.random.SeedSequence(seed); k = 0
    for (cfg, _), nc in zip(cfgs_pesos, n_c):
        rest = nc
        while rest > 0:
            c = min(chunk, rest); sd = int(ss.spawn(1)[0].generate_state(1)[0]); saidas.append(rodar_chunk(cfg, c, sd)); rest -= c; k += 1
    agg = {}
    for key in ("rank", "pts", "res", "x_lib", "n_t"): agg[key] = np.concatenate([s[key] for s in saidas])
    agg["n_tie_fronteira"] = sum(s["n_tie_fronteira"] for s in saidas); agg["n_cd"] = sum(s["n_cd"] for s in saidas)
    agg["N"] = N
    if "vivos" in saidas[0]:
        agg["vivos"] = sum(s["vivos"] * s["N"] for s in saidas) / N
    return agg
