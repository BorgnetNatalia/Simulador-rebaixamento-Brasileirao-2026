"""
Dados do Brasileirão Série A 2026 após a 35ª rodada + 3 jogos da 36ª 
Atualizar este arquivo a cada rodada: é o único lugar que precisa mudar.
"""

# Tabela atual: pontos, jogos, gols pró, gols contra
TABELA = {
    "Flamengo":      dict(pts=80, j=36, gp=74, gc=31),
    "Palmeiras":     dict(pts=74, j=36, gp=64, gc=26),
    "Cruzeiro":      dict(pts=66, j=36, gp=59, gc=50),
    "Fluminense":    dict(pts=60, j=35, gp=56, gc=42),
    "Athletico-PR":  dict(pts=54, j=35, gp=52, gc=45),
    "Bahia":         dict(pts=54, j=36, gp=50, gc=46),
    "Atlético-MG":   dict(pts=51, j=35, gp=44, gc=40),
    "Santos":        dict(pts=50, j=35, gp=51, gc=49),
    "Bragantino":    dict(pts=47, j=35, gp=45, gc=41),
    "Vasco":         dict(pts=47, j=35, gp=47, gc=51),
    "Coritiba":      dict(pts=45, j=36, gp=41, gc=52),
    "São Paulo":     dict(pts=44, j=35, gp=39, gc=40),
    "Botafogo":      dict(pts=42, j=35, gp=46, gc=57),
    "Vitória":       dict(pts=41, j=35, gp=37, gc=52),
    "Mirassol":      dict(pts=41, j=35, gp=39, gc=48),
    "Corinthians":   dict(pts=40, j=35, gp=40, gc=42),
    "Internacional": dict(pts=38, j=35, gp=39, gc=44),
    "Grêmio":        dict(pts=38, j=35, gp=38, gc=47),
    "Remo":          dict(pts=32, j=35, gp=39, gc=57),
    "Chapecoense":   dict(pts=21, j=36, gp=34, gc=74),
}

# Campanha em casa e fora: (jogos, gols pró, gols contra)
CASA = {
    "Flamengo": (18, 37, 14), "Palmeiras": (18, 37, 12), "Fluminense": (17, 31, 17),
    "Cruzeiro": (18, 32, 19), "Vitória": (18, 25, 17), "Atlético-MG": (17, 27, 18),
    "São Paulo": (18, 25, 14), "Athletico-PR": (17, 28, 18), "Bragantino": (17, 29, 18),
    "Vasco": (18, 27, 25), "Grêmio": (18, 27, 21), "Bahia": (18, 26, 20),
    "Mirassol": (18, 24, 22), "Botafogo": (17, 27, 22), "Corinthians": (18, 22, 17),
    "Santos": (17, 23, 21), "Internacional": (17, 22, 17), "Coritiba": (18, 22, 22),
    "Remo": (18, 24, 22), "Chapecoense": (18, 23, 40),
}
FORA = {
    "Flamengo": (18, 37, 17), "Palmeiras": (18, 27, 14), "Cruzeiro": (18, 27, 31),
    "Santos": (18, 28, 28), "Bahia": (18, 24, 26), "Athletico-PR": (18, 24, 27),
    "Coritiba": (18, 19, 30), "Fluminense": (18, 25, 25), "Vasco": (17, 20, 26),
    "Bragantino": (18, 16, 23), "Atlético-MG": (18, 17, 22), "Botafogo": (18, 19, 35),
    "Corinthians": (17, 18, 25), "Internacional": (18, 17, 27), "Mirassol": (17, 15, 26),
    "São Paulo": (17, 14, 26), "Remo": (17, 15, 35), "Grêmio": (17, 11, 26),
    "Chapecoense": (18, 11, 34), "Vitória": (17, 12, 35),
}

# Últimos 10 jogos: (pontos, gols pró, gols contra) — usado como "momento"
ULT10 = {
    "Flamengo": (26, 23, 10), "Cruzeiro": (24, 21, 13), "Vasco": (22, 20, 11),
    "Palmeiras": (21, 19, 5), "Fluminense": (18, 17, 10), "Santos": (18, 14, 11),
    "Mirassol": (16, 12, 9), "Atlético-MG": (15, 12, 10), "Internacional": (13, 13, 13),
    "Bragantino": (12, 14, 13), "Vitória": (12, 13, 15), "Bahia": (11, 10, 14),
    "São Paulo": (11, 8, 12), "Botafogo": (11, 9, 17), "Grêmio": (10, 11, 14),
    "Athletico-PR": (9, 15, 20), "Remo": (9, 9, 15), "Corinthians": (8, 14, 17),
    "Coritiba": (8, 7, 17), "Chapecoense": (4, 6, 22),
}

# Jogos restantes: (mandante, visitante)  — 27 jogos
JOGOS = [
    # 36ª rodada (restante)
    ("Fluminense", "Mirassol"),          # 22/11
    ("Internacional", "Vitória"),        # 23/11
    ("Santos", "Grêmio"),                # 23/11
    ("Botafogo", "São Paulo"),           # 23/11
    ("Bragantino", "Vasco"),             # 36ª
    ("Atlético-MG", "Corinthians"),      # 25/11
    ("Athletico-PR", "Remo"),            # 36ª
    # 37ª rodada
    ("Chapecoense", "Palmeiras"),        # 24/11
    ("São Paulo", "Remo"),               # 26/11
    ("Fluminense", "Cruzeiro"),          # 27/11
    ("Botafogo", "Bahia"),               # 28/11
    ("Coritiba", "Flamengo"),            # 29/11
    ("Vitória", "Santos"),               # 29/11
    ("Corinthians", "Grêmio"),           # 29/11
    ("Mirassol", "Athletico-PR"),        # 29/11
    ("Internacional", "Bragantino"),     # 29/11
    ("Atlético-MG", "Vasco"),            # 29/11
    # 38ª rodada (02/12, todos às 21h30)
    ("Flamengo", "Chapecoense"),
    ("Vasco", "Vitória"),
    ("Santos", "Botafogo"),
    ("Palmeiras", "Coritiba"),
    ("Bragantino", "Fluminense"),
    ("Cruzeiro", "Internacional"),
    ("Remo", "Corinthians"),
    ("Grêmio", "Mirassol"),
    ("Athletico-PR", "São Paulo"),
    ("Bahia", "Atlético-MG"),
]

ALVO = ["Internacional", "Corinthians", "Grêmio", "Vitória", "Mirassol"]

# Vitórias (1º critério de desempate depois dos pontos)
VITORIAS = {
    "Flamengo": 24, "Palmeiras": 21, "Cruzeiro": 20, "Fluminense": 17, "Athletico-PR": 15,
    "Bahia": 14, "Atlético-MG": 14, "Santos": 13, "Bragantino": 13, "Vasco": 13,
    "Coritiba": 12, "São Paulo": 12, "Botafogo": 11, "Vitória": 11, "Mirassol": 10,
    "Corinthians": 10, "Internacional": 9, "Grêmio": 9, "Remo": 7, "Chapecoense": 4,
}
