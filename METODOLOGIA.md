Metodologia do simulador estocástico de rebaixamento (Brasileirão 2026, 36ª rodada concluída)

1 Geral
Simula 1M de "finais de campeonato" (27 jogos restantes: rodadas 36, 37 e 38). Em cada uma:
i- sorteia os parâmetros de força dos times a partir da posteriori; ii- simula os jogos rodada a rodada, com intensidades que
dependem dos incentivos de cada time naquele momento; iii- aplica o critério de desempate; iv- registra a posição final.
A probabilidade de queda é a fração de simulações em que o time termina entre 17º e 20º.

2 Força dinâmica (espaço de estados)
Gols do mandante ~ f(λ_c), λ_c = exp(μ + h + A[c,k] − D[f,l]); visitante λ_f = exp(μ + A[f,l] − D[c,k])
Ataque A e defesa D de cada time seguem passeio aleatório por jogo: A[i,0] ~ N(0,τ²), A[i,k]−A[i,k−1] ~ N(0,σ²) (idem D).
τ = encolhimento à média da liga; σ = volatilidade da forma ("momento"). μ, h: priors vagos.
Inferência: moda a posteriori por Newton esparso + aproximação de Laplace (covariância = inversa da Hessiana).
Cada campeonato simulado sorteia seus parâmetros dessa normal => incerteza de parâmetros propagada.
Opcional: saltos raros de forma (prob. 5%/jogo, sd 0,08) — lesões, troca de técnico.

3 Distribuição de gols
Marginais Conway-Maxwell-Poisson com média fixa e ν≈1,4 (sub-dispersão: o Brasileirão tem menos placares extremos que Poisson),
mais dependência Dixon-Coles (ρ≈−0,15) nos placares baixos. Amostragem por tabelas de CDF + rejeição.

4 Seleção dos hiperparâmetros
Backtest walk-forward em 220 jogos (log-score do placar, log-loss e RPS). Os pesos dos 144 modelos (σ, τ, ρ, ν) vêm de
pseudo-BMA temperado: w ∝ exp(0,5·Σ log-score). A simulação final é uma mistura desses modelos.
Achados: forma é só fracamente persistente (σ ótimo 0,01–0,03); mando específico por time não é suportado pelos dados;
gols sub-dispersos; o modelo dinâmico supera o estático em +0,052±0,021 nat/jogo.

5 Crescimento endógeno ("jogo morto")
Antes de cada rodada simulada, calculam os melhores/piores colocados possíveis de cada time (critério pontos, vitórias).
Um time está "vivo" se algum limiar relevante (título, G4, vaga na Libertadores, Sula, rebaixamento) ainda está ao seu alcance.
Time sem nada em jogo perde intensidade (ataque −δa, defesa −δd, δ~N⁺(0,06; 0,04)). Palmeiras (finais de Libertadores/Copa) roda elenco
nas duas últimas rodadas (~N⁺(0,10; 0,05)). *δ e rodízio são premissas a priori, não estimados — por isso a análise de sensibilidade.*

6 Vagas na Libertadores como variável aleatória
N_T = 6 + 1[Palmeiras campeão da Libertadores, p~Beta] + 1[Grêmio rebaixado e regra Conmebol aplicável, p=0,9].
Assim, rebaixar o Grêmio abre vaga (efeito que beneficia Santos, Galo, Bahia etc.) e muda o que está em jogo para esses times.

7 Confronto direto
Os jogos entre os rivais diretos são simulados explicitamente (acoplamento). Como desempate: pontos, vitórias, saldo, gols pró, confronto direto
(empates de 2 times), sorteio. Cartões não são modelados. Empates exatos no 16º/17º pelos 3 primeiros critérios: 0,13% das simulações; decididos por confronto direto: 0,08%.

8 Limites
Não modela lesões/suspensões específicas, técnicos, arbitragem, clima, nem possíveis punições do STJD. Probabilidades são estimativas condicionais
às premissas, não previsões garantidas. Intervalo de Monte Carlo ≈ ±0,1 p.p.; a incerteza estrutural (16 hipóteses) é maior
