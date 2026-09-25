# Como funciona

[🇺🇸 English](../methods.md) · [README](../../README(pt-br).md) · [Resultados](results.md) · [Reproduzindo os experimentos](reproducing.md)

- [Polling](#polling-método-base-replicado), o método base replicado
- [Efficient Polling](#efficient-polling), que faz poll sob demanda
- [Efficient Relative Polling](#efficient-relative-polling), que dispensa a grade fixa
- [Efficient Relative Narrowing Polling](#efficient-relative-narrowing-polling), cujo pulo estreita entre duas taxas
- [Modelo de custo](#modelo-de-custo)

## Polling (método base replicado)

O Método de Polling é de Tan et al., [Expediting Convergence via Polling Optimisation for Gradient Descent in Neural Networks](https://doi.org/10.3390/jeta4010001) (2025). A cada batch, após calcular o gradiente `g` a partir dos pesos `θ`, cada candidato de LR é aplicado como passo de teste e o vencedor é mantido:

```
θ̂ₖ = θ − lrₖ · g               para cada lrₖ ∈ C
k*  = argmax acc(θ̂ₖ, batch)     (empates → menor lr)
θ   ← θ̂ₖ*
```

O conjunto de candidatos é `C = {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}`, abrangendo quatro ordens de magnitude em torno da LR base. Os passos de teste são feitos copiando o estado do modelo + otimizador uma vez e recarregando-o antes de cada candidato, de modo que todos partam de condições pré-passo idênticas. Como os testes sobrescrevem os pesos, o passo vencedor é reaplicado uma vez a partir do snapshot, então um poll custa `N + 1 = 6` passos de otimizador em vez de `N = 5`.

## Efficient Polling

O mecanismo de seleção é mantido intacto, mas só se faz poll num batch quando necessário:

1. **Cronograma adaptativo de poll.** Seja `K` o intervalo de poll. Após um poll, se a seleção não muda, `K ← min(2K, K_max)` (backoff geométrico, limitado a `K_max = 64`); se mudou, `K ← 1` (poll a cada dois batches até estabilizar de novo). Entre polls, um único passo SGD cego usa a última LR selecionada. O backoff só entra em ação depois que um poll mostrou *sinal*, isto é, depois que as acurácias dos candidatos diferiram entre si. Na inicialização todo candidato empata, e tratar um empate como seleção estável travaria o treino na menor LR candidata para sempre.

   A regra tem um limite. O sinal é registrado uma vez e para sempre, e uma única resposta certa basta para produzi-lo. Com muitas classes, um batch com acurácia de acaso ainda separa os candidatos por uma resposta de vez em quando; a partir daí, cada empate devolve de novo o menor candidato, e a repetição dobra o intervalo. No CIFAR-100 isso travou as cinco seeds em `1e-5`, duas delas pela run inteira (ver [Resultados](results.md#o-que-os-quatro-datasets-acrescentam)).

2. **Guarda anti-divergência em dois níveis.** Passos cegos não têm validação por passo, então um passo com LR alta pode divergir. A guarda reaproveita quantidades já computadas:
   - **Nível 2, polls disparados por spike (prevenção):** se a perda do batch ultrapassa `γ · EMA(perda)` (`γ = 3`, `β = 0,9`), faz poll imediatamente para que o critério de acurácia possa rejeitar um passo explosivo.
   - **Nível 1, checkpoints de rollback (recuperação):** o snapshot de cada poll também serve como checkpoint conhecidamente bom; se a perda for não-finita ou ultrapassar `ℓ_rb`, o dobro da perda de um chute uniforme, restaura o checkpoint e retoma o polling.

   Nas cinco execuções do CIFAR-10, os polls disparados por spike somaram em média 378 por execução (6,6% de todos os polls), e o nível de recuperação disparou apenas duas vezes nessas execuções, ambas na mesma seed. Sua necessidade é real, porém: uma execução inicial sem guarda divergiu para `NaN` na época 33, a partir de um único passo cego com `lr = 1e-1`, e nunca se recuperou, a mesma falha que matou a maioria dos schedulers manuais na comparação.

3. **Gatilhos alternativos (ablação).** `trigger="fixed"` faz poll a cada `K + 1` batches, com intervalo constante em vez de backoff, e `trigger="random"` faz poll em cada batch independentemente com probabilidade `p`. Ambos mantêm a regra 2 (a guarda) e o mecanismo de seleção intactos, e ambos são calibrados para a taxa de poll que o backoff adaptativo mede, de modo que a comparação isola o gatilho, não o orçamento de polls. Ver [Ablação do gatilho](results.md#ablação-do-gatilho).

| Símbolo | Valor | Papel |
|---|---|---|
| `C` | `{1e-5, …, 1e-1}` | taxas de aprendizado candidatas |
| `lr_init` | `1e-3` | LR antes do primeiro poll |
| `K_max` | `64` | intervalo máximo de poll (teto do backoff) |
| `γ` | `3` | limiar de spike (nível 2) |
| `β` | `0,9` | decaimento da EMA da perda |
| `ℓ_rb` | `2·ln(classes)`, `4,61` no CIFAR-10 | limiar de rollback (nível 1) |

## Efficient Relative Polling

Os dois métodos acima escolhem dentro de uma grade *fixa*, e nas runs registradas do CIFAR-10 a taxa escolhida passa a maior parte do treino encostada nas bordas dessa grade: `1e-1` na primeira fase, `1e-5` depois do annealing. O Efficient Relative Polling dispensa a grade. O usuário escolhe uma taxa de aprendizado e um multiplicador `m`, e cada poll testa três candidatos em torno da taxa em uso; o vencedor vira o novo centro:

```
C_t = {X/m, X, X·m}        X ← argmax acc(θ̂, batch)
```

Quando um poll volta cego (todos os candidatos com a mesma pontuação, o que a acurácia do batch faz sempre que um passo não muda nenhuma predição), o poll seguinte olha um multiplicador mais longe nos dois sentidos, e continua alargando até enxergar uma diferença; um poll com sinal estreita a janela de volta. Três regras completam o método:

1. **Backoff sem teto, limitado pela falha.** O intervalo de poll `k` dobra quando um poll agendado com sinal mantém a taxa, um poll cego o deixa como está, e não tem `K_max`. Cada poll registra o *melhor* ponto visto até então (pesos, estado do otimizador e taxa, julgados por uma tendência lenta da perda). Quando um trecho cego estoura, o treino volta no tempo até esse ponto, `k` vai a zero e o teto do próximo slow start passa a ser metade do intervalo que estourou; o crescimento é exponencial até o teto e linear acima dele, como no controle de congestionamento do TCP.
2. **Empate mantém a taxa, exceto depois de um restart.** Um empate não carrega informação, então um poll comum mantém `X`. O poll logo depois de um restart desempata um degrau *para baixo*, porque um restart tem uma única causa: taxa alta demais para passos cegos.
3. **Tendência no estilo Adam.** A tendência da perda é uma média exponencial com correção de viés mais um segundo momento dos desvios, de modo que um spike é uma perda mais de `z` desvios acima da tendência, em vez de uma razão fixa sobre ela.

```python
from efficient_polling_lr_scheduler import EfficientRelativePollingSGD

optimizer = EfficientRelativePollingSGD(model, lr=1e-3, multiplier=10.0, lr_max=1e-1)
```

| Símbolo | Padrão | Papel |
|---|---|---|
| `lr` | `1e-3` | a única taxa que o usuário escolhe |
| `m` | `10` | espaçamento dos candidatos |
| `lr_min`, `lr_max` | nenhum | limites opcionais; os experimentos usam o teto `1e-1`, o mesmo de todos os outros métodos |
| `z` | `3` | limiar de spike, em desvios acima da tendência |
| `ℓ_rb` | `2·ln(classes)` | limiar de estouro, como acima |

O `EfficientRelativeEpochPolling` aplica a mesma janela por época em vez de por batch, conduzido por `fit(..., epoch_polling=...)`: treina uma época inteira por candidato a partir de um snapshot e mantém a que teve a menor perda média de treino. Fica como resultado negativo documentado; ver [Resultados](results.md#efficient-relative-polling).

## Efficient Relative Narrowing Polling

O Efficient Relative Polling move a taxa um multiplicador inteiro por vez. Com `m = 10` ela vive nas décadas, e quando a taxa que o batch quer fica entre duas delas, ela salta de uma para a outra e nunca testa o que fica no meio. O Efficient Relative Narrowing Polling deixa o próprio pulo `f` se adaptar, começando em `m`:

```
C_t = {X/f, X, X·f}        f = m^((1 − ν)^d)
```

`d` conta os estreitamentos em vigor e `ν` é a fração do pulo, em ordens de grandeza, que um estreitamento tira. `d` muda quando um comportamento dos polls se mantém, nunca por um poll isolado:

1. **A taxa está cercada:** o centro ganha, ou o vencedor inverte (sobe depois de descer, ou desce depois de subir). Os dois são um comportamento só: com `{10¹, 10², 10³}`, `10²` ganhar sempre diz o mesmo que uma oscilação entre `10¹` e `10³`. Um cerco que se mantém é um platô, e o pulo estreita (`d + 1`) para olhar entre os candidatos.
2. **A taxa segue num sentido:** o vencedor se move no mesmo sentido da última mudança, só subindo ou só descendo. Uma sequência que se mantém quer dizer que a taxa ainda está longe, e o pulo volta a alargar (`d − 1`) dos dois lados, para buscar num range maior, nunca além de `m`.

Cada comportamento tem uma paciência contada em polls, `p_n` para estreitar e `p_w` para alargar, as duas começando em `p₀`. Um poll de um comportamento tira um poll inteiro da própria paciência e `ρ` de um poll da outra: um poll que quebra a tendência desacelera a contagem sem devolver o que já foi gasto, então um poll perdido não desfaz um platô que está se formando. Quando uma paciência chega a zero, `d` anda um passo (a não ser que já esteja em `0`, ou em `d_max` quando há um) e as duas paciências recomeçam de `p₀`. A paciência é um contador diferente do intervalo de poll `k` do backoff, que não muda: `k` decide quando fazer poll, `p_n` e `p_w` com que resolução.

Um empate (todos os candidatos com a mesma pontuação) com o pulo estreitado diz que o pulo ficou fino demais para distinguir os candidatos, e conta como poll de alargamento; em `m` o empate alarga a janela na hora, como no Efficient Relative Polling. Por padrão nada limita os estreitamentos: um platô que se mantém continua estreitando o pulo, e quem para isso são os empates. O único limite que sempre vale é o numérico, já que um estreitamento que poria os vizinhos a um arredondamento do centro não acontece. Um centro em `lr_min` ou `lr_max` tem um vizinho dobrado sobre ele e não cerca nada, então a vitória dele não conta para nenhum comportamento, nem o poll que encerra um trecho cego, que só traz a janela de volta a `m`. Um restart volta para `f = m` com as paciências cheias. Produtos de pulos fracionários desviam um ou dois ulps, então uma taxa a um arredondamento de um limite conta como estando nele. Com `ν = 0` o método é o Efficient Relative Polling, salvo esse arredondamento nos limites.

O poll também pode ranquear os testes pela loss do batch em vez da acurácia do batch (`criterion="loss"`, em todo otimizador com poll por batch; `--criterion loss` no benchmark, para o Efficient Relative Polling por batch e esta variante). A loss é contínua, então ainda distingue passos de teste próximos demais para mudar uma única previsão, e os candidatos quase nunca empatam: pela loss, os empates deixam de puxar o pulo de volta, e só as sequências num sentido só o alargam. A acurácia continua sendo o número reportado e o que escolhe o checkpoint.

```python
from efficient_polling_lr_scheduler import EfficientRelativeNarrowingPollingSGD

optimizer = EfficientRelativeNarrowingPollingSGD(
    model, lr=1e-3, multiplier=10.0, narrowing=0.5, patience=8, lr_max=1e-1
)
```

| Símbolo | Padrão | Papel |
|---|---|---|
| `m` | `10` | o pulo mais largo, como no Efficient Relative Polling |
| `ν` | `0.5` | `narrowing`: fração do pulo que um estreitamento tira; `0.5` põe o próximo vizinho no meio geométrico, `X·√10` para `m = 10` |
| `d_max` | nenhum | `max_narrowings`: limite opcional dos estreitamentos, que faz o pulo mais fino ser `m^((1 − ν)^d_max)`; `3` com os outros padrões para em `×1,33` |
| `p₀` | `8` | `patience`: polls que um comportamento precisa durar para o pulo andar um passo |
| `ρ` | `0.5` | `break_discount`: quanto um poll que quebra o comportamento tira da paciência dele, como fração de um poll, em `[0, 1)` |
| | `"score"` | `criterion`: o que ranqueia os testes, a pontuação da closure (acurácia do batch) ou, com `"loss"`, a loss do batch |

É experimental: nada foi rodado com ele além de um teste de fumaça.

## Modelo de custo

Um poll custa `N + 1` passos de otimizador: um teste por candidato, mais a reaplicação do vencedor. São 6 passos com a grade fixa de cinco candidatos e 4 com a janela relativa de três. Com `P` polls entre `B` batches:

```
S_eff = P·(N+1) + (B − P) = B + N·P
```

No CIFAR-10, `B = 105.600` e o Efficient Polling faz em média `P = 5.732,2` polls, o que dá `134.261` passos: uma redução de 79% em relação aos `633.600` do Polling base, reproduzindo exatamente a tabela medida. Apenas 26% desses passos vêm de polls; os 74% restantes são passos SGD comuns, o que explica o tempo por época ficar no nível do SGD puro. Os passos de todos os métodos em todos os datasets estão nos [Resultados](results.md).
