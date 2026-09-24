# Otimização Eficiente da Taxa de Aprendizado Baseada em Polling para Redes Neurais

> Escolha a taxa de aprendizado medindo-a no batch, a um custo próximo ao de um SGD comum.

[![PyPI](https://img.shields.io/pypi/v/efficient-polling-lr-scheduler.svg)](https://pypi.org/project/efficient-polling-lr-scheduler/)
[![Python](https://img.shields.io/pypi/pyversions/efficient-polling-lr-scheduler.svg)](https://pypi.org/project/efficient-polling-lr-scheduler/)
[![License](https://img.shields.io/pypi/l/efficient-polling-lr-scheduler.svg)](LICENSE)

```bash
pip install efficient-polling-lr-scheduler
```

[🇺🇸 English version](README.md) · [Como funciona](docs/pt-br/methods.md) · [Resultados](docs/pt-br/results.md) · [Reproduzindo os experimentos](docs/pt-br/reproducing.md)

O **Método de Polling** de [Tan et al.](https://doi.org/10.3390/jeta4010001) escolhe a taxa de aprendizado testando vários candidatos a cada batch e mantendo o que mais melhora a acurácia no batch. Funciona, e multiplica o custo do treino pelo número de candidatos. Este repositório o replica e acrescenta duas extensões que mantêm a seleção e cortam a maior parte do custo:

- **Efficient Polling** faz poll sob demanda. Um backoff exponencial alarga o intervalo entre polls enquanto a escolha é estável, e uma guarda anti-divergência em dois níveis protege os passos entre um poll e outro.
- **Efficient Relative Polling** dispensa a grade fixa de candidatos. O usuário escolhe uma taxa de aprendizado e um multiplicador, e cada poll testa a taxa em uso e suas duas vizinhas.

As duas são distribuídas como um pacote PyTorch. Elas são comparadas com oito métodos de comparação e duas ablações em cinco datasets, cinco seeds cada.

## Resultados em resumo

Acurácia de teste, média de cinco seeds de 150 épocas, com os mesmos hiperparâmetros em todos os datasets. A última linha é o melhor dos outros métodos de comparação em cada dataset: SGD de taxa fixa, Adam, os três schedulers, SPS e Armijo.

| Método | CIFAR-10 | CIFAR-100 | Fashion-MNIST | MNIST | Covertype | Passos do otimizador, × SGD |
|---|---|---|---|---|---|---|
| Polling (paper base) | 83,83% | 52,99% | 91,83% | 98,14% | 75,15% | 6,00 |
| Efficient Polling (nosso) | 83,76% | 25,07% | 91,84% | 98,82% | **75,79%** | 1,18–2,23 |
| Efficient Relative Polling (nosso) | **84,11%** | **53,05%** | 91,70% | 98,59% | 75,41% | 1,04–1,21 |
| Melhor método de comparação | 84,09%, Armijo | 53,03%, plateau | **92,70%**, Adam | **99,45%**, Adam | 75,64%, cosine | 1,00 |

- **O Efficient Relative Polling iguala o método base em todos os datasets, por uma fração do custo.** Fica acima do Polling base em quatro datasets e a menos de um desvio padrão dele no Fashion-MNIST, com no máximo 1,21× os passos de otimizador do SGD comum, contra 6× do Polling base. É o melhor método no CIFAR-10 e no CIFAR-100.
- **Nenhum método vence em todos.** O Adam lidera no MNIST e no Fashion-MNIST, onde os métodos de polling ficam 0,6 a 1,3 ponto atrás.
- **O Efficient Polling pode travar no menor candidato.** Quando a acurácia do batch no nível do acaso quase nunca distingue os candidatos, o desempate continua escolhendo `1e-5` e o backoff lê a repetição como escolha estável. No CIFAR-100 todas as seeds perderam épocas assim, e duas nunca treinaram.

Os [Resultados](docs/pt-br/results.md) trazem as tabelas e figuras completas, a ablação do gatilho, as runs de robustez à taxa inicial e o mecanismo da trava.

## Começando

O polling precisa reavaliar o modelo para pontuar um passo candidato, então, em vez do `optimizer.step()` puro, você passa uma **closure** que retorna `(loss, score)`: o mesmo contrato do `torch.optim.LBFGS`, mais o score a ser maximizado. O `make_closure` a constrói para você:

```python
import torch
from efficient_polling_lr_scheduler import EfficientPollingSGD, make_closure

model = MyModel().to(device)
loss_fn = torch.nn.CrossEntropyLoss()

# As LRs candidatas são, por padrão, {1e-5, 1e-4, 1e-3, 1e-2, 1e-1} em torno de lr.
optimizer = EfficientPollingSGD(model, lr=1e-3)

for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)
    info = optimizer.step(make_closure(model, loss_fn, inputs, targets))
    # info.lr, info.loss, info.polled, info.spike, info.rolled_back, ...
```

Sem cronograma de taxa de aprendizado, sem warmup, sem tuning: a LR é *medida*. O `PollingSGD` dá o método base, que faz poll a cada batch. O Efficient Relative Polling recebe uma taxa em vez de uma grade, e as duas extensões podem envolver um otimizador que você já usa:

```python
from efficient_polling_lr_scheduler import EfficientPollingOptimizer, EfficientRelativePollingSGD

# uma taxa e um multiplicador em vez de uma grade fixa
optimizer = EfficientRelativePollingSGD(model, lr=1e-3, multiplier=10.0, lr_max=1e-1)

# ou o Efficient Polling em volta de qualquer otimizador
optimizer = EfficientPollingOptimizer(
    torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9),
    candidate_lrs=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
    module=model,          # para restaurar os buffers de BatchNorm entre os testes
    max_poll_interval=64,  # teto do backoff; 0 faz poll a cada batch
)
```

O pacote também traz os dois baselines de comparação que derivam o tamanho do passo do batch atual, o Polyak step-size estocástico (SPS) e a busca de linha Armijo backtracking, e helpers opcionais de treino que também aceitam um otimizador comum e um `torch.optim.lr_scheduler`, para que todo método passe pelo mesmo loop:

```python
from efficient_polling_lr_scheduler import SPSSGD, ArmijoSGD, fit

optimizer = SPSSGD(model, lr=1e-3, max_lr=0.1)  # ou ArmijoSGD(model, lr=1e-3, lr_max=0.1)

history = fit(model, train_loader, val_loader, optimizer, loss_fn, epochs=150)
print(history.best_val_acc, sum(history.polls), sum(history.optimizer_steps))
```

### API

| Objeto | Papel |
|---|---|
| `EfficientPollingSGD` / `EfficientPollingOptimizer` | Efficient Polling: poll sob demanda, com a guarda anti-divergência |
| `PollingSGD` / `PollingOptimizer` | método base: poll a cada batch |
| `EfficientRelativePollingSGD` / `EfficientRelativePollingOptimizer` | Efficient Relative Polling (experimental): três candidatos em torno da taxa em uso, backoff sem teto no estilo TCP, restarts a partir do melhor ponto |
| `EfficientRelativeEpochPolling` | o mesmo método por época, conduzido por `fit(..., epoch_polling=...)` sobre um otimizador comum |
| `EfficientRelativeNarrowingPollingSGD` / `EfficientRelativeNarrowingPollingOptimizer` | Efficient Relative Narrowing Polling (experimental, ainda sem resultados): o Efficient Relative Polling com um pulo que estreita entre duas taxas quando o vencedor inverte ou o centro ganha, e volta a alargar quando ele segue no mesmo sentido |
| `SPSSGD` / `SPSOptimizer` | baseline de comparação: Polyak step-size estocástico |
| `ArmijoSGD` / `ArmijoOptimizer` | baseline de comparação: busca de linha Armijo backtracking estocástica |
| `TRIGGERS` | os três gatilhos de poll usados na ablação: `"backoff"` (padrão), `"fixed"`, `"random"` |
| `make_closure`, `accuracy`, `negative_loss` | closure do batch e critérios de seleção (acurácia ou perda) |
| `StepInfo`, `EpochStats`, `History` | telemetria: LR escolhida, polls, spikes, rollbacks, passos do otimizador |
| `fit`, `train_epoch`, `evaluate` | helpers opcionais do loop de treino, aceitando um `torch.optim.lr_scheduler` para os métodos de comparação com otimizador comum |
| `StateSnapshot` | salvamento/restauração exata de parâmetros, buffers e estado do otimizador |

**Observações.** Apesar do nome do pacote, estas classes **não** são subclasses de `torch.optim.lr_scheduler.LRScheduler`: elas envolvem o otimizador e são acionadas inteiramente por `optimizer.step(closure)`, então não existe um `scheduler.step()` separado para chamar depois. As LRs candidatas são absolutas e aplicadas a *todos* os parameter groups, sobrescrevendo LRs por grupo. Passe `module=` (ou o próprio modelo como primeiro argumento) sempre que o forward mutar buffers, para que os testes não contaminem as estatísticas de BatchNorm. A closure não deve chamar `backward()` nem `zero_grad()`: quem cuida disso é o otimizador.

## Documentação

- [Como funciona](docs/pt-br/methods.md): os três métodos, suas regras e hiperparâmetros, e o modelo de custo.
- [Resultados](docs/pt-br/results.md): todas as tabelas e figuras dos cinco datasets, a ablação do gatilho e as runs de robustez à taxa inicial.
- [Reproduzindo os experimentos](docs/pt-br/reproducing.md): configuração, datasets, linha de comando e notebook, e como o repositório está organizado.

## Citação

Se você usar este trabalho, cite:

```bibtex
@misc{souzasilva_efficient_polling_lr_scheduler,
  title  = {Efficient Polling-Based Learning Rate Optimization for Neural Networks},
  author = {de Souza Silva, Jos{\'e} R. and B. A. da Silva, Luiz Henrique and B. de Souza, Caio B. and Balieiro, Andson M.},
  year   = {2026},
  note   = {Centro de Inform{\'a}tica (CIn), Universidade Federal de Pernambuco (UFPE)},
  url    = {https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks}
}
```

O método de Polling base é de Tan, Choong e Lau, [Expediting Convergence via Polling Optimisation for Gradient Descent in Neural Networks](https://doi.org/10.3390/jeta4010001) (2025). Para citar especificamente o software, acrescente `note = {Pacote Python \texttt{efficient-polling-lr-scheduler}}` ou referencie [o projeto no PyPI](https://pypi.org/project/efficient-polling-lr-scheduler/).

## 🧑‍💻 Autores

| [<img src="https://github.com/luiz-linkezio.png" width=115><br><sub>Luiz Henrique</sub><br>](https://github.com/luiz-linkezio) <sub>Desenvolvedor</sub><br> <sub>[Linkedin](https://www.linkedin.com/in/lhbas/)</sub><br> <sub> Portfolio </sub> | [<img src="https://github.com/dev-joseronaldo.png" width=115><br><sub>José Ronaldo</sub><br>](https://github.com/Dev-JoseRonaldo) <sub>Desenvolvedor</sub><br> <sub>[Linkedin](https://www.linkedin.com/in/devjoseronaldo/)</sub><br> <sub>[Portfólio](https://joseronaldo.netlify.app/)</sub> |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |

Universidade Federal de Pernambuco, Recife, Brasil. O artigo credita ainda Caio B. B. de Souza (UPE) e Andson M. Balieiro (CIn/UFPE); ver [Citação](#citação).

## Licença

MIT, veja o arquivo [LICENSE](LICENSE) deste repositório.
