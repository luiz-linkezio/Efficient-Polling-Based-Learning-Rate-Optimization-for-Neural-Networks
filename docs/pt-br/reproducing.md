# Reproduzindo os experimentos

[🇺🇸 English](../reproducing.md) · [README](../../README(pt-br).md) · [Como funciona](methods.md) · [Resultados](results.md)

- [Configuração](#configuração)
- [Conjuntos de dados](#conjuntos-de-dados)
- [Execução](#execução): pela linha de comando ou pelo notebook
- [Rodadas de learning rate](#rodadas-de-learning-rate): todo método no SGD ou no Adam, a partir de uma taxa, uma tabela por taxa
- [Execução num cluster SLURM](#execução-num-cluster-slurm): o estudo inteiro num job, várias execuções por GPU
- [Configuração experimental](#configuração-experimental)
- [Estrutura do repositório](#estrutura-do-repositório)

## Configuração

Para *usar* os métodos, basta o pacote (Python 3.10+, PyTorch 2.0+):

```bash
pip install efficient-polling-lr-scheduler
```

Para *reproduzir os experimentos*, clone o repositório e instale-o com os extras. Uma GPU compatível com CUDA é recomendada (CPU funciona, mas é lento):

```bash
git clone https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks.git
cd Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,benchmark]" jupyter
nbstripout --install
```

A última linha faz o git tirar as saídas do notebook sempre que ele é commitado, então os logs e as animações de uma execução nunca entram no histórico; a CI recusa um notebook commitado com saídas. Rode a suíte de testes com `pytest`.

## Conjuntos de dados

Cada dataset é lido direto dos arquivos que seus autores publicam: sem torchvision, sem download escondido dentro da execução. Quatro são de imagem; o **Covertype não é**. Ele tem 581.012 linhas com 54 atributos cartográficos e sete tipos de cobertura florestal, o que troca a CNN por um MLP e tira a comparação da visão computacional:

| Dataset | Entrada | Classes | O diretório de dados deve conter | Origem |
|---|---|---|---|---|
| `cifar10` (padrão) | 32×32 colorida | 10 | `data_batch_1`…`data_batch_5`, `test_batch` | [cs.toronto.edu](https://www.cs.toronto.edu/~kriz/cifar.html) |
| `cifar100` | 32×32 colorida | 100 | `train`, `test` (fine labels) | [cs.toronto.edu](https://www.cs.toronto.edu/~kriz/cifar.html) |
| `mnist` | 28×28 cinza | 10 | os quatro arquivos IDX | [espelho ossci](https://ossci-datasets.s3.amazonaws.com/mnist/) |
| `fashion_mnist` | 28×28 cinza | 10 | os quatro arquivos IDX | [zalandoresearch](https://github.com/zalandoresearch/fashion-mnist) |
| `covertype` | 54 atributos | 7 | `covtype.data` (ou `.gz`) | [UCI](https://archive.ics.uci.edu/dataset/31/covertype) |

```bash
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz
```

Os arquivos IDX são lidos compactados ou não, com hífen (`train-images-idx3-ubyte`) ou com ponto (`train-images.idx3-ubyte`), então não há nada para descompactar nem renomear. O Covertype segue o protocolo publicado: as primeiras 15.120 linhas são o que a varredura usa para treinar (com o split 90/10 treino/val dela por dentro), e as 565.892 restantes são o teste. Imagens são normalizadas por canal e o Covertype por atributo, sempre com estatísticas do split de treino. As 44 colunas de área silvestre e tipo de solo do Covertype são flags 0/1 e ficam nessa escala: z-score numa flag ligada em meia dúzia de linhas vira um valor na casa das centenas, e dois tipos de solo nem aparecem nas 15.120 linhas de treino, o que com z-score puro mandaria para 1e8 uma linha de teste que os tenha.

## Execução

As execuções são registradas em `results/<dataset>/`, um JSON por par `(método, seed)`, e o melhor checkpoint de cada uma vai para `models/`. Uma execução que já tem registro é lida de volta em vez de retreinada, então uma varredura pode parar e retomar a qualquer momento, por qualquer das duas entradas. As duas rodam o mesmo código, o pacote `benchmark`.

### Pela linha de comando

Rode a partir da raiz do repositório:

```bash
python -m benchmark --data-dir /caminho/para/cifar-10-batches-py --seeds 42 43 44 45 46
# um método, uma seed, execução mais curta:
python -m benchmark --data-dir ... --methods efficient --seeds 42 --epochs 20
# outro dataset, com as ablações calibradas como nas execuções registradas dele:
python -m benchmark --dataset mnist --data-dir /caminho/para/MNIST --seeds 42 43 44 45 46 --calibrate-ablations
# as execuções de robustez à taxa inicial, registradas como <método>_lr<taxa>:
python -m benchmark --data-dir ... --methods efficient efficient_relative --seeds 42 --lr 1e-5
# uma rodada de learning rate, registrada em results/<dataset>/rounds/adam_lr1/:
python -m benchmark --data-dir ... --seeds 42 43 44 45 46 --round adam:1
# SPS e Armijo com teto 1e-7, registrados em results/<dataset>/ceilings/sgd_lr1e-07/:
python -m benchmark --data-dir ... --seeds 42 43 44 45 46 --ceiling 1e-7
```

A varredura vai seed por seed e imprime a tabela no fim. Todo hiperparâmetro tem uma flag, e `python -m benchmark --help` lista todas com os valores que as execuções registradas usaram. O `--calibrate-ablations` coloca as duas ablações do gatilho na taxa de poll que o `efficient` mediu na primeira seed, então essa execução precisa existir antes; na ordem padrão dos métodos, ela existe.

As figuras são desenhadas a partir dos registros, então uma figura nunca pode discordar da tabela:

```bash
python -m benchmark.plots                   # CIFAR-10, em images/cifar10/
python -m benchmark.plots --dataset mnist   # em images/mnist/
```

### Pelo notebook

```bash
jupyter notebook notebooks/benchmark.ipynb
```

Defina `DATASET` e a entrada correspondente de `DATA_DIRS` nas primeiras células de código e rode as células de cima para baixo: Setup, Hyperparameters, Data, Model, Training runs (uma célula por família de métodos), Figures e animações, Test. O notebook monta um `Experiment` do pacote `benchmark` e o chama, então o que ele roda é exatamente o que a linha de comando roda.

> **Reprodutibilidade.** Cinco seeds (42–46) fixam, cada uma, a inicialização dos pesos, o embaralhamento dos dados e o split treino/val, de modo que numa dada seed todos os métodos partem dos mesmos pesos e veem a mesma ordem de batches.

## Rodadas de learning rate

A tabela principal fixa o otimizador base no SGD e a taxa inicial em `1e-3`, e varia o método. Uma rodada fixa os dois em outros valores: todo método dá passos com um mesmo otimizador base e recebe uma mesma learning rate, então a rodada mostra como cada método lida com uma taxa alta ou baixa demais sem misturar o otimizador na comparação. São seis, SGD e Adam partindo cada um de `1`, `1e-3` e `1e-7`, definidas em `benchmark/rounds.py`. Cada método recebe a taxa da rodada onde quer que peça uma:

| Método | O que a taxa da rodada é |
|---|---|
| Taxa fixa | a taxa da execução inteira, no otimizador da rodada |
| Cosine annealing, step decay, ReduceLROnPlateau | onde o schedule começa, em vez do `1e-1` da tabela principal |
| Polling, Efficient Polling | o centro da grade de candidatas, duas décadas para cada lado, então a partir de `1` a grade chega a `1e+2`, como nas execuções de taxa inicial |
| Ablações do gatilho | como no Efficient Polling, com a taxa de poll calibrada pela execução do Efficient Polling da própria rodada |
| Efficient Relative Polling, por batch e por época | onde ele começa, sob o teto de `1e-1` da tabela principal, que as rodadas de `1` sobem para `1` porque o método recusa começar acima do teto |
| Efficient Relative Narrowing Polling | como o Efficient Relative Polling por batch, sob o mesmo teto |

Adam, SPS e Armijo ficam fora das rodadas. O Adam é o otimizador base de metade delas. SPS e Armijo nunca leem uma taxa inicial: o passo de Polyak a sobrescreve no primeiro batch, e a busca em linha começa todo batch pelo teto. O teste deles move esse teto pelas mesmas três taxas, só no SGD, porque as duas fórmulas supõem que o passo segue o gradiente, e o do Adam não segue.

Uma rodada registra em `results/<dataset>/rounds/<otimizador>_lr<taxa>/` e o teste de teto em `results/<dataset>/ceilings/sgd_lr<taxa>/`, com os checkpoints nas mesmas pastas dentro de `models/`, então nenhum dos dois se mistura com a tabela principal.

As seis rodadas e os três tetos são um estudo só, rodado por um comando e lido como **uma tabela por taxa inicial**: cada método das rodadas no SGD, depois SPS e Armijo com aquela taxa como teto, depois cada método das rodadas no Adam, cada linha com as colunas da tabela principal. Uma execução ainda não feita aparece como traço.

```bash
python -m benchmark.rounds --data-root ~/Datasets                      # todos os datasets
python -m benchmark.rounds --data-root ~/Datasets --datasets covertype  # um
python -m benchmark.rounds --datasets covertype --tables-only           # as tabelas do que já foi registrado
```

O comando roda todas as rodadas e tetos dos datasets pedidos num único pool de processos, vários por GPU (veja abaixo), pula execuções já registradas e imprime as tabelas no fim. No notebook, uma célula depois das execuções de taxa inicial varre o estudo inteiro do `DATASET`, uma execução depois da outra, e imprime as mesmas tabelas; a seção Figures desenha as figuras de cada rodada em `images/<dataset>/rounds/<otimizador>_lr<taxa>/`.

**Custo.** Pelas execuções registradas, uma rodada custa o que a tabela principal custa sem Adam, SPS e Armijo: cerca de 8 h de CIFAR-10 para cinco seeds, e umas 57 h de execução nos cinco datasets, quatro dos quais dividiram a GPU três por vez. Uma rodada pode demorar mais, porque o passo do Adam é mais lento que o do SGD e um método que trava na menor candidata faz poll a cada dois batches. SPS e Armijo somam cerca de 1,5 h de CIFAR-10 por teto, 15 h nos cinco datasets. Esses números são anteriores ao Efficient Relative Narrowing Polling, que ainda não tem execuções registradas; num teste de fumaça de três épocas no CIFAR-10 ele fez poll em 2,7 vezes mais batches que o Efficient Relative Polling (48% contra 18%).

## Execução num cluster SLURM

O `python -m benchmark.rounds` acima, e o `python -m benchmark.pool` para uma varredura avulsa (ele aceita as flags de `python -m benchmark`), fazem as execuções lado a lado, um processo por `(método, seed)`, quatro por GPU a menos que `--workers` diga outra coisa. Cada processo é o `python -m benchmark` daquele método e seed, então os registros são os que uma varredura sequencial escreveria, menos o `s_per_epoch`: execuções que dividem uma GPU atrasam umas às outras. Execuções com registro são puladas, as ablações calibradas esperam a execução do Efficient Polling da própria varredura na primeira seed, e cada execução grava seu log em `logs/`, numa pasta que espelha a do seu registro em `results/`.

`slurm/benchmark.sbatch` roda o estudo inteiro, todas as rodadas e tetos de todos os datasets, como um job só. No nó de login, clone o repositório e crie o ambiente que o job ativa:

```bash
git clone --branch dev https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks.git
cd Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks
python3 -m venv .venv && .venv/bin/pip install -e ".[benchmark]"
mkdir -p logs
```

Serve qualquer Python 3.10 ou mais novo, desde que os nós de computação enxerguem o interpretador que criou o ambiente. O `uv` monta o mesmo ambiente mais rápido, mas ele tranca o cache e o ambiente em que instala, e uma trava de arquivo fica pendurada para sempre num home montado por NFS com o serviço de travas quebrado, que é o que aconteceu no Apuana em setembro de 2026. O `pip` não usa esse tipo de trava.

O job lê os datasets de `~/Datasets/`, nas pastas que o `DATA_DIRS` do notebook nomeia (`DATA_ROOT` aponta para outro lugar). De uma máquina que os tenha:

```bash
rsync -av ~/Datasets/{cifar-10-python,cifar-100-python,MNIST,fashion-mnist,covertype} <usuário>@<nó de login>:Datasets/
```

Depois, submeta uma vez e acompanhe:

```bash
sbatch --nodelist=cluster-node7 slurm/benchmark.sbatch                 # todos os datasets
DATASETS="covertype cifar10" sbatch --nodelist=cluster-node7 slurm/benchmark.sbatch
squeue -u $USER
tail -f logs/polling-benchmark-<id do job>.out
```

O job pede duas GPUs, 16 CPUs e 64 GB, os limites da QoS simple do cluster para o qual foi escrito, o Apuana do CIn/UFPE, e até sete dias na `long-simple`: o estudo inteiro leva uns dois dias em duas A100, mais do que a `short-simple` permite. As GPUs não têm tipo no SLURM lá, então o nó se escolhe com `--nodelist`, e as A100 da `long-simple` são as do `cluster-node7`. Opções na linha de comando do `sbatch` sobrepõem as do arquivo. Um job preemptado volta para a fila, e um job submetido de novo depois de cancelado continua das execuções que não tinham terminado. Ele termina imprimindo uma tabela por taxa inicial de cada dataset. Os registros são escritos no clone do cluster; traga-os de volta com

```bash
rsync -av <usuário>@<nó de login>:Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/results/ results/
```

## Configuração experimental

- **Datasets:** CIFAR-10 e CIFAR-100, 45.000 treino / 5.000 val / 10.000 teste. MNIST e Fashion-MNIST, 54.000 / 6.000 / 10.000. Covertype, 13.608 / 1.512 / 565.892.
- **Modelos:** `SimpleCNN` nos datasets de imagem, uma CNN de 5 camadas (canais conv 64→64→128→128→256, kernels `3×3`, ReLU, MaxPool, AdaptiveAvgPool, cabeça Linear) com **557.898 parâmetros** no CIFAR-10, 581.028 no CIFAR-100 e 556.746 no MNIST e no Fashion-MNIST. Só a primeira convolução e o classificador mudam com o dataset; o pooling global faz o lado da imagem nunca entrar na conta. O Covertype usa o `SimpleMLP` (512→256→128, 193.287 parâmetros). Nenhum dos dois tem batch norm nem dropout, para que o otimizador seja a única fonte de adaptação.
- **Otimizador:** SGD puro (sem momentum, sem weight decay) para o método proposto e o replicado, batch 64, LR base `1e-3`, 150 épocas.
- **Métodos de comparação:** Adam, cosine annealing, step decay, ReduceLROnPlateau, SPS (Polyak step-size), Armijo backtracking line search e o método de Polling base replicado, oito no total, mais duas variantes de ablação do gatilho do Efficient Polling. Os três schedulers começam em `1e-1`, e SPS e Armijo são limitados a esse valor.
- **Seeds:** cinco (42–46) por configuração: treze configurações, 65 execuções por dataset e 325 no total, mais quatro runs de robustez à taxa inicial no CIFAR-10, seed 42.
- **Hardware:** uma única NVIDIA GeForce RTX 5070 (12 GB). CIFAR-100, Fashion-MNIST, MNIST e Covertype rodaram três por vez nela, então seus tempos não são relatados.

## Estrutura do repositório

```
.
├── src/efficient_polling_lr_scheduler/   # o pacote publicado no PyPI
│   ├── polling.py             # método base (Tan et al.)
│   ├── efficient.py           # Efficient Polling, incl. os gatilhos fixo/aleatório
│   ├── efficient_relative.py  # Efficient Relative Polling, por batch e por época
│   ├── efficient_relative_narrowing.py  # Efficient Relative Narrowing Polling
│   ├── baselines.py           # otimizadores de comparação SPS e Armijo backtracking
│   ├── _snapshot.py           # salvamento/restauração exata do estado nos testes
│   ├── closures.py            # closures do batch e critérios de seleção
│   └── training.py            # helpers opcionais fit/train_epoch/evaluate
├── benchmark/                 # o experimento, rodado por python -m benchmark e pelo notebook
│   ├── datasets.py            # os cinco leitores de dataset, sem torchvision
│   ├── models.py              # SimpleCNN e SimpleMLP
│   ├── methods.py             # todas as configurações e seus hiperparâmetros
│   ├── sweep.py               # roda configurações sobre seeds, registra e relê as execuções
│   ├── rounds.py              # o estudo de learning rate: rodadas, tetos, uma tabela por taxa inicial
│   ├── pool.py                # execuções lado a lado, várias por GPU
│   └── plots.py               # as figuras, desenhadas a partir dos registros
├── slurm/benchmark.sbatch     # o estudo de learning rate inteiro como um job do cluster
├── notebooks/benchmark.ipynb  # conduz o pacote benchmark, commitado sem saídas
├── tests/                     # suíte pytest do pacote e do benchmark
├── results/<dataset>/         # um JSON por (método, seed); rounds/ e ceilings/ guardam as rodadas
├── images/<dataset>/          # as figuras de cada dataset
├── docs/                      # como funciona, resultados, reprodução; pt-br/ tem o português
├── models/                    # melhores checkpoints (.pt, gitignored)
├── logs/                      # logs do pool e dos jobs do cluster (gitignored)
├── pyproject.toml
├── CHANGELOG.md
├── README.md
└── README(pt-br).md
```

Para acrescentar um método, crie uma subclasse de `PollingOptimizer` em `src/`. Ela herda o snapshot e a restauração exatos, o protocolo de otimizador e a telemetria `StepInfo` que o `fit()` agrega; preencha `optimizer_steps` com o que o batch realmente custou e o modelo de custo continua honesto. Depois acrescente uma entrada em `LABELS` e um ramo em `build_optimizer()` no `benchmark/methods.py`, e o método entra na varredura, nas tabelas e nas figuras.
