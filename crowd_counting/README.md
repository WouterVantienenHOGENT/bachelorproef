# CSRNet-experimenten

De scripts gebruiken automatisch mixed precision op CUDA en schrijven iedere run
naar een afzonderlijke map. Gebruik voor bedoelde GPU-runs altijd expliciet
`--device cuda`; de training stopt dan met een duidelijke fout in plaats van
stilletjes op CPU verder te gaan.

## Geteste GPU-omgeving

- NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB)
- NVIDIA-driver 595.71.05
- PyTorch 2.13.0+cu130
- torchvision 0.28.0+cu130

`requirements.txt` bevat bruikbare minimumversies voor ontwikkeling.
`requirements-final.txt` legt de exacte Pythonomgeving van de eindruns vast.

## Dataset voorbereiden

Start vanuit de repositoryroot. Het voorbereidingsscript maakt naast de drie
splits een `dataset_manifest.json` met de bestandslijsten, splitseed en gebruikte
Gaussische sigma. Een run neemt de hash en kerngegevens van dit manifest op in
`config.json`. Het script weigert bestaande bestanden met een andere splitsing te
vermengen.

```bash
bachproef/.venv/bin/python bachproef/prepare_shanghaitech.py \
  --source bachproef/data/raw/ShanghaiTech/part_B \
  --output bachproef/data/processed/shanghaitech_part_b \
  --val-fraction 0.1 --seed 42 --sigma 4.0
```

## Snelle controles

Voer voor iedere lange run de kleine regressietests uit:

Start vanuit de repositoryroot:

```bash
PYTHONPATH=crowd_counting bachproef/.venv/bin/python \
  -m unittest discover -s crowd_counting/tests -v
```

De tests controleren de behoudswet van density maps, padding van volledige
evaluatiebeelden, de foutstatistieken en het atomisch schrijven van checkpoints.

Controleer daarna de GPU-omgeving:

```bash
nvidia-smi

bachproef/.venv/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Volledige GPU-run

Start vanuit de map `bachproef`. Kies voor iedere run een nieuwe naam. De optie
`--deterministic` vraagt reproduceerbare PyTorch-algoritmen; als een gebruikte
CUDA-operatie dit niet ondersteunt, stopt de run met een fout in plaats van
onopgemerkt niet-deterministisch verder te gaan.

```bash
TORCH_HOME=data/torch_cache .venv/bin/python ../crowd_counting/train.py \
  --train-images data/processed/shanghaitech_part_b/train/images \
  --train-densities data/processed/shanghaitech_part_b/train/density_maps \
  --val-images data/processed/shanghaitech_part_b/val/images \
  --val-densities data/processed/shanghaitech_part_b/val/density_maps \
  --output-dir runs/shanghaitech_part_b_gpu_512_b4_e100_seed42 \
  --epochs 100 --batch-size 4 --crop-size 512 \
  --lr 1e-5 --workers 4 --seed 42 --device cuda \
  --deterministic --checkpoint-every 25
```

Een runmap bevat:

- `config.json`: configuratie, datasetgrootte en hardware;
- `metrics.csv`: loss, MAE, RMSE, tijden en GPU-geheugen per epoch;
- `summary.json`: de volledige beste én meest recente metriekrij;
- `best.pt`: checkpoint met de laagste validatie-MAE;
- `last.pt`: meest recente hervatbare checkpoint;
- `epoch_NNN.pt`: periodieke snapshots volgens `--checkpoint-every`.

Checkpoints bevatten het model, de optimizer, mixed-precision scaler,
random-generatorstates en de beste metriekrij. Hierdoor kan een afgebroken run
vanaf de volgende epoch worden hervat:

```bash
TORCH_HOME=data/torch_cache .venv/bin/python ../crowd_counting/train.py \
  --train-images data/processed/shanghaitech_part_b/train/images \
  --train-densities data/processed/shanghaitech_part_b/train/density_maps \
  --val-images data/processed/shanghaitech_part_b/val/images \
  --val-densities data/processed/shanghaitech_part_b/val/density_maps \
  --resume runs/shanghaitech_part_b_gpu_512_b4_e100_seed42/last.pt \
  --epochs 100 --batch-size 4 --crop-size 512 \
  --lr 1e-5 --workers 4 --seed 42 --device cuda \
  --deterministic --checkpoint-every 25
```

`--epochs` is bij hervatten het totale doel, niet het aantal extra epochs.

Vergelijk alle bewaarde runs:

```bash
.venv/bin/python ../crowd_counting/compare_runs.py \
  --runs-dir runs --csv runs/comparison.csv
```

## Trainingsgrafiek

```bash
MPLCONFIGDIR=/tmp/matplotlib-csrnet \
  .venv/bin/python ../crowd_counting/plot_metrics.py \
  --metrics runs/shanghaitech_part_b_gpu_512_b4_e100_seed42/metrics.csv \
  --output runs/shanghaitech_part_b_gpu_512_b4_e100_seed42/training_curves.png \
  --title "CSRNet op ShanghaiTech Part B"
```

De figuur bevat afzonderlijke panelen voor de trainingsloss en voor de
validatie-MAE/RMSE. De beste epoch volgens validatie-MAE wordt aangeduid.

## Eenmalige finale testevaluatie

Gebruik de officiële testset pas nadat alle keuzes op basis van de validatieset
vastliggen. `evaluate.py` schrijft zowel geaggregeerde statistieken als één rij per
beeld weg. Het rapporteert MAE, RMSE, bias, mediaan, 95e percentiel, grootste fout
en model-forwardtijden. De samenvatting selecteert bovendien deterministisch een
beeld met de laagste, mediane en hoogste absolute fout voor de kwalitatieve analyse.

```bash
.venv/bin/python ../crowd_counting/evaluate.py \
  --checkpoint runs/shanghaitech_part_b_gpu_512_b4_e100_seed42/best.pt \
  --images data/processed/shanghaitech_part_b/test/images \
  --densities data/processed/shanghaitech_part_b/test/density_maps \
  --output-dir runs/shanghaitech_part_b_gpu_512_b4_e100_seed42/test_evaluation \
  --split-name test --workers 4 --warmup-runs 3 --device cuda
```

Maak daarna de scatter- en residufiguur:

```bash
MPLCONFIGDIR=/tmp/matplotlib-csrnet \
  .venv/bin/python ../crowd_counting/plot_evaluation.py \
  --predictions runs/shanghaitech_part_b_gpu_512_b4_e100_seed42/test_evaluation/predictions.csv \
  --output runs/shanghaitech_part_b_gpu_512_b4_e100_seed42/test_evaluation/evaluation_scatter.png \
  --title "CSRNet: voorspelde en werkelijke tellingen"
```

## Inferencebenchmark op één beeld

```bash
MPLCONFIGDIR=/tmp/matplotlib-csrnet \
  .venv/bin/python ../crowd_counting/infer.py \
  --checkpoint runs/shanghaitech_part_b_gpu_512_b4_e100_seed42/best.pt \
  --image data/processed/shanghaitech_part_b/test/images/IMG_6.jpg \
  --output-dir predictions/gpu_512_b4_e100_seed42 \
  --device cuda --warmup-runs 3 --timed-runs 20
```

Naast de density map en een JSON-resultaat wordt iedere benchmark automatisch
toegevoegd aan `runs/inference_benchmarks.csv`. GPU-metingen synchroniseren vóór
en na de model-forward, zodat asynchrone CUDA-uitvoering de gemeten tijd niet
kunstmatig verlaagt. Het script toont en bewaart naast milliseconden per beeld
ook de modeldoorvoer in FPS. Dit is `1000 / mean_forward_ms` en omvat alleen de
voorwaartse modelberekening.

## Raspberry Pi 5-benchmark

Voor de gemeten Raspberry Pi 5 met 64-bit Raspberry Pi OS en Python 3.11 is een
afzonderlijk CPU-only requirementsbestand beschikbaar. Maak op de Pi een eigen
virtuele omgeving en voer daarna dezelfde inferentieopdracht uit als op de
laptop:

```bash
python3 -m venv .venv-rpi
.venv-rpi/bin/pip install --upgrade pip
.venv-rpi/bin/pip install -r crowd_counting/requirements-rpi.txt

MPLCONFIGDIR=/tmp/matplotlib-csrnet-rpi \
  .venv-rpi/bin/python crowd_counting/infer.py \
  --checkpoint bachproef/runs/shanghaitech_part_b_gpu_512_b4_e100_seed42/best.pt \
  --image bachproef/data/processed/shanghaitech_part_b/test/images/IMG_6.jpg \
  --output-dir bachproef/predictions/rpi \
  --device cpu --warmup-runs 3 --timed-runs 20 \
  --benchmark-csv bachproef/runs/inference_benchmarks.csv
```

Controleer vóór en na de meting de temperatuur- en throttlingstatus met
`vcgencmd measure_temp` en `vcgencmd get_throttled`. De vastgelegde meting op
20 augustus 2026 duurde gemiddeld 10.948,63 ms per beeld, of 0,091 FPS. Tijdens
de langdurige belasting traden thermische throttling en frequentiebegrenzing op.
De volledige systeem-, software-, timing- en thermische metadata staan in
`bachproef/runs/rpi5_cpu_epoch39_inference/summary.json`.

De platformvergelijking opnieuw genereren:

```bash
MPLCONFIGDIR=/tmp/matplotlib-csrnet-platforms \
  bachproef/.venv/bin/python crowd_counting/plot_inference_platforms.py \
  --input bachproef/runs/platform_inference_comparison.csv \
  --output graphics/csrnet-inference-platformvergelijking.png
```
