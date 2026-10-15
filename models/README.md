# Modellen voor de telapp

De Streamlit-app leest deze map automatisch. Ondersteunde architecturen zijn
CSRNet en P2PNet met VGG16-BN en 2 × 2 ankerpunten, zoals het officiële checkpoint.
Gewichten van andere architecturen vereisen eerst een eigen adapter.

```text
models/
  csrnet/
    csrnet_finaal.pt
    csrnet_finaal.json
  p2pnet/
    p2pnet_shanghaitech_a.pth
    p2pnet_shanghaitech_a.json
```

Kopieer een ondersteund `.pt`- of `.pth`-bestand naar de juiste submap en klik
in de app op **Modellen beheren → Modellen vernieuwen**. De bestandsnaam wordt
als naam gebruikt, met spaties in plaats van underscores en koppeltekens.
Voor een eigen naam en beschrijving kun je naast de gewichten een gelijknamig
JSON-bestand zetten, bijvoorbeeld `mijn_model.json` naast `mijn_model.pth`:

```json
{
  "name": "P2PNet · eigen checkpoint",
  "architecture": "p2pnet",
  "description": "Beschrijving van dataset en training."
}
```

Een JSON-bestand is optioneel als het gewicht direct in `csrnet/` of `p2pnet/`
staat. Een afwijkende of diepere map vereist `architecture` in de JSON.
Ongeldige metadata worden in de app gemeld. Het wijzigen van een naam wijzigt
de gewichten niet. Gebruik alleen vertrouwde checkpointbestanden.

## Grootte en downloadlocatie

De gewichten worden niet in Git opgenomen. Alleen modelnamen, herkomst,
SHA-256-hashes en de installatiecode staan in de repository.

| Gewichten | Bestandsgrootte | Beschikbaarheid |
|---|---:|---|
| P2PNet · ShanghaiTech A | 82,4 MiB | [Download officieel checkpoint](https://raw.githubusercontent.com/TencentYoutuResearch/CrowdCounting-P2PNet/5c91a81ca062b1c7fd3db3ad1c55b1c21f0a7455/weights/SHTechA.pth) |
| CSRNet · finale training | 62,1 MiB | Eigen training; nog geen publieke download |
| CSRNet · CPU-kalibratie | 62,1 MiB | Eigen training; nog geen publieke download |
| CSRNet · GPU-kalibratie | 62,1 MiB | Eigen training; nog geen publieke download |
| CSRNet · één epoch | 62,1 MiB | Eigen training; nog geen publieke download |

De vijf bestanden nemen samen ongeveer 330,6 MiB in. Plaats de P2PNet-download
in `models/p2pnet/p2pnet_shanghaitech_a.pth`, of gebruik het installatiescript.
Het script downloadt P2PNet automatisch, maar kan de eigen CSRNet-modellen alleen
kopiëren als hun trainingsruns lokaal aanwezig zijn. Op een verse checkout
zonder die runs is dus alleen P2PNet beschikbaar na installatie.

Om de eigen CSRNet-gewichten te delen, kunnen de bestanden later als GitHub
Release-assets of via een externe opslaglocatie worden gepubliceerd. Er is nu
nog geen downloadlocatie voor deze exacte checkpoints. De training kan worden
herhaald met de instructies in [crowd_counting/README.md](../crowd_counting/README.md).

## Eerste installatie

Vanuit de repositoryroot:

```bash
bachproef/.venv/bin/python crowd_counting/setup_models.py
```

Dit downloadt het officiële P2PNet-checkpoint van een vaste Git-commit en
controleert de SHA-256-hash. Bestaande CSRNet-trainingsruns worden als compacte
kopieën zonder optimizerstaat geïnstalleerd; de originele runs blijven intact.
Bestaande gewichten en zelf aangepaste namen worden niet overschreven.
Gewichten staan in `.gitignore`; JSON-bestanden met namen en herkomst worden
wel bijgehouden. Een verse checkout moet de gewichten dus opnieuw installeren.

P2PNet is volgens de [upstreamlicentie](../crowd_counting/vendor/p2pnet/LICENSE)
uitsluitend bestemd voor academisch onderzoek. Er is geen lokale bijtraining
uitgevoerd op dit model.
