# Natelling Grote Markt met voorgetraind P2PNet

Op 14 oktober 2026 zijn de oorspronkelijke foto en het bestaande finale CSRNet-checkpoint vergeleken met het officiële P2PNet-checkpoint `SHTechA.pth`. P2PNet is vooraf getraind op ShanghaiTech Part A; er is hier geen extra training uitgevoerd. CSRNet is de eigen finale run op Part B, met het checkpoint uit epoch 39. Deze vergelijking verandert dus zowel architectuur als trainingsdata en trainingsprocedure.

De handmatige telling bevat 738 markeringen en is onvolledig. Een voorspelling boven 738 is daardoor niet automatisch een overschatting. Dicht bij 738 uitkomen bewijst evenmin dat alle voorspelde hoofdposities correct zijn.

| Model | Werkelijke invoer | Telling | Verschil met 738 annotaties | Gem. modeltijd |
|---|---|---:|---:|---:|
| CSRNet, finale run | 1024 × 683 | 544,2 | −193,8 | 95,35 ms |
| P2PNet, officieel ShanghaiTech A | 1024 × 640 | 745 | +7 | 99,18 ms |
| CSRNet, finale run | 2048 × 1366 | 879,3 | +141,3 | 374,65 ms |
| P2PNet, officieel ShanghaiTech A | 2048 × 1280 | 808 | +70 | 385,82 ms |

Alle metingen gebruiken dezelfde RTX 3050 Ti, één opwarmrun en drie gemeten berekeningen. De tijden meten alleen de modelberekening en zijn een korte verkennende meting. De foto wordt eerst met behoud van beeldverhouding beperkt tot een maximale zijde van 1024 of 2048 pixels. CSRNet past vervolgens padding tot een veelvoud van acht toe. P2PNet volgt de officiële demo en verkleint beide zijden naar veelvouden van 128. De modellen krijgen dus niet exact dezelfde beeldafmetingen. De P2PNet-scoregrens blijft op de officiële waarde 0,5, zonder optimalisatie op de handmatige telling.

P2PNet telt bij de kleinere invoer 745 personen. Bij grotere invoer stijgt dit naar 808. Volgens de officiële telregel worden alle punten met een softmax-score groter dan 0,5 geteld, ook buiten het beeld. Bij 1024 pixels zijn er nul zulke punten; bij 2048 pixels zijn er drie, zodat 805 van de 808 voorspelde punten binnen de foto liggen. Deze aantallen staan in de individuele JSON-bestanden. De CSV-coördinaten zijn teruggeschaald naar de oorspronkelijke foto van 5120 × 3415 pixels.

Bij visuele controle van de P2PNet-uitvoer op 1024 pixels zijn ook foutpositieve punten op dakornamenten zichtbaar. De telling van 745 mag daarom niet als een bevestigde telling van 745 echte personen worden gelezen.

Ook CSRNet verandert sterk met de resolutie: van 544,2 naar 879,3. De eerdere onderschatting geldt dus voor de eerder gekozen invoer van 1024 × 683 pixels. Deze verkenning motiveert een gecontroleerde evaluatie op meerdere volledig geannoteerde foto's en een controle van de individuele P2PNet-punten. Ze bewijst niet dat de grotere voorspelling of het model dat het dichtst bij 738 uitkomt de juiste telling geeft.

## Bestanden en herhalen

Git bevat per submap de numerieke uitvoer en de instellingen in `origineel_inference.json`. `comparison.csv` en `comparison.json` bundelen de resultaten en hashes. De visualisaties worden lokaal gegenereerd en niet meegecommit. Ook de oorspronkelijke foto wordt niet meegeleverd. Plaats een rechtmatig verkregen kopie lokaal als `graphics/voorbeelden/grote-markt-sint-niklaas/origineel.png`; de bronverwijzing en instructies staan in [crowd_counting/README.md](../../../crowd_counting/README.md).

Vanuit de repositoryroot, na installatie via `crowd_counting/setup_models.py`:

```bash
MPLBACKEND=Agg bachproef/.venv/bin/python crowd_counting/infer.py \
  --checkpoint models/p2pnet/p2pnet_shanghaitech_a.pth \
  --image graphics/voorbeelden/grote-markt-sint-niklaas/origineel.png \
  --max-edge 1024 --threshold 0.5 --warmup-runs 1 --timed-runs 3 \
  --output-dir bachproef/predictions/modelvergelijking/p2pnet-1024 \
  --benchmark-csv ''
```

Gebruik `--max-edge 2048` en een andere outputmap voor de hogere resolutie. Voor CSRNet is het checkpoint `models/csrnet/csrnet_finaal.pt`.

## Herkomst en controle

- Officiële bron: https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet
- Vastgelegde commit: `5c91a81ca062b1c7fd3db3ad1c55b1c21f0a7455`.
- SHA-256 van het ongewijzigde P2PNet-checkpoint: `506047732b128ff09efef18e94bfacbe35fcfef300e5e9eeeece259b0488c63f`.
- De inferentieadapter is numeriek vergeleken met de originele netwerkklassen op dezelfde CPU-invoer van 128 × 256 pixels. Zowel `pred_logits` als `pred_points` waren exact gelijk, met absolute en relatieve tolerantie nul.
- De oorspronkelijke CSRNet-telling van 544,1508 is met de nieuwe gedeelde inferentie opnieuw verkregen.
- De code bevat regressietests voor modelontdekking, hergebruik, architectuurcontrole, coördinatenschaling, scoregrens en CPU-ankers op een systeem met CUDA.
