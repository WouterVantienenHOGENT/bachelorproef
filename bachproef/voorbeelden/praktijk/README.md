# Praktijkvoorbeelden

De twee fotopagina's in sectie 4.7 worden gegenereerd met hetzelfde protocol:
vier eigen CSRNet-checkpoints en het officiële P2PNet-checkpoint voor
ShanghaiTech Part A, maximale beeldzijde 1024, drie opwarmruns en twintig
gemeten forward-passes op CUDA. De heatmap hoort bij het finale CSRNet-model,
geselecteerd op de laagste validatie-MAE op Part B.

Herhalen vanuit de repositoryroot:

```bash
MPLBACKEND=Agg bachproef/.venv/bin/python crowd_counting/example_praktijk.py
```

Het script verwacht de lokale bronbeelden als
`graphics/voorbeelden/grote-markt-sint-niklaas/origineel.png` en
`graphics/voorbeelden/trevifontein-rome/origineel.png`.
De bronbeelden blijven lokaal; de verkleinde foto's en heatmaps voor LaTeX
staan in de bijbehorende graphics-mappen. De volledige schermafbeelding van
Rome wordt gebruikt, inclusief logo's en reclame. Bron:
https://www.skylinewebcams.com/en/webcam/italia/lazio/roma/fontana-di-trevi.html
De gebruikte bestanden zijn vastgelegd met SHA-256 in `comparison.json`.

Elke modelmap bevat numerieke uitvoer en een JSON met de instellingen en
metingen. `comparison.json`, `comparison.csv` en `tabel.tex` bundelen de vijf
modellen. Nieuwe uitvoeringen vervangen deze resultaten. De oude experimenten
voor de Grote Markt blijven afzonderlijk bewaard.

De handmatige telling voor de Grote Markt bevat 738 onvolledige annotaties.
Voor Rome kan de telling later worden ingevuld in
`bachproef/voorbeelden/trevifontein-rome/handmatige-telling.tex`.
Pas na het invullen ook de bespreking aan die naar het ontbreken van een
referentietelling verwijst.

## Figurenraster opnieuw genereren

De extra panelen worden zonder nieuwe inferentie gemaakt met:

```bash
MPLBACKEND=Agg bachproef/.venv/bin/python crowd_counting/plot_praktijk.py
```

De handmatige hoofdposities worden als cyaanblauwe cirkels op de foto getoond.
De P2PNet-heatmap gebruikt de bestaande voorspelde punten, geschaald naar de
fotoweergave met maximale beeldzijde 1024. Een Gaussische kernel met sigma 10
pixels en reflecterende randen behoudt de som van de punten binnen het beeld.
Punten buiten het beeld worden apart geregistreerd; bij beide foto's zijn dat
er nul. De kleurenschaal wordt per heatmap bepaald en is niet rechtstreeks
vergelijkbaar met de CSRNet-kleurenschaal. De numerieke map en instellingen
staan bij de P2PNet-resultaten als `afgeleide_heatmap.npy` en `heatmap.json`.

Voor Rome reserveert LaTeX een vak voor de latere annotaties. Plaats een
`markers.csv` met hetzelfde formaat als de Grote Markt in
`bachproef/voorbeelden/trevifontein-rome/handmatige-telling/` en voer het
figuurscript opnieuw uit. Het gegenereerde bestand
`graphics/voorbeelden/trevifontein-rome/handmatige-hoofdposities.png` vervangt
het lege vak automatisch. Vul ook de telling en bespreking in de tekst aan.
