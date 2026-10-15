# Handmatige telling van de Grote Markt

`markers.csv` en `summary.csv` zijn ongewijzigd overgenomen uit de aangeleverde `Grote markt.zip`. De 738 unieke puntannotaties komen overeen met het totaal in de samenvatting en het aantal markers in `Grote markt.html`. De oorspronkelijke foto in de HTML-export is byte voor byte gelijk aan `graphics/voorbeelden/grote-markt-sint-niklaas/origineel.png`. `herkomst.json` bewaart de SHA-256-hashes van de exports en de foto. De grote HTML-export en de JPEG met ingetekende markeringen worden niet gedupliceerd in de repository.

De auteur vermeldt dat hoofden in dichte groepen, soms alleen te onderscheiden door haarkleur, en personen achter de boom niet volledig zijn aangeduid. Het totaal is daarom een onvolledige, conservatieve referentie. De status `completed=true` in de export betekent niet dat ieder zichtbaar hoofd is gemarkeerd. Ook het tijdstip `image_captured_at` is overgenomen exportmetadata en wordt niet gebruikt als geverifieerde opnamedatum.

Reproduceer de figuur en de afgeleide cijfers vanuit de repositoryroot:

```sh
python crowd_counting/plot_manual_grote_markt.py
```

Benodigd: NumPy, SciPy, Pillow en Matplotlib, ook opgenomen in `crowd_counting/requirements.txt`. Het script gebruikt de genormaliseerde CSV-coördinaten, controleert aantallen en beeldafmetingen en maakt een raster van 1024 × 683 pixels. Elke annotatie draagt één eenheid bij. Een Gaussisch filter met sigma 10 pixels en reflecterende randen behoudt de som van 738. De weergegeven heatmap is gedeeld door haar maximum; kleuren zijn niet rechtstreeks vergelijkbaar met de afzonderlijk geschaalde modelheatmap en drukken geen personen per vierkante meter uit. `analyse.json` bevat de afgeleide modelafwijkingen en figuurparameters.
