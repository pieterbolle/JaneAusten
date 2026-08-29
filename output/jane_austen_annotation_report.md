# Samenvattend rapport automatische annotatie Jane Austen

## Reikwijdte en beperking

Dit rapport bundelt de technisch verifieerbare resultaten voor **Lady Susan**, **Sense and Sensibility**, **Northanger Abbey**, **Emma** en **Mansfield Park**. Voor **Persuasion** was in de huidige bijlage geen XML- of auditbestand aanwezig. Daarom is voor dat werk wel een voorzichtig normalisatiewoordenboek voorbereid, maar nog geen kwantitatieve resultaatanalyse uitgevoerd.

## Kernresultaten per werk

| Werk | Paragrafen | Voorspelde turns | Turns in XML | Mapping | Interruptions | Review aanbevolen | Conf. < 0,50 | Indirect=yes | Malformed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Lady Susan | 287 | 54 | 50 | 92,6% | 4 | 8 (16,0%) | 8 (16,0%) | 0 | 0 |
| Sense and Sensibility | 1.923 | 1.116 | 1.002 | 89,8% | 373 | 291 (29,0%) | 246 (24,6%) | 3 | 9 |
| Northanger Abbey | 1.122 | 906 | 860 | 94,9% | 161 | 342 (39,8%) | 286 (33,3%) | 3 | 0 |
| Emma | 2.440 | 1.638 | 1.482 | 90,5% | 472 | 409 (27,6%) | 320 (21,6%) | 9 | 3 |
| Mansfield Park | 1.905 | 1.264 | 1.136 | 89,9% | 340 | 372 (32,7%) | 302 (26,6%) | 5 | 8 |
| Persuasion | niet beschikbaar | niet beschikbaar | niet beschikbaar | niet beschikbaar | niet beschikbaar | niet beschikbaar | niet beschikbaar | niet beschikbaar | niet beschikbaar |

### Totaal voor de vijf beschikbare werken

- 7.677 verwerkte paragrafen.
- 4.978 door het model voorspelde turns.
- 4.530 succesvol in de originele tekst geprojecteerde turns.
- Globaal mappingsucces: **91,0%**.
- 1.350 geneste interruption-elementen.
- 1.422 turns met `review_recommended="true"`, of **31,4%** van de opgeslagen turns.
- 1.162 turns onder confidence 0,50, of **25,7%**.
- Slechts 20 turns met `indirectspeech="yes"`.

## Beoordeling per werk

### Lady Susan

De run is technisch stabiel, maar het epistolaire genre past slecht bij een paragraafgebaseerde dialoogtaak. Slechts delen van brieven worden als turn gemarkeerd, terwijl schrijver en geadresseerde via het briefhoofd over meerdere paragrafen geldig blijven. Dit werk vraagt daarom een afzonderlijke deterministische briefparser. De 50 turns mogen niet als volledige dekking van de briefcommunicatie worden geïnterpreteerd.

### Sense and Sensibility

Dit is een bruikbare automatische voorannotatie. De pipeline schaalt naar bijna tweeduizend paragrafen en meer dan duizend gemapte turns. De 373 interruptions tonen dat nested XML technisch goed wordt opgebouwd. De mappingscore is met 89,8% wel de laagste van de vijf onderzochte werken. Mapping failures moeten dus systematisch in de review worden opgenomen.

### Northanger Abbey

Dit werk heeft het beste mappingsucces, 94,9%, maar de zwakste semantische zekerheid. Bijna 40% van de turns wordt voor review aanbevolen en een derde ligt onder confidence 0,50. Vooral `Unknown`, pronouns en varianten als `Henry`, `Mr. Tilney` en `Eleanor` tonen dat turnherkenning en entiteitsresolutie verschillende problemen zijn.

### Emma

Emma levert de grootste dataset: 1.482 gemapte turns en 472 interruptions. De globale confidence is relatief gunstig, met een mediaan van ongeveer 0,679 en 21,6% onder 0,50. Toch bevat de uitvoer duidelijke contextfouten. Reeds in het begin worden korte opeenvolgende replieken soms aan `Unknown` toegewezen of aan de verkeerde spreker. Ook worden gesproken woorden soms foutief als interruption aangemerkt. Het boek bevestigt dat drie voorgaande paragrafen onvoldoende kunnen zijn bij veel korte beurtwisselingen.

### Mansfield Park

De verwerking is technisch geslaagd, met 1.136 gemapte turns en 340 interruptions. De mediaan van de confidence bedraagt ongeveer 0,651. Het werk bevat relatief veel relationele en generieke labels zoals `he`, `cousin`, `aunt`, `the other` en `the group`. Sommige mapping failures ontstaan omdat het model indirect weergegeven tekst herschrijft in de eerste persoon of reporting clauses vervangt. Dit is precies het type fout dat de bronbehoudende projectie veilig opvangt.

## Corpusbrede conclusies

1. **Turndetectie en XML-projectie zijn het sterkste onderdeel.** Over vijf werken werd 91,0% van de voorspelde turns exact genormaliseerd op de oorspronkelijke tekst teruggevonden.
2. **Speaker- en addressee-identificatie zijn de voornaamste inhoudelijke bottleneck.** Naamvarianten, titels, familienamen, relationele labels, pronouns en `Unknown` versnipperen de entiteiten.
3. **Een vaste context van drie paragrafen is niet optimaal.** Bij korte dialoogparagrafen bevat die context te weinig conversationele geschiedenis. Een volgend systeem gebruikt beter een minimumcontext van bijvoorbeeld 400 tot 600 woorden of een tokenbudget, met een begrensd maximum.
4. **Indirect speech is sterk ondergedetecteerd.** Twintig positieve gevallen over 4.530 gemapte turns is geen geloofwaardige exhaustieve dekking. `yes` kan als positief signaal worden gebruikt, maar `no` niet als sterk bewijs van afwezigheid.
5. **Interruption-detectie is technisch beschikbaar, maar semantisch niet steeds correct.** De output bevat goede reporting clauses, maar ook nadrukwoorden of gesproken zinsdelen die foutief als interruption worden gemarkeerd.
6. **Genre vraagt om hybride regels.** Lady Susan vereist structurele briefherkenning; de gewone romans profiteren vooral van contextgebaseerde entity resolution.
7. **Confidence is bruikbaar voor prioritering, niet als gekalibreerde foutkans.** Een efficiënte review combineert confidence met fouttypes: mapping failures, `Unknown`, pronouns, generieke labels, indirect speech en afwijkende interruptions.

## Aanbevolen vervolgstappen

1. Pas exacte, boekspecifieke aliasnormalisatie toe, maar bewaar altijd `S_raw` en `A_raw`.
2. Laat ambigue labels zoals `Mr. Bertram`, `Tilney`, `Mr. Ferrars`, `Mr. Musgrove`, `aunt`, `cousin`, `he` en `she` ongemoeid voor contextuele of menselijke resolutie.
3. Maak per boek een lijst van onopgeloste labels en hun frequenties. Die lijst is de basis voor een tweede, gecontroleerde uitbreiding van het woordenboek.
4. Behandel alle mapping failures als verplichte reviewgevallen.
5. Trek gestratificeerde steekproeven uit confidencebanden, bijvoorbeeld `<0,30`, `0,30-0,50`, `0,50-0,70`, `0,70-0,90` en `>=0,90`.
6. Laat onderzoekers trainingsvoorbeelden selecteren over alle werken en genres heen, niet uit één volledig boek. Neem bewust korte beurtwisselingen, groepsgesprekken, indirect speech, free indirect speech, brieven en complexe interruptions op.
7. Gebruik in een volgende trainingsronde dezelfde contextstrategie in training en inference, maar met een token- of woordbudget in plaats van exact drie paragrafen.
8. Ontwikkel voor Lady Susan een aparte epistolaire parser die briefhoofd, schrijver, geadresseerde, brieflichaam en ondertekening herkent.

## Normalisatiebeleid

Het bijgeleverde woordenboek bevat alleen aliases die redelijk veilig exact naar één canonieke entiteit kunnen worden vertaald. Ambigue familienamen en relationele termen zijn bewust niet systematisch opgenomen. De postprocessor wijzigt nooit het bronbestand, bewaart gewijzigde waarden in `S_raw` en `A_raw`, en produceert een afzonderlijke audit en samenvatting.
