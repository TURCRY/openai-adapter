# Perplexica v1.12.1 runtime patch

Archive du patch runtime valide en production pour Perplexica 1.12.1, image Docker `itzcrazykns1337/perplexica:slim-latest`.

Runtime cible dans le conteneur :

```text
/home/perplexica/.next/server/chunks/641.js
```

## Fichiers

- `641.js.original` : copie intacte du chunk runtime upstream extrait du conteneur avant instrumentation/correction.
- `641.js.diag` : chunk runtime valide, incluant correctifs fonctionnels et diagnostics temporaires.
- `641.js.patch` : diff de reference entre `641.js.original` et `641.js.diag`.
- `install_patch.sh` : script d'installation/restauration operationnelle du patch dans le conteneur `perplexica`.

## Corrections fonctionnelles

- Summary URL / Discover : au premier tour Researcher, une requete contenant une URL HTTP/HTTPS force temporairement le tool `scrape_url` si disponible, avant la regle `web_search`.
- Recherche web classique : au premier tour Researcher, si la source `web` est demandee, `skipSearch=false`, et aucun resultat/action n'existe encore, le premier tool reste force a `web_search`.
- A partir du deuxieme tour Researcher, tous les tools disponibles sont restaures.
- `web_search.execute()` : les mises a jour UI/progress via `session.updateBlock()` sont conditionnelles afin que l'absence de session/updateBlock ne bloque pas le retour des resultats.
- `scrape_url.execute()` : nettoyage HTML conservateur avant Turndown (`script`, `style`, `noscript`, `svg`, `iframe`, `nav`, `footer`, `header`, `aside`, `form`).
- `scrape_url.execute()` : extraction prioritaire de `<article>`, puis `<main>`, puis fallback sur HTML nettoye complet.
- `scrape_url.execute()` : garde-fous de contexte avec 24000 caracteres maximum par URL et 36000 caracteres maximum cumules.
- `scrape_url.execute()` : support premium pour `lemonde.fr`, `*.lemonde.fr`, `lesechos.fr`, `*.lesechos.fr` via scraper externe, avec fallback standard automatique.

## Scraper premium

Variables runtime requises pour activer le scraping premium :

```text
PREMIUM_SCRAPER_URL
PREMIUM_SCRAPER_API_KEY
```

Valeur attendue pour l'URL :

```text
PREMIUM_SCRAPER_URL=http://10.0.1.10:5050/scrape_premium
```

Ne jamais archiver ni documenter la valeur reelle de `PREMIUM_SCRAPER_API_KEY`.

Domaines premium actuels :

- `lemonde.fr`
- `*.lemonde.fr`
- `lesechos.fr`
- `*.lesechos.fr`

Le scraper premium est considere comme reussi uniquement si la reponse est HTTP 2xx, JSON parseable, `ok === true`, et `content` est une chaine non vide. Dans tous les autres cas, le scraping standard existant est utilise immediatement.

Aucune cle API n'est loggee. Les headers complets ne sont jamais logges.

## Garde-fous de contexte

- 24000 caracteres maximum par URL.
- 36000 caracteres maximum au total pour un appel `scrape_url`.
- Suffixe ajoute en cas de troncature : `[Content truncated for context safety]`.
- `metadata.url` et `metadata.title` sont conserves.
- Les plafonds s'appliquent aussi au contenu premium, par defense en profondeur.

## Logs temporaires

Les logs suivants sont volontairement temporaires et doivent etre retires apres validation definitive :

- `[perplexica_search_diag]` : classification, iteration Researcher, tool calls, execution des actions, resultats SearXNG, `searchFindings`.
- `[perplexica_summary_diag]` : chemin Summary URL, `scrape_url`, extraction HTML, tailles nettoyees/retournees, troncature.
- `[perplexica_writer_diag]` : evenements SessionManager, creation/update de blocks, emissions vers l'UI.
- `[perplexica_updateblock_diag]` : etat de `session`, presence de `updateBlock`, `researchBlockId`, type/cles du block avant les mises a jour progress.
- `[perplexica_premium_diag]` : detection domaine premium, tentative backend premium, succes ou fallback standard sans secrets.

Filtrage recommande :

```bash
docker logs -f --tail 0 perplexica 2>&1 | grep -E 'perplexica_search_diag|perplexica_summary_diag|perplexica_writer_diag|perplexica_updateblock_diag|perplexica_premium_diag'
```

## Injection

Depuis ce dossier sur l'hote Docker :

```bash
docker cp 641.js.diag perplexica:/home/perplexica/.next/server/chunks/641.js
docker restart perplexica
```

Ou avec le script fourni :

```bash
chmod +x install_patch.sh
./install_patch.sh
```

Le script :

1. verifie que le conteneur `perplexica` existe ;
2. sauvegarde le chunk runtime courant avec un timestamp ;
3. copie `641.js.diag` vers `/home/perplexica/.next/server/chunks/641.js` ;
4. redemarre le conteneur `perplexica` ;
5. affiche les dernieres lignes de log.

## Restauration

Lister les sauvegardes creees dans le conteneur :

```bash
docker exec perplexica sh -lc 'ls -lh /home/perplexica/.next/server/chunks/641.js.backup-*'
```

Restaurer une sauvegarde :

```bash
docker exec perplexica sh -lc 'cp /home/perplexica/.next/server/chunks/641.js.backup-YYYYmmdd-HHMMSS /home/perplexica/.next/server/chunks/641.js'
docker restart perplexica
```

Restauration depuis l'original conserve dans ce dossier :

```bash
docker cp 641.js.original perplexica:/home/perplexica/.next/server/chunks/641.js
docker restart perplexica
```

## Avertissement

Ce patch est specifique a Perplexica 1.12.1 et au chunk runtime `641.js` de ce build. Ne pas l'appliquer a une autre version ou a une image reconstruite sans verifier le diff et relancer `node --check`.
