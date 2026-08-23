# Perplexica v1.12.1 runtime patch

Patch experimental pour Perplexica v1.12.1, build Docker `itzcrazykns1337/perplexica:slim-latest`.

Fichier runtime cible dans le conteneur :

```text
/home/perplexica/.next/server/chunks/641.js
```

## Fichiers

- `641.js.original` : copie intacte du chunk runtime extrait du conteneur avant instrumentation/correction.
- `641.js.diag` : chunk modifie, avec corrections fonctionnelles minimales et logs temporaires de diagnostic.
- `641.js.patch` : diff de reference entre l'original et la version diagnostique.
- `install_patch.sh` : installe `641.js.diag` dans le conteneur Perplexica actif.

## Corrections fonctionnelles

- Researcher : au premier tour uniquement, si la source `web` est demandee, si `skipSearch=false`, et avant tout resultat d'action, la liste de tools envoyee au modele est restreinte a `web_search`.
- Researcher : a partir du deuxieme tour, le comportement existant est conserve avec les tools disponibles (`web_search`, `scrape_url`, `done`, etc.).
- `web_search.execute()` : les mises a jour UI/progress via `session.updateBlock()` sont rendues conditionnelles.
- `web_search.execute()` : l'absence de `session` ou `updateBlock` n'empeche plus le retour des resultats SearXNG.
- `web_search.execute()` : les erreurs reelles de recherche/SearXNG ne sont pas masquees.

## Logs temporaires

Les logs suivants sont volontairement temporaires et doivent etre retires apres validation :

- `[perplexica_search_diag]` : classification, iteration Researcher, tool calls, execution des actions, resultats SearXNG, `searchFindings`.
- `[perplexica_updateblock_diag]` : etat de `session`, presence de `updateBlock`, `researchBlockId`, type/cles du block avant les mises a jour progress.

Filtrage recommande :

```bash
docker logs -f --tail 0 perplexica 2>&1 | grep -E 'perplexica_search_diag|perplexica_updateblock_diag'
```

## Installation

Depuis ce dossier sur l'hote Docker :

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

Ce patch est specifique a Perplexica v1.12.1 et au chunk runtime `641.js` de ce build. Ne pas l'appliquer a une autre version ou a une image reconstruite sans verifier le diff et relancer `node --check`.
