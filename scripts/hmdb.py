#!/usr/bin/env python3
"""
HMDB XML → SQLite database builder
====================================
Parses the HMDB bulk metabolite XML (hmdb_metabolites.xml) and builds a
queryable SQLite database with three tables:

  metabolites      – one row per HMDB entry (hmdb_id, name, kegg_id, chebi_id, …)
  diseases         – one row per disease association (hmdb_id → disease name/omim_id)
  pathways         – one row per pathway association (hmdb_id → pathway name/kegg_map_id/smpdb_id)

Usage
-----
  # 1. Download the full metabolite XML from https://hmdb.ca/downloads
  #    (the file is ~6 GB compressed; unzip it first)

  python3 hmdb_to_sqlite.py --xml hmdb_metabolites.xml --db hmdb.db

  # Then query it:
  python3 hmdb_to_sqlite.py --query HMDB0000001

Options
-------
  --xml    PATH   Path to unzipped hmdb_metabolites.xml   [required for build]
  --db     PATH   Output / existing SQLite file           [default: hmdb.db]
  --query  ID     HMDB accession to look up after build   [optional]
  --limit  N      Stop after N metabolites (useful for testing; 0 = no limit)
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

try:
    from lxml import etree
except ImportError:
    sys.exit("lxml is required: pip install lxml --break-system-packages")


# ---------------------------------------------------------------------------
# Namespace used in every HMDB XML file
# ---------------------------------------------------------------------------
NS = "http://www.hmdb.ca"
TAG = lambda name: f"{{{NS}}}{name}"  # noqa: E731


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text(el, path):
    """Return stripped text from the first child matching `path`, or ''."""
    child = el.find(path, namespaces={"h": NS})
    if child is not None and child.text:
        return child.text.strip()
    return ""


def _tag(name):
    return f"{{{NS}}}{name}"


def create_schema(conn):
    conn.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous  = NORMAL;

        CREATE TABLE IF NOT EXISTS metabolites (
            hmdb_id          TEXT PRIMARY KEY,
            name             TEXT,
            common_name      TEXT,
            description      TEXT,
            inchikey         TEXT,
            kegg_id          TEXT,   -- KEGG Compound id, e.g. C00001
            chebi_id         TEXT,
            pubchem_id       TEXT,
            drugbank_id      TEXT,
            bigg_id          TEXT,
            status           TEXT    -- "Quantified", "Expected", etc.
        );

        CREATE TABLE IF NOT EXISTS diseases (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            hmdb_id          TEXT NOT NULL REFERENCES metabolites(hmdb_id),
            disease_name     TEXT,
            omim_id          TEXT,
            references_text  TEXT
        );

        CREATE TABLE IF NOT EXISTS pathways (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            hmdb_id          TEXT NOT NULL REFERENCES metabolites(hmdb_id),
            pathway_name     TEXT,
            smpdb_id         TEXT,   -- PathBank/SMPDB id, e.g. SMP00001
            kegg_map_id      TEXT    -- KEGG map id, e.g. map00010
        );

        -- Indexes so lookups by hmdb_id are fast
        CREATE INDEX IF NOT EXISTS idx_diseases_hmdb  ON diseases(hmdb_id);
        CREATE INDEX IF NOT EXISTS idx_pathways_hmdb  ON pathways(hmdb_id);
        CREATE INDEX IF NOT EXISTS idx_diseases_omim  ON diseases(omim_id);
        CREATE INDEX IF NOT EXISTS idx_pathways_kegg  ON pathways(kegg_map_id);
        CREATE INDEX IF NOT EXISTS idx_pathways_smpdb ON pathways(smpdb_id);
        CREATE INDEX IF NOT EXISTS idx_metabolites_kegg ON metabolites(kegg_id);
    """)
    conn.commit()


def parse_and_load(xml_path: Path, db_path: Path, limit: int = 0):
    """Stream-parse the HMDB XML and insert into SQLite."""

    conn = sqlite3.connect(db_path)
    create_schema(conn)

    met_buf, dis_buf, path_buf = [], [], []
    BATCH = 2000
    count = 0
    t0 = time.time()

    print(f"Parsing {xml_path} …")

    context = etree.iterparse(str(xml_path), events=("end",),
                              tag=_tag("metabolite"),
                              recover=True)

    for _, el in context:
        hmdb_id = _text(el, f".//{_tag('accession')}")
        if not hmdb_id:
            el.clear()
            continue

        # ---- metabolite row ------------------------------------------------
        met_buf.append((
            hmdb_id,
            _text(el, f".//{_tag('name')}"),
            _text(el, f".//{_tag('common_name')}"),
            _text(el, f".//{_tag('description')}"),
            _text(el, f".//{_tag('inchikey')}"),
            _text(el, f".//{_tag('kegg_id')}"),
            _text(el, f".//{_tag('chebi_id')}"),
            _text(el, f".//{_tag('pubchem_compound_id')}"),
            _text(el, f".//{_tag('drugbank_id')}"),
            _text(el, f".//{_tag('bigg_id')}"),
            _text(el, f".//{_tag('status')}"),
        ))

        # ---- disease rows --------------------------------------------------
        diseases_el = el.find(_tag("diseases"))
        if diseases_el is not None:
            for d in diseases_el.findall(_tag("disease")):
                dis_buf.append((
                    hmdb_id,
                    _text(d, f".//{_tag('name')}"),
                    _text(d, f".//{_tag('omim_id')}"),
                    _text(d, f".//{_tag('references')}"),
                ))

        # ---- pathway rows --------------------------------------------------
        pathways_el = el.find(_tag("biological_properties"))
        if pathways_el is None:
            pathways_el = el   # fallback: search whole element
        for p in pathways_el.findall(f".//{_tag('pathway')}"):
            path_buf.append((
                hmdb_id,
                _text(p, f".//{_tag('name')}"),
                _text(p, f".//{_tag('smpdb_id')}"),
                _text(p, f".//{_tag('kegg_map_id')}"),
            ))

        count += 1

        # flush to DB in batches
        if len(met_buf) >= BATCH:
            _flush(conn, met_buf, dis_buf, path_buf)
            met_buf.clear(); dis_buf.clear(); path_buf.clear()
            elapsed = time.time() - t0
            print(f"  {count:>8,} metabolites loaded … ({elapsed:.0f}s)", end="\r")

        el.clear()          # free memory

        if limit and count >= limit:
            print(f"\nStopped at limit={limit}")
            break

    # flush remainder
    if met_buf:
        _flush(conn, met_buf, dis_buf, path_buf)

    conn.close()
    elapsed = time.time() - t0
    print(f"\nDone. {count:,} metabolites in {elapsed:.1f}s → {db_path}")


def _float(s):
    try:
        return float(s) if s else None
    except ValueError:
        return None


def _flush(conn, met_buf, dis_buf, path_buf):
    conn.executemany("""
        INSERT OR IGNORE INTO metabolites
          (hmdb_id, name, common_name, description, inchikey, kegg_id, chebi_id,
           pubchem_id, drugbank_id, bigg_id, status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, met_buf)

    conn.executemany("""
        INSERT INTO diseases (hmdb_id, disease_name, omim_id, references_text)
        VALUES (?,?,?,?)
    """, dis_buf)

    conn.executemany("""
        INSERT INTO pathways (hmdb_id, pathway_name, smpdb_id, kegg_map_id)
        VALUES (?,?,?,?)
    """, path_buf)

    conn.commit()


# ---------------------------------------------------------------------------
# Query helper
# ---------------------------------------------------------------------------

def query_metabolite(db_path: Path, hmdb_id: str):
    """Pretty-print diseases and pathways for one HMDB accession."""
    if not db_path.exists():
        sys.exit(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    met = conn.execute(
        "SELECT * FROM metabolites WHERE hmdb_id = ?", (hmdb_id,)
    ).fetchone()

    if not met:
        print(f"No entry found for {hmdb_id}")
        conn.close()
        return

    print(f"\n{'='*60}")
    print(f"  {met['hmdb_id']}  |  {met['name']}")
    print(f"{'='*60}")
    print(f"  Formula   : {met['formula']}")
    print(f"  KEGG ID   : {met['kegg_id']}")
    print(f"  ChEBI ID  : {met['chebi_id']}")
    print(f"  PubChem   : {met['pubchem_id']}")
    print(f"  Status    : {met['status']}")

    diseases = conn.execute(
        "SELECT disease_name, omim_id FROM diseases WHERE hmdb_id = ?",
        (hmdb_id,)
    ).fetchall()

    print(f"\n  DISEASES ({len(diseases)}):")
    if diseases:
        for d in diseases:
            omim = f"  [OMIM:{d['omim_id']}]" if d['omim_id'] else ""
            print(f"    • {d['disease_name']}{omim}")
    else:
        print("    (none recorded)")

    pathways = conn.execute(
        "SELECT pathway_name, kegg_map_id, smpdb_id FROM pathways WHERE hmdb_id = ?",
        (hmdb_id,)
    ).fetchall()

    print(f"\n  PATHWAYS ({len(pathways)}):")
    if pathways:
        for p in pathways:
            kegg = f"  [KEGG:{p['kegg_map_id']}]" if p['kegg_map_id'] else ""
            smpdb = f"  [SMPDB:{p['smpdb_id']}]" if p['smpdb_id'] else ""
            print(f"    • {p['pathway_name']}{kegg}{smpdb}")
    else:
        print("    (none recorded)")

    conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build a SQLite database from HMDB bulk XML, then query it."
    )
    parser.add_argument("--xml",   help="Path to hmdb_metabolites.xml")
    parser.add_argument("--db",    default="hmdb.db", help="SQLite output file [hmdb.db]")
    parser.add_argument("--query", help="HMDB accession to look up (e.g. HMDB0000001)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N metabolites (0 = all; handy for testing)")
    args = parser.parse_args()

    db_path = Path(args.db)

    if args.xml:
        xml_path = Path(args.xml)
        if not xml_path.exists():
            sys.exit(f"XML file not found: {xml_path}")
        parse_and_load(xml_path, db_path, limit=args.limit)

    if args.query:
        query_metabolite(db_path, args.query)

    if not args.xml and not args.query:
        parser.print_help()


if __name__ == "__main__":
    main()