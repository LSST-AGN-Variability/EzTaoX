import logging
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory

from astropy.config import set_temp_cache, get_cache_dir
from astropy.table import Table
from astroquery.vizier import Vizier
from dask.distributed import Client
from hats_import.catalog.arguments import ImportArguments
from hats_import.margin_cache.margin_cache_arguments import MarginCacheArguments
from hats_import.pipeline import pipeline_with_client


def cache_dir(path: Path | None):
    if path is None:
        @contextmanager
        def dummy_context():
            yield get_cache_dir()
        
        return dummy_context
    return set_temp_cache(path, delete=False)


def download_vizier_table(
        table_id: str,
        *,
        astroquery_cache_path: Path | None = None,
        **vizier_kwargs,
) -> Table:
    vizier = Vizier(**vizier_kwargs)
    vizier.ROW_LIMIT = -1  # Removes row limit
    vizier.TIMEOUT = 600  # Set timeout to a sufficient value for large tables
    
    with cache_dir(astroquery_cache_path):
        tables = vizier.get_catalogs(table_id)
    
    if len(tables) != 1:
        raise ValueError(f"{len(tables)} where found for '{table_id}', single table expected")
    
    table = tables[0]
    return table


def hats_import_main_catalog(
    input_file_list: list[Path],
    hats_path: Path,
    hats_name: str,
    dask_client_kwargs: dict | None,
) -> None:
    args = ImportArguments(
        ra_column="_RAJ2000",
        dec_column="_DEJ2000",
        input_file_list=input_file_list,
        file_reader="parquet",
        output_artifact_name=hats_name,
        output_path=hats_path,
        pixel_threshold=100_000,
    )
    
    dask_client_kwargs = dask_client_kwargs or {"n_workers": 4}
    with Client(**dask_client_kwargs) as client:
        logging.info(str(client))
        pipeline_with_client(args, client)
        

def hats_import_margin_catalog(
    *,
    main_catalog_path: Path,
    main_catalog_name: str,
    margin_threshold_arcsec: int,
    dask_client_kwargs: dict | None,
) -> bool:
    margin_catalog_name = f"{main_catalog_name}_{margin_threshold_arcsec}arcsec"
    args = MarginCacheArguments(
        input_catalog_path=main_catalog_path,
        output_path=main_catalog_path.parent,
        margin_threshold=float(margin_threshold_arcsec),
        output_artifact_name=margin_catalog_name,
        fine_filtering=False,
    )
    
    dask_client_kwargs = dask_client_kwargs or {"n_workers": 4}
    with Client(**dask_client_kwargs) as client:
        logging.info(str(client))
        try:
            pipeline_with_client(args, client)
        except ValueError as e:
            if "Margin cache contains no rows" in str(e):
                margin_path = main_catalog_path.parent / margin_catalog_name
                rmtree(margin_path)
                return False
            raise
    return True


def hats_import(
        *,
        input_file_list: list[Path],
        hats_path: Path,
        hats_name: str,
        margin_threshold_arcsec: int,
        dask_client_kwargs: dict | None,
) -> bool:
    
    hats_import_main_catalog(
        input_file_list=input_file_list,
        hats_path=hats_path,
        hats_name=hats_name,
        dask_client_kwargs=dask_client_kwargs,
    )

    main_catalog_path = hats_path / hats_name
    if not main_catalog_path.exists():
        raise FileNotFoundError(f"{main_catalog_path} does not exist")
    
    
    if not hats_import_margin_catalog(
            main_catalog_path=main_catalog_path,
            main_catalog_name=hats_name,
            margin_threshold_arcsec=margin_threshold_arcsec,
            dask_client_kwargs=dask_client_kwargs,
    ):
        logging.warning("Margin cache is empty and was not generated")


def vizier_to_hats(
        table_id: str,
        *,
        hats_path: Path,
        hats_name: str,
        margin_threshold_arcsec: int = 10,
        tmp_path: Path | None = None,
        cache_path: Path | None = None,
        vizier_kwargs: dict | None = None,
        dask_client_kwargs: dict | None = None,
) -> None:
    vizier_kwargs = deepcopy(vizier_kwargs) or {}
    vizier_kwargs.setdefault("columns", ["*"]).extend(["_RAJ2000", "_DEJ2000"])
    logging.info(f"Getting Vizier table '{table_id}'")
    vizier_table = download_vizier_table(
        table_id,
        astroquery_cache_path=cache_path,
        **vizier_kwargs,
    )
    
    with TemporaryDirectory(dir=tmp_path) as temp_dir:
        temp_dir = Path(temp_dir)
        parquet_path = temp_dir / "file.parquet"
        logging.info(f"Saving Vizier table '{table_id}' to '{parquet_path}'")
        vizier_table.write(parquet_path, format="parquet")
        
        # hats_name = hats_name or vizier_table.meta["name"]
        logging.info(f"Hats-importing Vizier table '{table_id}' to {hats_path}")
        hats_import(
            input_file_list=[parquet_path],
            hats_path=hats_path,
            hats_name=hats_name,
            margin_threshold_arcsec=margin_threshold_arcsec,
            dask_client_kwargs=dask_client_kwargs,
        )
