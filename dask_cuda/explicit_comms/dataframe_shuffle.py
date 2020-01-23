import asyncio
import uuid
import numpy as np
import pandas

import rmm
import cudf
from distributed.protocol import to_serialize

from . import comms, utils


async def send_df(ep, df):
    if df is None:
        return await ep.write("empty")
    else:
        return await ep.write([to_serialize(df)])


async def recv_df(ep):
    ret = await ep.read()
    if ret == "empty":
        return None
    else:
        return ret[0]


async def send_parts(eps, parts):
    futures = []
    for rank, ep in eps.items():
        futures.append(send_df(ep, parts[rank]))
    await asyncio.gather(*futures)


async def recv_parts(eps, parts):
    futures = []
    for ep in eps.values():
        futures.append(recv_df(ep))
    parts.extend(await asyncio.gather(*futures))


async def exchange_and_concat_parts(rank, eps, parts, sort=False):
    ret = [parts[rank]]
    await asyncio.gather(recv_parts(eps, ret), send_parts(eps, parts))
    df = concat(list(filter(None, ret)))
    if sort:
        return df.sort_values(sort)
    return df


def concat(df_list):
    if len(df_list) == 0:
        return None
    return cudf.concat(df_list)


def partition_by_column(df, column, n_chunks):
    """Partition the dataframe by the hashed value of data in column.
        Supports both Pandas and cuDF DataFrames
    """
    if df is None:
        return [None] * n_chunks
    elif hasattr(df, "scatter_by_map"):
        return df.scatter_by_map(column, map_size=n_chunks)
    else:
        raise NotImplementedError(
            "partition_by_column not yet implemented for pandas backend.\n"
        )


async def distributed_rearrange_and_set_index(
    n_chunks, rank, eps, table, partitions, index, drop, scatter
):
    if scatter:
        parts = partition_by_column(table, partitions, n_chunks)
    else:
        parts = [
            table.iloc[partitions[i]:partitions[i+1]]
            for i in range(0,len(partitions)-1)
        ]
    df = await exchange_and_concat_parts(
        rank, eps, parts, sort=index
    )
    return df.set_index(index, drop=drop)


async def _set_index(
    s, df_parts, index, divisions, drop, scatter
):
    def df_concat(df_parts):
        """Making sure df_parts is a single dataframe or None"""
        if len(df_parts) == 0:
            return None
        elif len(df_parts) == 1:
            return df_parts[0]
        else:
            return concat(df_parts)

    # Concatenate all parts owned by this worker into
    # a single cudf DataFrame
    df = df_concat(df_parts)

    divisions = cudf.Series(divisions)
    if not scatter:
        df = df.sort_values(index)
        splits = df[index].searchsorted(divisions, side="left")
        splits[-1]+=1
        partitions = splits.tolist()
    else:
        partitions = divisions.searchsorted(df[index], side="right") - 1
        partitions[(df[index] >= divisions.iloc[-1]).values] = len(divisions) - 2
    
    # Run distributed shuffle and set_index algorithm
    return await distributed_rearrange_and_set_index(
        s["nworkers"], s["rank"], s["eps"], df, partitions, index, drop, scatter
    )


def dataframe_set_index(
    df,
    index,
    npartitions=None,
    shuffle="tasks",
    compute=False,
    drop=True,
    n_workers=None,
    divisions=None,
    scatter=True,
    **kwargs
):

    if not isinstance(index, str):
        raise NotImplementedError(
            "Index must be column name (for now).\n"
        )

    if divisions:
        raise NotImplementedError(
            "Cannot accept divisions argument (for now).\n"
        )

    if n_workers == None:
        raise NotImplementedError(
            "Must provide n_workers to calculate divisions.\n"
        )

    if shuffle != "tasks":
        raise NotImplementedError(
            "Task-based shuffle is required for explicit comms.\n"
        )

    if compute == True:
        raise NotImplementedError(
            "compute=True argument not supported (row now).\n"
        )

    if npartitions and npartitions != df.npartitions:
        raise NotImplementedError(
            "Input partitions must equal output partitions (for now).\n"
        )
    npartitions = df.npartitions

    # Calculate divisions for n_workers (not npartitions)
    divisions = df[index]._repartition_quantiles(
        n_workers, upsample=1.0
    ).compute().to_list()

    # Explict-comms shuffle and local set_index
    df_out = comms.default_comms().dataframe_operation(
        _set_index,
        df_list=(df,),
        extra_args=(index, divisions, drop, scatter)
    )

    # Final repartition (should be fast - intra-worker partitioning)
    if n_workers != npartitions:
        return df_out.repartition(npartitions=npartitions)
    return df_out
